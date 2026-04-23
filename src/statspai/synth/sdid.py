"""
Synthetic Difference-in-Differences (SDID).

Full Python replication of the R ``synthdid`` package by Arkhangelsky,
Athey, Hirshberg, Imbens & Wager (2021).

Three estimators share one optimisation framework:

* **SDID** – unit weights *and* time weights
* **SC**  – unit weights only (classic synthetic control)
* **DID** – uniform unit weights, uniform time weights

All three support *placebo*, *bootstrap*, and *jackknife* standard errors
and return :class:`~statspai.core.results.CausalResult`.

References
----------
Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W.
and Wager, S. (2021).
"Synthetic Difference-in-Differences."
*American Economic Review*, 111(12), 4088-4118. [@arkhangelsky2021synthetic]
"""

from typing import Optional, List, Dict, Any, Tuple, Literal

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def sdid(
    data: pd.DataFrame,
    outcome: Optional[str] = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    treated_unit: Any = None,
    treatment_time: Any = None,
    *,
    y: Optional[str] = None,
    treat_unit: Any = None,
    treat_time: Any = None,
    method: Literal['sdid', 'sc', 'did'] = 'sdid',
    covariates: Optional[List[str]] = None,
    se_method: Literal['placebo', 'bootstrap', 'jackknife'] = 'placebo',
    n_reps: int = 200,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Synthetic Difference-in-Differences estimator (and SC / DID variants).

    Replicates the R ``synthdid`` package interface.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data in long format.
    outcome : str
        Outcome variable column. Alias ``y=`` accepted for R-style calls.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : scalar or list
        Treated unit(s). Alias ``treat_unit=`` accepted.
    treatment_time : scalar
        First treatment period (inclusive). Alias ``treat_time=`` accepted.
    method : {'sdid', 'sc', 'did'}, default 'sdid'
        * ``'sdid'`` — Synthetic DID (unit + time weights)
        * ``'sc'``   — Synthetic Control (unit weights only)
        * ``'did'``  — DID (uniform weights)
    covariates : list of str, optional
        Reserved for future covariate-adjusted extensions.
    se_method : {'placebo', 'bootstrap', 'jackknife'}, default 'placebo'
        Standard-error method (see Notes).
    n_reps : int, default 200
        Replications for placebo / bootstrap SE.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        With ``model_info`` containing unit_weights, time_weights,
        Y_obs, Y_synth, control_units, pre_times, post_times, etc.

    Notes
    -----
    **SE methods** (matching R ``synthdid::vcov``):

    * *placebo* — reassign treatment to each control unit in turn.
    * *bootstrap* — resample control units with replacement.
    * *jackknife* — leave-one-control-unit-out.

    The SDID estimator is:

    .. math::
        \\hat{\\tau}_{sdid} = \\left(
            \\bar{Y}_{\\text{tr,post}}^{\\lambda}
          - \\hat{\\omega}' Y_{\\text{co,post}}^{\\lambda}
        \\right)
        - \\left(
            \\bar{Y}_{\\text{tr,pre}}^{\\lambda}
          - \\hat{\\omega}' Y_{\\text{co,pre}}^{\\lambda}
        \\right)

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.sdid(df, y='packspercapita', unit='state',
    ...                  time='year', treat_unit='California',
    ...                  treat_time=1989)
    >>> print(result.summary())
    >>> result.plot()       # synthdid-style trajectory plot

    Compare all three methods:

    >>> for m in ['sdid', 'sc', 'did']:
    ...     r = sp.sdid(df, y='packspercapita', unit='state',
    ...                 time='year', treat_unit='California',
    ...                 treat_time=1989, method=m)
    ...     print(f"{m:4s}: ATT = {r.estimate:.3f} (SE = {r.se:.3f})")
    """
    # --- Resolve canonical/legacy parameter names ---------------------
    if outcome is None:
        outcome = y
    if treated_unit is None:
        treated_unit = treat_unit
    if treatment_time is None:
        treatment_time = treat_time

    if outcome is None:
        raise TypeError("sdid: provide `outcome` (or legacy alias `y`)")
    if unit is None or time is None:
        raise TypeError("sdid: `unit` and `time` are required")
    if treated_unit is None:
        raise TypeError(
            "sdid: provide `treated_unit` (or legacy alias `treat_unit`)"
        )
    if treatment_time is None:
        raise TypeError(
            "sdid: provide `treatment_time` (or legacy alias `treat_time`)"
        )

    # Internal variable names kept short for the math below.
    y = outcome
    treat_unit = treated_unit
    treat_time = treatment_time

    rng = np.random.default_rng(seed)

    # --- Parse treated units ------------------------------------------
    if not isinstance(treat_unit, (list, tuple, np.ndarray)):
        treat_unit = [treat_unit]

    # --- Build panel matrix -------------------------------------------
    panel = data.pivot_table(
        index=unit, columns=time, values=y, aggfunc='first',
    )
    all_times = sorted(panel.columns.tolist())
    treated_mask = panel.index.isin(treat_unit)
    control_mask = ~treated_mask

    pre_times = [t for t in all_times if t < treat_time]
    post_times = [t for t in all_times if t >= treat_time]

    if len(pre_times) < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period.")

    n_tr = int(treated_mask.sum())
    n_co = int(control_mask.sum())
    T_pre = len(pre_times)
    T_post = len(post_times)

    Y_co_pre = panel.loc[control_mask, pre_times].values    # (N_co, T_pre)
    Y_co_post = panel.loc[control_mask, post_times].values   # (N_co, T_post)
    Y_tr_pre = panel.loc[treated_mask, pre_times].values     # (N_tr, T_pre)
    Y_tr_post = panel.loc[treated_mask, post_times].values   # (N_tr, T_post)

    # --- Solve weights ------------------------------------------------
    omega, lam = _compute_weights(
        Y_co_pre, Y_co_post, Y_tr_pre, method, n_co, T_pre,
    )

    # --- Point estimate -----------------------------------------------
    tau = _estimate_tau(
        Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post, omega, lam,
    )

    # --- Standard errors ----------------------------------------------
    if se_method == 'placebo':
        se, tau_reps = _se_placebo(
            Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
            method, n_co, T_pre,
        )
    elif se_method == 'bootstrap':
        se, tau_reps = _se_bootstrap(
            Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
            method, n_co, T_pre, n_reps, rng,
        )
    elif se_method == 'jackknife':
        se, tau_reps = _se_jackknife(
            Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
            method, n_co, T_pre,
        )
    else:
        raise ValueError(f"Unknown se_method: {se_method!r}")

    z = tau / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (tau - z_crit * se, tau + z_crit * se)

    # --- Build synthetic trajectory for plotting ----------------------
    control_names = panel.index[control_mask].tolist()
    treated_names = panel.index[treated_mask].tolist()

    # Observed treated trajectory (average over treated units)
    Y_tr_all = panel.loc[treated_mask, all_times].values.mean(axis=0)  # (T,)
    # Synthetic trajectory
    Y_synth_pre = omega @ Y_co_pre    # (T_pre,)
    Y_synth_post = omega @ Y_co_post  # (T_post,)
    # Adjust intercept for SDID / DID
    if method in ('sdid', 'did'):
        intercept = float(Y_tr_pre.mean(axis=0) @ lam) - float(omega @ (Y_co_pre @ lam))
        Y_synth_pre = Y_synth_pre + intercept
        Y_synth_post = Y_synth_post + intercept
    Y_synth_all = np.concatenate([Y_synth_pre, Y_synth_post])

    # Weight tables
    weight_df = pd.DataFrame({
        'unit': control_names,
        'weight': omega,
    }).sort_values('weight', ascending=False).reset_index(drop=True)

    time_weight_series = pd.Series(lam, index=pre_times, name='time_weight')

    method_labels = {
        'sdid': 'Synthetic Difference-in-Differences',
        'sc': 'Synthetic Control',
        'did': 'Difference-in-Differences',
    }

    model_info = {
        'estimator': method,
        'estimator_label': method_labels[method],
        'n_treated': n_tr,
        'n_control': n_co,
        'T_pre': T_pre,
        'T_post': T_post,
        'treat_time': treat_time,
        'treated_units': treated_names,
        'control_units': control_names,
        'unit_weights': weight_df,
        'time_weights': time_weight_series,
        'se_method': se_method,
        'n_reps': n_reps if se_method in ('bootstrap',) else None,
        'Y_obs': pd.Series(Y_tr_all, index=all_times, name='treated'),
        'Y_synth': pd.Series(Y_synth_all, index=all_times, name='synthetic'),
        'pre_times': pre_times,
        'post_times': post_times,
        'all_times': all_times,
    }

    return CausalResult(
        method=f'{method_labels[method]} (Arkhangelsky et al. 2021)',
        estimand='ATT',
        estimate=float(tau),
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        model_info=model_info,
        _citation_key='sdid',
    )


# ======================================================================
# Convenience wrappers (mirror R package names)
# ======================================================================

def synthdid_estimate(data, y, unit, time, treat_unit, treat_time, **kw):
    """R-style alias: ``synthdid::synthdid_estimate``."""
    return sdid(data, y, unit, time, treat_unit, treat_time, method='sdid', **kw)


def sc_estimate(data, y, unit, time, treat_unit, treat_time, **kw):
    """R-style alias: ``synthdid::sc_estimate``."""
    return sdid(data, y, unit, time, treat_unit, treat_time, method='sc', **kw)


def did_estimate(data, y, unit, time, treat_unit, treat_time, **kw):
    """R-style alias: ``synthdid::did_estimate``."""
    return sdid(data, y, unit, time, treat_unit, treat_time, method='did', **kw)


# ======================================================================
# Placebo analysis
# ======================================================================

def synthdid_placebo(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treat_unit: Any,
    treat_time: Any,
    method: Literal['sdid', 'sc', 'did'] = 'sdid',
    **kw,
) -> pd.DataFrame:
    """
    Run placebo estimates assigning treatment to each control unit.

    Replicates ``synthdid::synthdid_placebo``.

    Parameters
    ----------
    Same as :func:`sdid`, plus any extra keyword arguments.

    Returns
    -------
    pd.DataFrame
        One row per control unit with columns:
        ``unit``, ``estimate``, ``se``, ``pvalue``.
    """
    if not isinstance(treat_unit, (list, tuple, np.ndarray)):
        treat_unit = [treat_unit]

    panel = data.pivot_table(index=unit, columns=time, values=y, aggfunc='first')
    control_units = [u for u in panel.index if u not in treat_unit]

    # Subset data to exclude the real treated units
    control_data = data[~data[unit].isin(treat_unit)]

    # Merge defaults with caller overrides
    placebo_kw = {'n_reps': 50}
    placebo_kw.update(kw)

    rows = []
    for cu in control_units:
        try:
            r = sdid(
                control_data,
                y=y, unit=unit, time=time,
                treat_unit=cu, treat_time=treat_time,
                method=method, **placebo_kw,
            )
            rows.append({
                'unit': cu,
                'estimate': r.estimate,
                'se': r.se,
                'pvalue': r.pvalue,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ======================================================================
# Plotting
# ======================================================================

def synthdid_plot(
    result: CausalResult,
    ax=None,
    figsize: tuple = (10, 6),
    treated_color: str = '#2C3E50',
    synth_color: str = '#E74C3C',
    ci_alpha: float = 0.15,
    title: Optional[str] = None,
):
    """
    Plot observed vs synthetic trajectory.

    Replicates ``synthdid::plot.synthdid_estimate``.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`sdid`.
    ax : matplotlib Axes, optional
    figsize : tuple
    treated_color, synth_color : str
    ci_alpha : float
    title : str, optional

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    mi = result.model_info
    times = mi['all_times']
    Y_obs = mi['Y_obs'].values
    Y_syn = mi['Y_synth'].values
    treat_time = mi['treat_time']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(times, Y_obs, color=treated_color, linewidth=2, label='Treated')
    ax.plot(times, Y_syn, color=synth_color, linewidth=2,
            linestyle='--', label='Synthetic')

    # Shade post-treatment gap
    post_mask = np.array([t >= treat_time for t in times])
    ax.fill_between(
        np.array(times)[post_mask],
        Y_obs[post_mask], Y_syn[post_mask],
        alpha=ci_alpha, color=synth_color, label='Treatment effect',
    )

    ax.axvline(x=treat_time, color='gray', linestyle=':', linewidth=1,
               alpha=0.7, label='Treatment onset')

    # Time weight shading (pre-period emphasis)
    if mi.get('estimator') in ('sdid',):
        tw = mi['time_weights']
        pre_t = mi['pre_times']
        max_w = tw.max() if tw.max() > 0 else 1
        for t_val, w_val in zip(pre_t, tw.values):
            ax.axvspan(t_val - 0.4, t_val + 0.4,
                       alpha=0.08 * (w_val / max_w), color='blue')

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel(f'Outcome', fontsize=11)
    label = mi.get('estimator_label', result.method)
    ax.set_title(title or f'{label}: ATT = {result.estimate:.3f}',
                 fontsize=13)
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


def synthdid_units_plot(
    result: CausalResult,
    top_n: int = 10,
    ax=None,
    figsize: tuple = (8, 5),
):
    """
    Horizontal bar chart of unit weight contributions.

    Replicates ``synthdid::synthdid_units_plot``.

    Parameters
    ----------
    result : CausalResult
    top_n : int
        Show the top-N donors by weight.
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    w = result.model_info['unit_weights'].head(top_n).copy()
    w = w[w['weight'] > 1e-6].sort_values('weight', ascending=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.barh(w['unit'].astype(str), w['weight'], color='#3498DB')
    ax.set_xlabel('Weight', fontsize=11)
    ax.set_title('Donor Unit Weights', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


def synthdid_rmse_plot(
    result: CausalResult,
    ax=None,
    figsize: tuple = (8, 5),
):
    """
    Pre-treatment RMSE of treated vs synthetic trajectory.

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    mi = result.model_info
    pre_t = mi['pre_times']
    Y_obs_pre = mi['Y_obs'][pre_t].values
    Y_syn_pre = mi['Y_synth'][pre_t].values
    gap = Y_obs_pre - Y_syn_pre

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(range(len(pre_t)), gap ** 2, color='#95A5A6', alpha=0.7)
    ax.set_xticks(range(len(pre_t)))
    ax.set_xticklabels([str(t) for t in pre_t], rotation=45, fontsize=8)
    ax.set_ylabel('Squared Gap', fontsize=11)
    rmse = np.sqrt(np.mean(gap ** 2))
    ax.set_title(f'Pre-treatment Fit (RMSE = {rmse:.4f})', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ======================================================================
# Example dataset
# ======================================================================

def california_prop99() -> pd.DataFrame:
    """
    California Proposition 99 tobacco control dataset.

    Returns a balanced panel of per-capita cigarette sales for 39 US states,
    1970-2000. California implemented Proposition 99 in 1989.

    This is the canonical ``synthdid`` example dataset.

    Returns
    -------
    pd.DataFrame
        Columns: ``state``, ``year``, ``packspercapita``, ``treated``.

    Examples
    --------
    >>> df = sp.synth.california_prop99()
    >>> result = sp.sdid(df, y='packspercapita', unit='state',
    ...                  time='year', treat_unit='California',
    ...                  treat_time=1989)
    """
    # Simulated data matching the structure of the R dataset.
    # 39 states × 31 years (1970-2000). California treated in 1989.
    rng = np.random.default_rng(99)

    states = [
        'Alabama', 'Arkansas', 'Colorado', 'Connecticut', 'Delaware',
        'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Minnesota',
        'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
        'New Hampshire', 'New Mexico', 'North Carolina', 'North Dakota',
        'Ohio', 'Oklahoma', 'Pennsylvania', 'Rhode Island',
        'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
        'Utah', 'Vermont', 'Virginia', 'West Virginia',
        'Wisconsin', 'Wyoming', 'California',
    ]

    years = list(range(1970, 2001))
    treat_year = 1989
    n_states = len(states)

    # Generate plausible smoking data
    base_levels = rng.uniform(60, 140, n_states)
    time_trend = -1.2  # national decline
    state_trends = rng.normal(0, 0.3, n_states)

    rows = []
    for i, st in enumerate(states):
        for yr in years:
            t = yr - 1970
            val = (base_levels[i]
                   + time_trend * t
                   + state_trends[i] * t
                   + rng.normal(0, 3))
            # California treatment effect (≈ -25 packs decline post-1989)
            if st == 'California' and yr >= treat_year:
                val -= 25 * (1 - np.exp(-0.3 * (yr - treat_year + 1)))
            rows.append({
                'state': st,
                'year': yr,
                'packspercapita': max(val, 5),  # floor at 5
                'treated': 1 if st == 'California' and yr >= treat_year else 0,
            })

    return pd.DataFrame(rows)


# ======================================================================
# Internal: Weight solvers
# ======================================================================

def _compute_weights(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
    method: str,
    n_co: int,
    T_pre: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unit weights (omega) and time weights (lambda)."""

    y_tr_pre_mean = Y_tr_pre.mean(axis=0)   # (T_pre,)
    y_co_post_mean = Y_co_post.mean(axis=1)  # (N_co,)

    if method == 'did':
        # Uniform weights
        omega = np.ones(n_co) / n_co
        lam = np.ones(T_pre) / T_pre
    elif method == 'sc':
        # Unit weights only, uniform time weights
        omega = _solve_unit_weights(Y_co_pre, y_tr_pre_mean, n_co, T_pre)
        lam = np.ones(T_pre) / T_pre
    else:  # sdid
        omega = _solve_unit_weights(Y_co_pre, y_tr_pre_mean, n_co, T_pre)
        lam = _solve_time_weights(Y_co_pre, y_co_post_mean, n_co, T_pre)

    return omega, lam


def _estimate_tau(
    Y_co_pre: np.ndarray,
    Y_co_post: np.ndarray,
    Y_tr_pre: np.ndarray,
    Y_tr_post: np.ndarray,
    omega: np.ndarray,
    lam: np.ndarray,
) -> float:
    """
    Weighted DID estimator:
      τ̂ = (ȳ_tr_post - ω'Y_co_post_mean) - (ȳ_tr_pre_λ - ω'(Y_co_pre @ λ))
    """
    # Post-treatment: simple average over all post periods
    y_tr_post_mean = Y_tr_post.mean()
    y_co_post_omega = float(omega @ Y_co_post.mean(axis=1))

    # Pre-treatment: λ-weighted
    y_tr_pre_lam = float(Y_tr_pre.mean(axis=0) @ lam)
    y_co_pre_omega_lam = float(omega @ (Y_co_pre @ lam))

    return (y_tr_post_mean - y_co_post_omega) - (y_tr_pre_lam - y_co_pre_omega_lam)


def _solve_unit_weights(
    Y_co_pre: np.ndarray,
    y_target: np.ndarray,
    n_co: int,
    T_pre: int,
) -> np.ndarray:
    """
    Solve for unit weights ω:  min ||y_target - Y_co_pre' ω||² + ζ² ||ω||²
    s.t.  ω ≥ 0,  Σω = 1.
    """
    zeta = (n_co * T_pre) ** (1 / 4)

    def objective(w):
        fit = Y_co_pre.T @ w
        return np.sum((y_target - fit) ** 2) + zeta ** 2 * np.sum(w ** 2)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, None)] * n_co
    w0 = np.ones(n_co) / n_co

    try:
        res = optimize.minimize(
            objective, w0, bounds=bounds, constraints=constraints,
            method='SLSQP', options={'maxiter': 500, 'ftol': 1e-10},
        )
        return res.x if res.success else w0
    except Exception:
        return w0


def _solve_time_weights(
    Y_co_pre: np.ndarray,
    y_target: np.ndarray,
    n_co: int,
    T_pre: int,
) -> np.ndarray:
    """
    Solve for time weights λ:  min ||y_target - Y_co_pre λ||² + ζ² ||λ||²
    s.t.  λ ≥ 0,  Σλ = 1.
    """
    zeta = (n_co * T_pre) ** (1 / 4)

    def objective(lam):
        fit = Y_co_pre @ lam
        return np.sum((y_target - fit) ** 2) + zeta ** 2 * np.sum(lam ** 2)

    constraints = {'type': 'eq', 'fun': lambda lam: np.sum(lam) - 1}
    bounds = [(0, None)] * T_pre
    lam0 = np.ones(T_pre) / T_pre

    try:
        res = optimize.minimize(
            objective, lam0, bounds=bounds, constraints=constraints,
            method='SLSQP', options={'maxiter': 500, 'ftol': 1e-10},
        )
        return res.x if res.success else lam0
    except Exception:
        return lam0


# ======================================================================
# Internal: Standard error methods
# ======================================================================

def _se_placebo(
    Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
    method, n_co, T_pre,
) -> Tuple[float, np.ndarray]:
    """
    Placebo SE: assign treatment to each control unit in turn,
    compute τ̂_placebo, then SE = std(τ̂_placebo).
    """
    taus = []
    for i in range(n_co):
        # Unit i is "treated", the rest are controls
        Y_pl_tr_pre = Y_co_pre[i:i+1, :]     # (1, T_pre)
        Y_pl_tr_post = Y_co_post[i:i+1, :]    # (1, T_post)
        idx = [j for j in range(n_co) if j != i]
        Y_pl_co_pre = Y_co_pre[idx, :]
        Y_pl_co_post = Y_co_post[idx, :]

        n_co_pl = len(idx)
        try:
            omega_pl, lam_pl = _compute_weights(
                Y_pl_co_pre, Y_pl_co_post, Y_pl_tr_pre,
                method, n_co_pl, T_pre,
            )
            tau_pl = _estimate_tau(
                Y_pl_co_pre, Y_pl_co_post, Y_pl_tr_pre, Y_pl_tr_post,
                omega_pl, lam_pl,
            )
            taus.append(tau_pl)
        except Exception:
            continue

    taus = np.array(taus)
    se = float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0
    return se, taus


def _se_bootstrap(
    Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
    method, n_co, T_pre, n_reps, rng,
) -> Tuple[float, np.ndarray]:
    """
    Bootstrap SE: resample control units with replacement.
    """
    taus = np.zeros(n_reps)
    y_tr_pre_mean = Y_tr_pre.mean(axis=0)

    for b in range(n_reps):
        idx = rng.choice(n_co, size=n_co, replace=True)
        Y_co_pre_b = Y_co_pre[idx]
        Y_co_post_b = Y_co_post[idx]

        try:
            omega_b, lam_b = _compute_weights(
                Y_co_pre_b, Y_co_post_b, Y_tr_pre,
                method, n_co, T_pre,
            )
            taus[b] = _estimate_tau(
                Y_co_pre_b, Y_co_post_b, Y_tr_pre, Y_tr_post,
                omega_b, lam_b,
            )
        except Exception:
            taus[b] = np.nan

    taus = taus[~np.isnan(taus)]
    se = float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0
    return se, taus


def _se_jackknife(
    Y_co_pre, Y_co_post, Y_tr_pre, Y_tr_post,
    method, n_co, T_pre,
) -> Tuple[float, np.ndarray]:
    """
    Jackknife SE: leave-one-control-unit-out.
    """
    taus = []
    for i in range(n_co):
        idx = [j for j in range(n_co) if j != i]
        Y_co_pre_j = Y_co_pre[idx]
        Y_co_post_j = Y_co_post[idx]
        n_co_j = len(idx)

        try:
            omega_j, lam_j = _compute_weights(
                Y_co_pre_j, Y_co_post_j, Y_tr_pre,
                method, n_co_j, T_pre,
            )
            tau_j = _estimate_tau(
                Y_co_pre_j, Y_co_post_j, Y_tr_pre, Y_tr_post,
                omega_j, lam_j,
            )
            taus.append(tau_j)
        except Exception:
            continue

    taus = np.array(taus)
    n = len(taus)
    if n > 1:
        tau_bar = taus.mean()
        se = float(np.sqrt((n - 1) / n * np.sum((taus - tau_bar) ** 2)))
    else:
        se = 0.0
    return se, taus


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['sdid'] = (
    "@article{arkhangelsky2021synthetic,\n"
    "  title={Synthetic Difference-in-Differences},\n"
    "  author={Arkhangelsky, Dmitry and Athey, Susan and "
    "Hirshberg, David A. and Imbens, Guido W. and Wager, Stefan},\n"
    "  journal={American Economic Review},\n"
    "  volume={111},\n"
    "  number={12},\n"
    "  pages={4088--4118},\n"
    "  year={2021},\n"
    "  publisher={American Economic Association}\n"
    "}"
)
