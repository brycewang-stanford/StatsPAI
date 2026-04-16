"""
Distributional Synthetic Controls (DiSCo).

Instead of matching means (classic SCM), DiSCo matches entire quantile
functions across units, enabling estimation of distributional treatment
effects — shifts across the entire outcome distribution, not just the
average.

Model
-----
For each quantile level τ ∈ [0, 1]:

    Q̂_counterfactual(τ) = Σ_j  ω_j  Q_j(τ)

where Q_j(τ) is control unit j's quantile function and ω are weights
chosen to minimise an integrated (Wasserstein-type) loss over the
pre-treatment period.

The distributional treatment effect at quantile τ is:

    Δ(τ) = Q_treated(τ) − Q̂_counterfactual(τ)

and the average distributional effect integrates over τ.

Two approaches
--------------
- **mixture** (default): weighted mixture of control CDFs
  (ω ≥ 0, Σω = 1). Minimises the L₂-Wasserstein distance between
  the treated unit's quantile function and the convex combination
  of control quantile functions.
- **quantile**: quantile-on-quantile regression, projecting the
  treated unit's quantile function onto control quantile functions
  without sign or summation constraints.

References
----------
Gunsilius, F. F. (2023).
"Distributional Synthetic Controls."
*Econometrica*, 91(3), 1105-1117.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize

from ..core.results import CausalResult


# ====================================================================== #
#  Public API
# ====================================================================== #

def discos(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    method: str = "mixture",
    n_quantiles: int = 100,
    placebo: bool = True,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Distributional Synthetic Controls (Gunsilius 2023).

    Matches the entire quantile function of a treated unit to a weighted
    combination of control units' quantile functions, then estimates
    distributional treatment effects across quantiles.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with columns for unit, time, and outcome.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : any
        Value in *unit* that identifies the treated unit.
    treatment_time : any
        First period of treatment (inclusive).
    method : {'mixture', 'quantile'}, default 'mixture'
        ``'mixture'``: constrained (ω ≥ 0, Σω = 1) — minimises the
        L₂-Wasserstein distance between quantile functions.
        ``'quantile'``: unconstrained quantile-on-quantile regression.
    n_quantiles : int, default 100
        Number of quantile grid points on (0, 1).
    placebo : bool, default True
        Run in-space placebo permutation tests for inference.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    seed : int, optional
        Random seed (currently unused; reserved for bootstrap extensions).

    Returns
    -------
    CausalResult
        With ``.estimate`` equal to the average quantile treatment effect
        (mean of Δ(τ) across τ), and ``model_info`` containing full
        distributional results.

    Notes
    -----
    The method requires panel data where each unit has observations across
    multiple time periods. Pre-treatment time-series observations for each
    unit are used to form empirical quantile functions.

    When ``method='mixture'``, the optimisation problem is:

    .. math::
        \\min_{\\omega} \\sum_{t \\in \\text{pre}}
        \\int_0^1 \\bigl[Q_{1t}(\\tau) -
        \\sum_j \\omega_j Q_{jt}(\\tau)\\bigr]^2 \\, d\\tau
        \\quad \\text{s.t.} \\; \\omega \\geq 0,\\; \\sum \\omega = 1

    Examples
    --------
    >>> result = sp.discos(df, outcome='gdp', unit='state', time='year',
    ...                    treated_unit='California', treatment_time=1989)
    >>> print(result.summary())

    >>> # Quantile-level effects
    >>> qte = result.model_info['quantile_effects']
    >>> print(qte.head())

    >>> # Visualise distributional effects
    >>> sp.discos_plot(result, type='quantile_effect')

    See Also
    --------
    synth : Classic (mean-matching) synthetic control.
    qqsynth : Alias for ``discos(..., method='quantile')``.
    """
    rng = np.random.default_rng(seed)

    if method not in ("mixture", "quantile"):
        raise ValueError(
            f"method must be 'mixture' or 'quantile', got '{method}'"
        )

    # --- Build panel ---
    pivot = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(pivot.columns.tolist())
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("DiSCo needs at least 2 pre-treatment periods")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period")

    donors = [u for u in pivot.index if u != treated_unit]
    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    if J < 2:
        raise ValueError("Need at least 2 control (donor) units")

    # --- Quantile grid ---
    tau_grid = np.linspace(1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)

    # --- Build quantile matrices for pre- and post-treatment ---
    # Q_treated_pre[t, q]: treated unit's q-th quantile at pre-period t
    # We use the time-series of outcomes up to each pre-period to form
    # expanding-window empirical quantile functions (rolling CDF from
    # the panel values at that cross-section). For standard panel data
    # with a single outcome per unit-time, we use the full pre-treatment
    # time-series per unit to construct one empirical quantile function.

    Y_treated_pre = pivot.loc[treated_unit, pre_times].values.astype(np.float64)
    Y_treated_post = pivot.loc[treated_unit, post_times].values.astype(np.float64)
    Y_donors_pre = pivot.loc[donors, pre_times].values.astype(np.float64)   # (J, T0)
    Y_donors_post = pivot.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    # Empirical quantile functions from the pre-treatment time-series
    Q_treated_pre = _empirical_quantile_function(Y_treated_pre, tau_grid)    # (n_q,)
    Q_treated_post = _empirical_quantile_function(Y_treated_post, tau_grid)  # (n_q,)

    Q_donors_pre = np.zeros((J, n_quantiles))   # (J, n_q)
    Q_donors_post = np.zeros((J, n_quantiles))  # (J, n_q)
    for j in range(J):
        Q_donors_pre[j] = _empirical_quantile_function(
            Y_donors_pre[j], tau_grid
        )
        Q_donors_post[j] = _empirical_quantile_function(
            Y_donors_post[j], tau_grid
        )

    # --- Fit weights ---
    if method == "mixture":
        weights = _mixture_weights(Q_treated_pre, Q_donors_pre)
    else:
        weights = _quantile_weights(Q_treated_pre, Q_donors_pre)

    # --- Counterfactual quantile function (post-treatment) ---
    Q_counterfactual_post = weights @ Q_donors_post  # (n_q,)
    Q_counterfactual_pre = weights @ Q_donors_pre    # (n_q,)

    # --- Distributional treatment effects ---
    quantile_effects = Q_treated_post - Q_counterfactual_post  # (n_q,)
    avg_qte = float(np.mean(quantile_effects))

    # Pre-treatment fit
    pre_residuals = Q_treated_pre - Q_counterfactual_pre
    pre_rmsqe = float(np.sqrt(np.mean(pre_residuals ** 2)))

    # --- Placebo inference ---
    placebo_avg_qtes: List[float] = []
    placebo_quantile_effects: List[np.ndarray] = []

    if placebo and J >= 3:
        for j in range(J):
            other_idx = [i for i in range(J) if i != j]
            Q_plac_pre = Q_donors_pre[j]
            Q_plac_post = Q_donors_post[j]
            Q_ctrl_pre = Q_donors_pre[other_idx]
            Q_ctrl_post = Q_donors_post[other_idx]

            try:
                if method == "mixture":
                    w_plac = _mixture_weights(Q_plac_pre, Q_ctrl_pre)
                else:
                    w_plac = _quantile_weights(Q_plac_pre, Q_ctrl_pre)

                Q_cf_plac = w_plac @ Q_ctrl_post
                plac_effects = Q_plac_post - Q_cf_plac
                placebo_avg_qtes.append(float(np.mean(plac_effects)))
                placebo_quantile_effects.append(plac_effects)
            except Exception:
                continue

    # --- Standard errors and p-value ---
    if len(placebo_avg_qtes) > 0:
        se = float(np.std(placebo_avg_qtes, ddof=1))
        pvalue = float(
            np.mean(np.abs(placebo_avg_qtes) >= abs(avg_qte))
        )
        pvalue = max(pvalue, 1 / (len(placebo_avg_qtes) + 1))

        # Quantile-level CIs from placebo distribution
        if len(placebo_quantile_effects) > 0:
            plac_q_arr = np.array(placebo_quantile_effects)  # (n_plac, n_q)
            q_se = np.std(plac_q_arr, axis=0, ddof=1)        # (n_q,)
        else:
            q_se = np.full(n_quantiles, np.nan)
    else:
        se = float(np.std(quantile_effects)) / max(np.sqrt(n_quantiles), 1)
        pvalue = np.nan
        q_se = np.full(n_quantiles, np.nan)

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (avg_qte - z_crit * se, avg_qte + z_crit * se)

    # --- Build quantile-level results table ---
    q_ci_lower = quantile_effects - z_crit * q_se
    q_ci_upper = quantile_effects + z_crit * q_se

    quantile_effects_df = pd.DataFrame({
        "quantile": tau_grid,
        "effect": quantile_effects,
        "ci_lower": q_ci_lower,
        "ci_upper": q_ci_upper,
    })

    # --- Gap table (period-level, using raw outcomes) ---
    Y_synth_pre = weights @ Y_donors_pre   # (T0,)
    Y_synth_post = weights @ Y_donors_post  # (T1,)
    Y_synth = np.concatenate([Y_synth_pre, Y_synth_post])
    Y_treated = np.concatenate([Y_treated_pre, Y_treated_post])

    gap_df = pd.DataFrame({
        "time": all_times,
        "treated": Y_treated,
        "synthetic": Y_synth,
        "gap": Y_treated - Y_synth,
    })

    # --- Effects by period ---
    effects_df = pd.DataFrame({
        "time": post_times,
        "treated": Y_treated_post,
        "counterfactual": Y_synth_post,
        "effect": Y_treated_post - Y_synth_post,
    })

    # --- Build model_info ---
    model_info: Dict[str, Any] = {
        "method_variant": method,
        "n_quantiles": n_quantiles,
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "pre_rmsqe": round(pre_rmsqe, 6),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "quantile_effects": quantile_effects_df,
        "weights": dict(zip(donors, weights)),
        "tau_grid": tau_grid,
        "counterfactual_quantiles": Q_counterfactual_post,
        "treated_quantiles": Q_treated_post,
        "gap_table": gap_df,
        "effects_by_period": effects_df,
        "Y_synth": Y_synth,
        "Y_treated": Y_treated,
        "times": all_times,
    }

    if placebo_avg_qtes:
        model_info["placebo_atts"] = placebo_avg_qtes
        model_info["n_placebos"] = len(placebo_avg_qtes)
    if len(placebo_quantile_effects) > 0:
        model_info["placebo_quantile_effects"] = np.array(
            placebo_quantile_effects
        )

    return CausalResult(
        method="Distributional Synthetic Controls (Gunsilius 2023)",
        estimand="Distributional ATT",
        estimate=avg_qte,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info=model_info,
        _citation_key="discos",
    )


def qqsynth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    n_quantiles: int = 100,
    placebo: bool = True,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Quantile Synthetic Control (alias for DiSCo with ``method='quantile'``).

    Applies quantile-on-quantile regression to match quantile functions
    without the convexity constraints of the mixture approach.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    outcome : str
        Outcome variable column.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period (inclusive).
    n_quantiles : int, default 100
        Number of quantile grid points.
    placebo : bool, default True
        Run placebo permutation inference.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> result = sp.qqsynth(df, outcome='gdp', unit='state', time='year',
    ...                     treated_unit='California', treatment_time=1989)
    >>> print(result.summary())

    See Also
    --------
    discos : Full distributional synthetic controls with method selection.
    """
    return discos(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        method="quantile",
        n_quantiles=n_quantiles,
        placebo=placebo,
        alpha=alpha,
        seed=seed,
    )


# ====================================================================== #
#  Post-estimation: testing
# ====================================================================== #

def discos_test(
    result: CausalResult,
    test: str = "ks",
) -> Dict[str, Any]:
    """
    Test for distributional treatment effects.

    Parameters
    ----------
    result : CausalResult
        Output from ``discos()`` or ``qqsynth()``.
    test : {'ks', 'cvm', 'stochastic_dominance'}, default 'ks'
        ``'ks'``: two-sample Kolmogorov-Smirnov test comparing treated
        and counterfactual quantile functions.
        ``'cvm'``: Cramér-von Mises test statistic (permutation-based).
        ``'stochastic_dominance'``: first-order stochastic dominance test.

    Returns
    -------
    dict
        Keys: ``'test'``, ``'statistic'``, ``'pvalue'``, ``'reject'``,
        ``'alpha'``, and test-specific fields.

    Examples
    --------
    >>> result = sp.discos(df, outcome='gdp', unit='state', time='year',
    ...                    treated_unit='California', treatment_time=1989)
    >>> sp.discos_test(result, test='ks')
    {'test': 'Kolmogorov-Smirnov', 'statistic': 0.32, 'pvalue': 0.014, ...}
    """
    mi = result.model_info
    Q_treated = mi["treated_quantiles"]
    Q_counterfactual = mi["counterfactual_quantiles"]
    alpha = result.alpha

    if test == "ks":
        return _ks_test(Q_treated, Q_counterfactual, alpha)
    elif test == "cvm":
        return _cvm_test(Q_treated, Q_counterfactual, mi, alpha)
    elif test == "stochastic_dominance":
        return _stochastic_dominance_test(Q_treated, Q_counterfactual, mi, alpha)
    else:
        raise ValueError(
            f"test must be 'ks', 'cvm', or 'stochastic_dominance', "
            f"got '{test}'"
        )


def stochastic_dominance(
    result: CausalResult,
    order: int = 1,
) -> Dict[str, Any]:
    """
    Test for stochastic dominance of the treated distribution over the
    counterfactual distribution.

    Parameters
    ----------
    result : CausalResult
        Output from ``discos()`` or ``qqsynth()``.
    order : {1, 2}, default 1
        Order of stochastic dominance.
        1 = first-order (CDF dominance).
        2 = second-order (integrated CDF dominance).

    Returns
    -------
    dict
        Keys: ``'order'``, ``'dominates'`` (bool), ``'min_gap'``,
        ``'max_gap'``, ``'fraction_positive'``, ``'statistic'``,
        ``'pvalue'``.

    Examples
    --------
    >>> result = sp.discos(df, outcome='gdp', unit='state', time='year',
    ...                    treated_unit='California', treatment_time=1989)
    >>> sp.stochastic_dominance(result, order=1)
    """
    mi = result.model_info
    Q_treated = mi["treated_quantiles"]
    Q_cf = mi["counterfactual_quantiles"]
    alpha = result.alpha

    if order == 1:
        return _stochastic_dominance_test(Q_treated, Q_cf, mi, alpha)
    elif order == 2:
        return _second_order_dominance(Q_treated, Q_cf, mi, alpha)
    else:
        raise ValueError("order must be 1 or 2")


# ====================================================================== #
#  Post-estimation: plotting
# ====================================================================== #

def discos_plot(
    result: CausalResult,
    type: str = "quantile_effect",
    ax=None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#2C3E50",
    ci_alpha: float = 0.2,
    title: Optional[str] = None,
):
    """
    Visualise distributional synthetic control results.

    Parameters
    ----------
    result : CausalResult
        Output from ``discos()`` or ``qqsynth()``.
    type : {'quantile_effect', 'quantile_comparison', 'gap', 'weights'},
           default 'quantile_effect'
        ``'quantile_effect'``: treatment effect Δ(τ) across quantiles
        with CIs.
        ``'quantile_comparison'``: overlay treated vs. counterfactual
        quantile functions.
        ``'gap'``: gap plot (treated − synthetic) over time.
        ``'weights'``: horizontal bar chart of donor weights.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot.
    figsize : tuple, default (10, 6)
        Figure size.
    color : str, default '#2C3E50'
        Primary plot colour.
    ci_alpha : float, default 0.2
        Transparency for CI band.
    title : str, optional
        Plot title override.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> result = sp.discos(df, outcome='gdp', unit='state', time='year',
    ...                    treated_unit='California', treatment_time=1989)
    >>> sp.discos_plot(result, type='quantile_effect')
    >>> sp.discos_plot(result, type='quantile_comparison')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install: pip install matplotlib"
        )

    mi = result.model_info

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if type == "quantile_effect":
        qte = mi["quantile_effects"]
        tau = qte["quantile"].values
        eff = qte["effect"].values
        ci_lo = qte["ci_lower"].values
        ci_hi = qte["ci_upper"].values

        ax.fill_between(tau, ci_lo, ci_hi, alpha=ci_alpha, color=color,
                        label=f"{int(100*(1-result.alpha))}% CI")
        ax.plot(tau, eff, color=color, linewidth=1.5, label="Δ(τ)")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(y=result.estimate, color="#E74C3C", linestyle=":",
                   linewidth=1, alpha=0.6, label=f"Mean = {result.estimate:.4f}")
        ax.set_xlabel("Quantile (τ)", fontsize=11)
        ax.set_ylabel("Treatment Effect Δ(τ)", fontsize=11)
        ax.set_title(
            title or "Distributional Treatment Effect by Quantile",
            fontsize=13,
        )
        ax.legend(fontsize=9, frameon=False)

    elif type == "quantile_comparison":
        tau = mi["tau_grid"]
        Q_tr = mi["treated_quantiles"]
        Q_cf = mi["counterfactual_quantiles"]

        ax.plot(tau, Q_tr, color=color, linewidth=1.5,
                label="Treated (observed)")
        ax.plot(tau, Q_cf, color="#E74C3C", linewidth=1.5,
                linestyle="--", label="Counterfactual (DiSCo)")
        ax.set_xlabel("Quantile (τ)", fontsize=11)
        ax.set_ylabel("Outcome", fontsize=11)
        ax.set_title(
            title or "Quantile Functions: Treated vs. Counterfactual",
            fontsize=13,
        )
        ax.legend(fontsize=9, frameon=False)

    elif type == "gap":
        gap = mi["gap_table"]
        times = gap["time"].values
        gaps = gap["gap"].values
        treatment_time = mi["treatment_time"]

        ax.plot(times, gaps, color=color, linewidth=1.5, marker="o",
                markersize=4)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(x=treatment_time, color="#E74C3C", linestyle=":",
                   linewidth=1, alpha=0.6, label="Treatment onset")
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Gap (Treated − Synthetic)", fontsize=11)
        ax.set_title(title or "Gap Plot", fontsize=13)
        ax.legend(fontsize=9, frameon=False)

    elif type == "weights":
        weights = mi["weights"]
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        # Show donors with weight > 0.001
        sorted_w = [(k, v) for k, v in sorted_w if v > 0.001]
        if not sorted_w:
            sorted_w = sorted(weights.items(), key=lambda x: x[1],
                              reverse=True)[:10]

        labels, vals = zip(*sorted_w)
        y_pos = np.arange(len(labels))

        ax.barh(y_pos, vals, color=color, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Weight", fontsize=11)
        ax.set_title(title or "DiSCo Donor Weights", fontsize=13)
        ax.invert_yaxis()

    else:
        raise ValueError(
            f"type must be 'quantile_effect', 'quantile_comparison', "
            f"'gap', or 'weights', got '{type}'"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig, ax


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _empirical_quantile_function(
    y: np.ndarray,
    tau_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the empirical quantile function of *y* evaluated at each
    probability level in *tau_grid*.

    Uses linear interpolation between order statistics (type-7 quantile,
    same as NumPy default).

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Observed values (e.g., a unit's time-series of outcomes).
    tau_grid : np.ndarray, shape (n_q,)
        Probability levels in (0, 1).

    Returns
    -------
    np.ndarray, shape (n_q,)
        Quantile values.
    """
    y_clean = y[~np.isnan(y)]
    if len(y_clean) < 2:
        return np.full_like(tau_grid, np.nan)
    return np.quantile(y_clean, tau_grid)


def _mixture_weights(
    Q_treated: np.ndarray,
    Q_donors: np.ndarray,
) -> np.ndarray:
    """
    Solve for mixture weights that minimise the integrated squared
    difference between the treated quantile function and the weighted
    combination of donor quantile functions.

    .. math::
        \\min_{\\omega} \\| Q_{\\text{treated}} -
        Q_{\\text{donors}}^\\top \\omega \\|_2^2
        \\quad \\text{s.t.} \\; \\omega \\ge 0,\\; \\mathbf{1}^\\top \\omega = 1

    Parameters
    ----------
    Q_treated : np.ndarray, shape (n_q,)
    Q_donors : np.ndarray, shape (J, n_q)

    Returns
    -------
    np.ndarray, shape (J,)
    """
    J = Q_donors.shape[0]

    def objective(w):
        residual = Q_treated - w @ Q_donors
        return float(np.sum(residual ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * J
    w0 = np.ones(J) / J

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x


def _quantile_weights(
    Q_treated: np.ndarray,
    Q_donors: np.ndarray,
) -> np.ndarray:
    """
    Unconstrained quantile-on-quantile regression weights.

    Solves Q_treated = Q_donors^T w via OLS (no sign or sum constraints).

    Parameters
    ----------
    Q_treated : np.ndarray, shape (n_q,)
    Q_donors : np.ndarray, shape (J, n_q)

    Returns
    -------
    np.ndarray, shape (J,)
    """
    # OLS: w = (Q Q')^{-1} Q y
    # Q_donors: (J, n_q), Q_treated: (n_q,)
    QQt = Q_donors @ Q_donors.T  # (J, J)
    Qy = Q_donors @ Q_treated    # (J,)
    # Ridge regularisation for numerical stability
    lam = 1e-8 * np.trace(QQt) / max(QQt.shape[0], 1)
    w = np.linalg.solve(QQt + lam * np.eye(QQt.shape[0]), Qy)
    return w


def _ks_test(
    Q_treated: np.ndarray,
    Q_counterfactual: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    """
    Kolmogorov-Smirnov test on the quantile functions.

    The KS statistic is the maximum absolute difference between the
    two quantile functions: D = max_τ |Q_treated(τ) - Q_cf(τ)|.

    Approximation: use the two-sample KS test on the quantile values
    as if they were samples.
    """
    stat, pval = sp_stats.ks_2samp(Q_treated, Q_counterfactual)
    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": float(stat),
        "pvalue": float(pval),
        "reject": bool(pval < alpha),
        "alpha": alpha,
    }


def _cvm_test(
    Q_treated: np.ndarray,
    Q_counterfactual: np.ndarray,
    model_info: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """
    Cramér-von Mises test statistic for distributional difference.

    CvM = (1/n_q) Σ [Q_treated(τ) - Q_cf(τ)]²

    P-value via placebo distribution if available, else asymptotic.
    """
    n_q = len(Q_treated)
    diff_sq = (Q_treated - Q_counterfactual) ** 2
    cvm_stat = float(np.mean(diff_sq))

    # Placebo-based p-value
    if "placebo_quantile_effects" in model_info:
        plac_arr = model_info["placebo_quantile_effects"]  # (n_plac, n_q)
        plac_cvm = np.mean(plac_arr ** 2, axis=1)
        pval = float(np.mean(plac_cvm >= cvm_stat))
        pval = max(pval, 1 / (len(plac_cvm) + 1))
    else:
        # Asymptotic: treat as chi-squared approximation
        # Under H0, n_q * CvM ~ sum of squared normals
        pval = float(1.0 - sp_stats.chi2.cdf(n_q * cvm_stat / max(np.var(Q_treated), 1e-10), df=n_q))

    return {
        "test": "Cramer-von Mises",
        "statistic": float(cvm_stat),
        "pvalue": float(pval),
        "reject": bool(pval < alpha),
        "alpha": alpha,
    }


def _stochastic_dominance_test(
    Q_treated: np.ndarray,
    Q_counterfactual: np.ndarray,
    model_info: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """
    First-order stochastic dominance test.

    Checks whether Q_treated(τ) >= Q_counterfactual(τ) for all τ,
    meaning the treated distribution first-order stochastically dominates
    the counterfactual (outcomes are uniformly higher).
    """
    gaps = Q_treated - Q_counterfactual
    min_gap = float(np.min(gaps))
    max_gap = float(np.max(gaps))
    frac_positive = float(np.mean(gaps >= 0))
    dominates = bool(min_gap >= 0)

    # Permutation-based p-value for the minimum gap statistic
    if "placebo_quantile_effects" in model_info:
        plac_arr = model_info["placebo_quantile_effects"]
        plac_min_gaps = np.min(plac_arr, axis=1)
        # H0: no dominance. p = fraction of placebos with min_gap >= observed
        pval = float(np.mean(plac_min_gaps >= min_gap))
        pval = max(pval, 1 / (len(plac_min_gaps) + 1))
    else:
        # Approximate: use KS test as fallback
        ks_stat, pval = sp_stats.ks_2samp(Q_treated, Q_counterfactual,
                                           alternative="less")
        pval = float(pval)

    return {
        "test": "First-Order Stochastic Dominance",
        "order": 1,
        "dominates": dominates,
        "min_gap": min_gap,
        "max_gap": max_gap,
        "fraction_positive": frac_positive,
        "statistic": min_gap,
        "pvalue": pval,
        "reject_no_dominance": bool(pval < alpha),
        "alpha": alpha,
    }


def _second_order_dominance(
    Q_treated: np.ndarray,
    Q_counterfactual: np.ndarray,
    model_info: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """
    Second-order stochastic dominance test.

    The treated distribution second-order dominates if the cumulative
    sum of quantile differences is non-negative at every point:

    .. math::
        \\sum_{\\tau' \\leq \\tau} [Q_{\\text{treated}}(\\tau') -
        Q_{\\text{cf}}(\\tau')] \\geq 0 \\quad \\forall \\tau
    """
    gaps = Q_treated - Q_counterfactual
    n_q = len(gaps)
    # Normalise by grid spacing (1/n_q)
    cumulative_gaps = np.cumsum(gaps) / n_q
    min_cum_gap = float(np.min(cumulative_gaps))
    dominates = bool(min_cum_gap >= 0)
    frac_positive = float(np.mean(cumulative_gaps >= 0))

    # Permutation-based inference
    if "placebo_quantile_effects" in model_info:
        plac_arr = model_info["placebo_quantile_effects"]
        plac_cum = np.cumsum(plac_arr, axis=1) / n_q
        plac_min_cum = np.min(plac_cum, axis=1)
        pval = float(np.mean(plac_min_cum >= min_cum_gap))
        pval = max(pval, 1 / (len(plac_min_cum) + 1))
    else:
        pval = np.nan

    return {
        "test": "Second-Order Stochastic Dominance",
        "order": 2,
        "dominates": dominates,
        "min_cumulative_gap": min_cum_gap,
        "fraction_positive": frac_positive,
        "statistic": min_cum_gap,
        "pvalue": pval,
        "reject_no_dominance": bool(pval < alpha) if not np.isnan(pval) else False,
        "alpha": alpha,
    }


# ====================================================================== #
#  Citation
# ====================================================================== #

CausalResult._CITATIONS["discos"] = (
    "@article{gunsilius2023distributional,\n"
    "  title={Distributional Synthetic Controls},\n"
    "  author={Gunsilius, Florian F.},\n"
    "  journal={Econometrica},\n"
    "  volume={91},\n"
    "  number={3},\n"
    "  pages={1105--1117},\n"
    "  year={2023},\n"
    "  publisher={Wiley}\n"
    "}"
)
