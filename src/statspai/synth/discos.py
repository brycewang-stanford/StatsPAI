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

Two data layouts
----------------
**Individual-level data** (several rows per unit-period — the setting of
Gunsilius 2023). ``discos`` runs the distributional synthetic control of
the paper, following the conventions of the authors' R package
``DiSCos::DiSCo`` (version 0.1.4):

- ``method='quantile'`` (R ``mixture = FALSE``, the R default): for every
  pre-period ``t`` the weights solve the constrained least-squares problem
  ``min_w sum_m (Q_1t(u_m) - sum_j w_j Q_jt(u_m))^2`` over quantile nodes
  ``u_m`` (type-7 empirical quantiles) subject to ``sum w = 1``,
  ``w <= 1`` and, with ``simplex=True``, ``w >= 0``. The weights are then
  averaged over the pre-periods, and the counterfactual quantile function
  in every period is ``sum_j w_j Q_jt`` on the grid ``0, 1/G, ..., 1``.
- ``method='mixture'`` (R ``mixture = TRUE``): the weights minimise the L1
  distance between the treated CDF and the mixture of control CDFs on a
  grid of ``G`` outcome values (``sum w = 1``; ``w >= 0`` with
  ``simplex=True``); the counterfactual quantile inverts the mixture CDF on
  that grid.

R draws the quantile nodes (``M`` uniform draws) and the CDF grid (``G``
uniform draws) at random; StatsPAI uses deterministic nodes / grids by
default (``M`` mid-points; ``G`` equispaced points on R's range) and
accepts R's draws through ``q_nodes`` / ``cdf_grid`` for exact
reproduction. Inference is DiSCo's permutation test (each control in turn
as the pseudo-treated unit; statistic = post / pre root-mean squared
Wasserstein distance; p = rank / (J + 1)).

**Aggregate panels** (one row per unit-period). The distributional
estimator is undefined there (it needs at least ``J`` observations per
unit-period). ``discos`` then falls back to a StatsPAI heuristic that
treats each unit's *time series* of outcomes as its distribution
(``'mixture'``: simplex-constrained quantile-function least squares;
``'quantile'``: unconstrained least squares) and emits a warning — that
fallback is not the Gunsilius estimator.

References
----------
Gunsilius, F. F. (2023).
"Distributional Synthetic Controls."
*Econometrica*, 91(3), 1105-1117. [@gunsilius2023distributional]
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import linprog
from scipy.stats import rankdata

from ..core.results import CausalResult
from ._core import placebo_rank_pvalue

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
    method: Optional[str] = None,
    n_quantiles: Optional[int] = None,
    placebo: bool = True,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    *,
    M: int = 1000,
    simplex: bool = False,
    q_nodes: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    cdf_grid: Optional[Sequence[np.ndarray]] = None,
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
    method : {'quantile', 'mixture'}, optional
        Individual-level data: ``'quantile'`` (default; R
        ``mixture = FALSE``) or ``'mixture'`` (R ``mixture = TRUE``) — see
        the module docstring. Aggregate panels: ``'mixture'`` (default) or
        ``'quantile'`` select the fallback heuristic.
    n_quantiles : int, optional
        Individual-level data: ``G`` — the counterfactual quantile function
        is reported on ``0, 1/G, ..., 1`` and the mixture CDF grid has
        ``G`` points (default 1000, as R). Aggregate panels: number of
        quantile grid points on (0, 1) (default 100).
    placebo : bool, default True
        Run in-space placebo permutation tests for inference.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    seed : int, optional
        Unused (all node sets are deterministic unless supplied).
    M : int, default 1000
        Individual-level data, ``method='quantile'``: number of quantile
        nodes in each weight regression (R ``M``). Default nodes are the
        mid-points ``(m - 0.5) / M``.
    simplex : bool, default False
        Individual-level data: restrict weights to be non-negative (R
        ``simplex``). ``False`` (R default) allows negative weights.
    q_nodes : array (M,) or sequence of arrays, optional
        Quantile nodes for the weight regressions: one array reused for
        every pre-period, or one array per pre-period (e.g. R's
        ``runif(M)`` draws, to reproduce ``DiSCo`` exactly).
    cdf_grid : sequence of arrays, optional
        ``method='mixture'``: outcome grid per period (all periods, in time
        order) on which the CDFs are compared and inverted (R's
        ``runif(G)`` grid). Default: ``G`` equispaced points on
        ``[floor(10 min)/10 - 0.25, ceil(10 max)/10 + 0.25]``.

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
    Individual-level data: 100 draws per unit-period, unit 0 treated from
    period 4 with a unit shift of its whole distribution.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> loc = {0: 2.5, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    >>> rows = [
    ...     (u, t, rng.normal(loc[u] + 0.2 * t + (u == 0 and t >= 4), 1.0))
    ...     for u in loc for t in range(1, 7) for _ in range(100)
    ... ]
    >>> df = pd.DataFrame(rows, columns=['unit', 'time', 'y'])
    >>> result = sp.discos(df, outcome='y', unit='unit', time='time',
    ...                    treated_unit=0, treatment_time=4,
    ...                    M=200, n_quantiles=100)
    >>> bool(0.5 < result.estimate < 1.5)
    True
    >>> qte = result.model_info['quantile_effects']
    >>> bool(set(['quantile', 'effect']).issubset(qte.columns))
    True

    See Also
    --------
    synth : Classic (mean-matching) synthetic control.
    qqsynth : Alias for ``discos(..., method='quantile')``.

    References
    ----------
    gunsilius2023distributional
    """
    if method is not None and method not in ("mixture", "quantile"):
        raise ValueError(  # pragma: no cover
            f"method must be 'mixture' or 'quantile', got '{method}'"
        )
    for col in (outcome, unit, time):
        if col not in data.columns:
            raise ValueError(f"column {col!r} not found in data")

    cell_sizes = data.groupby([unit, time]).size()
    if int(cell_sizes.max()) > 1:
        return _discos_micro(
            data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            method=method or "quantile",
            G=int(n_quantiles) if n_quantiles is not None else 1000,
            M=int(M),
            simplex=simplex,
            q_nodes=q_nodes,
            cdf_grid=cdf_grid,
            placebo=placebo,
            alpha=alpha,
        )

    warnings.warn(
        "sp.discos received one observation per unit-period. Gunsilius' "
        "distributional synthetic control needs individual-level data "
        "(several observations per unit-period); falling back to a "
        "heuristic that treats each unit's time series as its distribution. "
        "This is not the DiSCo estimator of Gunsilius (2023) / R DiSCos.",
        UserWarning,
        stacklevel=2,
    )
    method = method or "mixture"
    n_quantiles = 100 if n_quantiles is None else int(n_quantiles)

    # --- Build panel ---
    pivot = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(pivot.columns.tolist())
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise ValueError("DiSCo needs at least 2 pre-treatment periods")
    if len(post_times) < 1:
        raise ValueError("Need at least 1 post-treatment period")  # pragma: no cover

    donors = [u for u in pivot.index if u != treated_unit]
    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    if J < 2:
        raise ValueError("Need at least 2 control (donor) units")

    # --- Quantile grid ---
    tau_grid = np.linspace(
        1 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles
    )

    # --- Build quantile matrices for pre- and post-treatment ---
    # Q_treated_pre[t, q]: treated unit's q-th quantile at pre-period t
    # We use the time-series of outcomes up to each pre-period to form
    # expanding-window empirical quantile functions (rolling CDF from
    # the panel values at that cross-section). For standard panel data
    # with a single outcome per unit-time, we use the full pre-treatment
    # time-series per unit to construct one empirical quantile function.

    Y_treated_pre = pivot.loc[treated_unit, pre_times].values.astype(np.float64)
    Y_treated_post = pivot.loc[treated_unit, post_times].values.astype(np.float64)
    Y_donors_pre = pivot.loc[donors, pre_times].values.astype(np.float64)  # (J, T0)
    Y_donors_post = pivot.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    # Empirical quantile functions from the pre-treatment time-series
    Q_treated_pre = _empirical_quantile_function(Y_treated_pre, tau_grid)  # (n_q,)
    Q_treated_post = _empirical_quantile_function(Y_treated_post, tau_grid)  # (n_q,)

    Q_donors_pre = np.zeros((J, n_quantiles))  # (J, n_q)
    Q_donors_post = np.zeros((J, n_quantiles))  # (J, n_q)
    for j in range(J):
        Q_donors_pre[j] = _empirical_quantile_function(Y_donors_pre[j], tau_grid)
        Q_donors_post[j] = _empirical_quantile_function(Y_donors_post[j], tau_grid)

    # --- Fit weights ---
    if method == "mixture":
        weights = _mixture_weights(Q_treated_pre, Q_donors_pre)
    else:
        weights = _quantile_weights(Q_treated_pre, Q_donors_pre)

    # --- Counterfactual quantile function (post-treatment) ---
    Q_counterfactual_post = weights @ Q_donors_post  # (n_q,)
    Q_counterfactual_pre = weights @ Q_donors_pre  # (n_q,)

    # --- Distributional treatment effects ---
    quantile_effects = Q_treated_post - Q_counterfactual_post  # (n_q,)
    avg_qte = float(np.mean(quantile_effects))

    # Pre-treatment fit
    pre_residuals = Q_treated_pre - Q_counterfactual_pre
    pre_rmsqe = float(np.sqrt(np.mean(pre_residuals**2)))

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
            except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
                continue  # pragma: no cover

    # --- Standard errors and p-value ---
    if len(placebo_avg_qtes) > 0:
        se = float(np.std(placebo_avg_qtes, ddof=1))
        pvalue = placebo_rank_pvalue(abs(avg_qte), np.abs(placebo_avg_qtes))

        # Quantile-level CIs from placebo distribution
        if len(placebo_quantile_effects) > 0:
            plac_q_arr = np.array(placebo_quantile_effects)  # (n_plac, n_q)
            q_se = np.std(plac_q_arr, axis=0, ddof=1)  # (n_q,)
        else:
            q_se = np.full(n_quantiles, np.nan)  # pragma: no cover
    else:
        se = float(np.std(quantile_effects)) / max(np.sqrt(n_quantiles), 1)
        pvalue = np.nan
        q_se = np.full(n_quantiles, np.nan)

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (avg_qte - z_crit * se, avg_qte + z_crit * se)

    # --- Build quantile-level results table ---
    q_ci_lower = quantile_effects - z_crit * q_se
    q_ci_upper = quantile_effects + z_crit * q_se

    quantile_effects_df = pd.DataFrame(
        {
            "quantile": tau_grid,
            "effect": quantile_effects,
            "ci_lower": q_ci_lower,
            "ci_upper": q_ci_upper,
        }
    )

    # --- Gap table (period-level, using raw outcomes) ---
    Y_synth_pre = weights @ Y_donors_pre  # (T0,)
    Y_synth_post = weights @ Y_donors_post  # (T1,)
    Y_synth = np.concatenate([Y_synth_pre, Y_synth_post])
    Y_treated = np.concatenate([Y_treated_pre, Y_treated_post])

    gap_df = pd.DataFrame(
        {
            "time": all_times,
            "treated": Y_treated,
            "synthetic": Y_synth,
            "gap": Y_treated - Y_synth,
        }
    )

    # --- Effects by period ---
    effects_df = pd.DataFrame(
        {
            "time": post_times,
            "treated": Y_treated_post,
            "counterfactual": Y_synth_post,
            "effect": Y_treated_post - Y_synth_post,
        }
    )

    # --- Build model_info ---
    model_info: Dict[str, Any] = {
        "method_variant": method,
        "estimator": "time_series_quantiles_fallback",
        "n_quantiles": n_quantiles,
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "pre_rmsqe": pre_rmsqe,
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
        model_info["placebo_quantile_effects"] = np.array(placebo_quantile_effects)

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
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.qqsynth(df, outcome='packspercapita', unit='state',
    ...                     time='year', treated_unit='California',
    ...                     treatment_time=1989)
    >>> bool(result.estimate is not None)
    True

    See Also
    --------
    discos : Full distributional synthetic controls with method selection.

    References
    ----------
    gunsilius2023distributional
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
#  Individual-level DiSCo (Gunsilius 2023; R ``DiSCos`` conventions)
# ====================================================================== #


def _quant7_sorted(xs: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Type-7 quantiles of a sorted sample, operation-for-operation as
    R ``DiSCos:::quant7_sorted`` (``stats::quantile(type = 7)``)."""
    n = xs.shape[0]
    if n == 1:
        return np.full(probs.shape[0], xs[0])
    index = 1.0 + (n - 1) * probs
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    qs = xs[lo - 1].copy()
    h = index - lo
    m = (index > lo) & (xs[hi - 1] != qs)
    qs[m] = (1.0 - h[m]) * qs[m] + h[m] * xs[hi - 1][m]
    return np.asarray(qs, dtype=np.float64)


def _ecdf_sorted(xs: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Empirical CDF of a sorted sample at ``v`` (R ``stats::ecdf``)."""
    return np.asarray(np.searchsorted(xs, v, side="right") / xs.shape[0])


def _eq_bounded_lsq(C: np.ndarray, t: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """``min ||C w - t||^2  s.t.  sum(w) = 1, lb <= w <= ub`` (exact).

    The problem R ``pracma::lsqlincon`` hands to ``quadprog::solve.QP``
    inside ``DiSCos:::DiSCo_weights_reg``; ``lb`` may be ``-inf``. Primal
    active-set method: on each working set the equality-constrained least
    squares problem is solved in the null space of the adding-up
    constraint by ``lstsq`` on ``C`` itself (``C'C`` is never formed, so the
    conditioning is not squared); a blocking bound is added on a step that
    leaves the box, and the bound with the most negative multiplier is
    released at a stationary point.
    """
    n, J = C.shape
    # quadprog needs C'C positive definite (R errors otherwise). A rank-
    # deficient problem has a non-unique minimiser; a ridge of 1e-10 of the
    # mean squared column norm then selects one deterministically (the
    # aggregate-panel fallback, where J donors can exceed the number of
    # distinct quantile values).
    if np.linalg.matrix_rank(C) < J:
        delta = 1e-10 * float(np.mean(np.sum(C**2, axis=0)))
        C = np.vstack([C, np.sqrt(delta) * np.eye(J)])
        t = np.concatenate([t, np.zeros(J)])
    scale = max(1.0, float(np.abs(C.T @ t).max()), float(np.sum(C**2, axis=0).max()))
    tol = 1e-12 * scale
    feas = 1e-13
    w = np.full(J, 1.0 / J)
    at_lb = np.zeros(J, dtype=bool)
    at_ub = np.zeros(J, dtype=bool)
    last_released = -1
    for _ in range(50 * J + 50):
        free = ~(at_lb | at_ub)
        F = np.flatnonzero(free)
        fixed = np.flatnonzero(~free)
        fixed_val = np.where(at_lb, lb, ub)[fixed]
        nF = F.size
        r = t - C[:, fixed] @ fixed_val
        s_F = 1.0 - float(fixed_val.sum())
        base = np.full(nF, s_F / nF)
        if nF > 1:
            Q = np.linalg.qr(np.column_stack([np.ones(nF), np.eye(nF)[:, : nF - 1]]))[0]
            Z = Q[:, 1:]
            CF = C[:, F]
            v = np.linalg.lstsq(CF @ Z, r - CF @ base, rcond=None)[0]
            wF = base + Z @ v
        else:
            wF = base
        cur = w[F]
        lo_v = wF < lb - feas
        hi_v = wF > ub + feas
        if lo_v.any() or hi_v.any():
            step_dir = wF - cur
            ratios = np.full(nF, np.inf)
            ratios[lo_v] = (lb - cur[lo_v]) / step_dir[lo_v]
            ratios[hi_v] = (ub - cur[hi_v]) / step_dir[hi_v]
            k = int(np.argmin(ratios))
            if F[k] == last_released and ratios[k] <= 0.0:
                # releasing that bound gives no descent: it is optimal
                return np.clip(w, lb, ub)
            w[F] = cur + max(ratios[k], 0.0) * step_dir
            if lo_v[k]:
                w[F[k]] = lb
                at_lb[F[k]] = True
            else:
                w[F[k]] = ub
                at_ub[F[k]] = True
            last_released = -1
            continue
        w[F] = np.clip(wF, lb, ub)
        grad = C.T @ (C @ w - t)
        mu = -float(np.mean(grad[F]))
        red = grad + mu  # >= 0 at a lower bound, <= 0 at an upper bound
        bad = np.flatnonzero((at_lb & (red < -tol)) | (at_ub & (red > tol)))
        if bad.size == 0:
            return w
        k = int(bad[np.argmax(np.abs(red[bad]))])
        at_lb[k] = False
        at_ub[k] = False
        last_released = k
    raise RuntimeError("DiSCo weight regression (active set) did not converge")


def _disco_quantile_weights(
    controls: Sequence[np.ndarray],
    target: np.ndarray,
    nodes: np.ndarray,
    simplex: bool,
) -> np.ndarray:
    """Per-period quantile weights of R ``DiSCo_weights_reg`` (sorted data)."""
    C = np.column_stack([_quant7_sorted(c, nodes) for c in controls])
    tq = _quant7_sorted(target, nodes)
    return _eq_bounded_lsq(C, tq, 0.0 if simplex else -np.inf, 1.0)


def _disco_mixture_weights(Fc: np.ndarray, ft: np.ndarray, simplex: bool) -> np.ndarray:
    """``min ||Fc w - ft||_1  s.t.  sum(w) = 1 (, w >= 0)`` — the linear
    programme R ``DiSCo_mixture_solve`` gives to CVXR/SCS, solved exactly
    with HiGHS."""
    from scipy import sparse

    G, J = Fc.shape
    cost = np.concatenate([np.zeros(J), np.ones(G)])
    eye = sparse.identity(G, format="csr")
    A_ub = sparse.vstack(
        [
            sparse.hstack([sparse.csr_matrix(Fc), -eye]),
            sparse.hstack([sparse.csr_matrix(-Fc), -eye]),
        ],
        format="csr",
    )
    b_ub = np.concatenate([ft, -ft])
    A_eq = np.concatenate([np.ones(J), np.zeros(G)])[None, :]
    bounds = [(0.0, None) if simplex else (None, None)] * J + [(0.0, None)] * G
    res = linprog(
        cost,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"DiSCo mixture LP failed: {res.message}")
    return np.asarray(res.x[:J])


def _invert_cdf(cdf: np.ndarray, grid: np.ndarray, evgrid: np.ndarray) -> np.ndarray:
    """R DiSCo's CDF inversion: smallest grid point with
    ``cdf >= tau - 1e-5`` (NaN when none)."""
    target = evgrid - 1e-5
    if np.all(np.diff(cdf) >= 0):
        idx = np.searchsorted(cdf, target, side="left")
    else:
        idx = np.array(
            [
                (np.flatnonzero(cdf >= v)[0] if np.any(cdf >= v) else cdf.size)
                for v in target
            ]
        )
    out = np.full(evgrid.shape[0], np.nan)
    ok = idx < cdf.size
    out[ok] = grid[idx[ok]]
    return out


def _default_cdf_grid(
    target: np.ndarray, controls: Sequence[np.ndarray], G: int
) -> np.ndarray:
    """Equispaced version of R ``DiSCos:::getGrid``'s uniform grid."""
    lo = min(float(target.min()), min(float(c.min()) for c in controls))
    hi = max(float(target.max()), max(float(c.max()) for c in controls))
    lo = np.floor(lo * 10) / 10
    hi = np.ceil(hi * 10) / 10
    return np.linspace(lo - 0.25, hi + 0.25, G)


def _disco_fit(
    target: List[np.ndarray],
    controls: List[List[np.ndarray]],
    T0: int,
    evgrid: np.ndarray,
    method: str,
    simplex: bool,
    node_fn: Callable[[int], np.ndarray],
    grids: Optional[List[np.ndarray]],
) -> Dict[str, Any]:
    """One DiSCo fit (target vs controls) over all periods.

    ``target[t]`` / ``controls[t][j]`` are sorted samples; the first ``T0``
    periods are pre-treatment. Returns weights, per-period weights,
    counterfactual / target quantiles on ``evgrid`` and per-period squared
    Wasserstein distances (R ``DiSCo_per``'s ``distt``).
    """
    n_per = len(target)
    if method == "quantile":
        per_w = [
            _disco_quantile_weights(controls[t], target[t], node_fn(t), simplex)
            for t in range(T0)
        ]
    else:
        assert grids is not None
        Fcs = [
            np.column_stack([_ecdf_sorted(c, grids[t]) for c in controls[t]])
            for t in range(n_per)
        ]
        fts = [_ecdf_sorted(target[t], grids[t]) for t in range(n_per)]
        per_w = [_disco_mixture_weights(Fcs[t], fts[t], simplex) for t in range(T0)]
    w = np.sum(per_w, axis=0) / T0
    tq = [_quant7_sorted(target[t], evgrid) for t in range(n_per)]
    if method == "quantile":
        cq = [
            np.column_stack([_quant7_sorted(c, evgrid) for c in controls[t]]) @ w
            for t in range(n_per)
        ]
        dist = [float(np.mean((cq[t] - tq[t]) ** 2)) for t in range(n_per)]
    else:
        assert grids is not None
        cdf_cf = [Fcs[t] @ w for t in range(n_per)]
        cq = [_invert_cdf(cdf_cf[t], grids[t], evgrid) for t in range(n_per)]
        dist = [float(np.mean((cdf_cf[t] - fts[t]) ** 2)) for t in range(n_per)]
    return {
        "weights": w,
        "period_weights": per_w,
        "target_q": tq,
        "cf_q": cq,
        "dist": dist,
    }


def _disco_tea_table(
    effects: List[np.ndarray],
    times: List[Any],
    post_idx: List[int],
    G: int,
    samples: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> pd.DataFrame:
    """Unrounded version of R ``DiSCoTEA(agg = 'quantileDiff')``'s table:
    mean quantile effect over quantile ranges, per post-period."""
    evgrid = np.linspace(0.0, 1.0, G + 1)
    rows = []
    grid_q = [s * G + 1 for s in samples]  # 1-based, R semantics
    for i in post_idx:
        for a, b in zip(grid_q[:-1], grid_q[1:]):
            idx = np.floor(np.arange(a, b + 1e-9, 1.0)).astype(int) - 1
            rows.append(
                {
                    "time": times[i],
                    "q_from": float(evgrid[int(a) - 1]),
                    "q_to": float(evgrid[int(b) - 1]),
                    "effect": float(np.mean(effects[i][idx])),
                }
            )
    return pd.DataFrame(rows)


def _discos_micro(
    data: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    method: str,
    G: int,
    M: int,
    simplex: bool,
    q_nodes: Optional[Union[np.ndarray, Sequence[np.ndarray]]],
    cdf_grid: Optional[Sequence[np.ndarray]],
    placebo: bool,
    alpha: float,
    placebo_node_fn: Optional[Callable[[int, int], np.ndarray]] = None,
) -> CausalResult:
    """Gunsilius (2023) distributional SC on individual-level data."""
    if G < 2:
        raise ValueError("n_quantiles (G) must be >= 2")
    if M < 1:
        raise ValueError("M must be >= 1")
    df = data[[unit, time, outcome]]
    if df[outcome].isna().any():
        raise ValueError(f"{outcome!r} contains missing values")
    times = sorted(df[time].unique().tolist())
    pre_idx = [i for i, t in enumerate(times) if t < treatment_time]
    post_idx = [i for i, t in enumerate(times) if t >= treatment_time]
    if len(pre_idx) < 2:
        # R DiSCo accepts a single pre-period; StatsPAI keeps its long-standing
        # requirement of two so that the pre-period fit can be assessed.
        raise ValueError("DiSCo needs at least 2 pre-treatment periods")
    if not post_idx:
        raise ValueError("Need at least 1 post-treatment period")
    T0 = len(pre_idx)
    units_all = sorted(df[unit].unique().tolist())
    if treated_unit not in units_all:
        raise ValueError(f"treated unit {treated_unit!r} not found")
    donors = [u for u in units_all if u != treated_unit]
    J = len(donors)
    if J < 2:
        raise ValueError("Need at least 2 control (donor) units")
    cells = {
        key: np.sort(grp.to_numpy(dtype=np.float64))
        for key, grp in df.groupby([unit, time])[outcome]
    }
    target: List[np.ndarray] = []
    controls: List[List[np.ndarray]] = []
    for t in times:
        if (treated_unit, t) not in cells:
            raise ValueError(f"treated unit has no observations in period {t!r}")
        tv = cells[(treated_unit, t)]
        cv = []
        for u in donors:
            if (u, t) not in cells:
                raise ValueError(f"donor {u!r} has no observations in period {t!r}")
            cv.append(cells[(u, t)])
        if tv.shape[0] < J:
            raise ValueError(
                f"period {t!r}: the treated unit has {tv.shape[0]} observations "
                f"but there are {J} weights to estimate (DiSCo needs at least "
                "as many observations as donors)"
            )
        target.append(tv)
        controls.append(cv)

    # --- quantile nodes (pre-periods) and CDF grids (all periods) ---
    if q_nodes is None:
        base_nodes = (np.arange(M) + 0.5) / M
        nodes_by_t = [base_nodes] * T0
    else:
        arr = q_nodes
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            nodes_by_t = [np.asarray(arr, dtype=float)] * T0
        else:
            nodes_by_t = [np.asarray(a, dtype=float) for a in arr]
            if len(nodes_by_t) != T0:
                raise ValueError(
                    f"q_nodes must hold one node array per pre-period ({T0})"
                )
    if any(np.any((nd < 0) | (nd > 1)) for nd in nodes_by_t):
        raise ValueError("q_nodes must lie in [0, 1]")
    grids: Optional[List[np.ndarray]] = None
    if method == "mixture":
        if cdf_grid is None:
            grids = [
                _default_cdf_grid(target[t], controls[t], G) for t in range(len(times))
            ]
        else:
            grids = [np.sort(np.asarray(g, dtype=float)) for g in cdf_grid]
            if len(grids) != len(times):
                raise ValueError(
                    f"cdf_grid must hold one grid per period ({len(times)})"
                )

    evgrid = np.linspace(0.0, 1.0, G + 1)
    fit = _disco_fit(
        target,
        controls,
        T0,
        evgrid,
        method,
        simplex,
        lambda t: nodes_by_t[t],
        grids,
    )
    weights = fit["weights"]
    effects = [fit["target_q"][t] - fit["cf_q"][t] for t in range(len(times))]
    if any(np.isnan(e).any() for e in effects):
        warnings.warn(
            "DiSCo mixture counterfactual CDF never reaches some quantile "
            "levels (negative weights); those quantile effects are NaN.",
            RuntimeWarning,
            stacklevel=3,
        )
    post_eff = np.vstack([effects[i] for i in post_idx])
    estimate = float(np.mean(post_eff))

    # --- permutation inference (R DiSCo_per) ---
    pvalue = np.nan
    se = np.nan
    placebo_avg: List[float] = []
    placebo_q_eff: List[np.ndarray] = []
    perm_dist: List[List[float]] = []
    ratio_stats: Optional[np.ndarray] = None
    if placebo:
        for idx in range(J):
            keep = [k for k in range(J) if k != idx]
            p_target = [controls[t][idx] for t in range(len(times))]
            p_controls = [
                [target[t]] + [controls[t][k] for k in keep] for t in range(len(times))
            ]
            if placebo_node_fn is None:
                node_fn = lambda t: nodes_by_t[t]  # noqa: E731
            else:
                node_fn = (lambda i: (lambda t: placebo_node_fn(i, t)))(idx)
            pf = _disco_fit(
                p_target, p_controls, T0, evgrid, method, simplex, node_fn, grids
            )
            perm_dist.append(pf["dist"])
            p_eff = np.vstack([pf["target_q"][i] - pf["cf_q"][i] for i in post_idx])
            placebo_avg.append(float(np.mean(p_eff)))
            placebo_q_eff.append(p_eff.mean(axis=0))
        dist_all = np.vstack(perm_dist + [fit["dist"]])
        ratio_stats = np.sqrt(dist_all[:, post_idx].mean(axis=1)) / np.sqrt(
            dist_all[:, pre_idx].mean(axis=1)
        )
        # R: rank(-R)[target] / (J + 1), average ranks for ties
        pvalue = float(rankdata(-ratio_stats, method="average")[-1] / (J + 1))
        se = float(np.std(placebo_avg, ddof=1))

    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - z_crit * se, estimate + z_crit * se)

    avg_tq = np.mean([fit["target_q"][i] for i in post_idx], axis=0)
    avg_cq = np.mean([fit["cf_q"][i] for i in post_idx], axis=0)
    avg_eff = post_eff.mean(axis=0)
    if placebo_q_eff:
        q_se = np.std(np.vstack(placebo_q_eff), axis=0, ddof=1)
    else:
        q_se = np.full(evgrid.shape[0], np.nan)
    quantile_effects_df = pd.DataFrame(
        {
            "quantile": evgrid,
            "effect": avg_eff,
            "ci_lower": avg_eff - z_crit * q_se,
            "ci_upper": avg_eff + z_crit * q_se,
        }
    )
    by_period = pd.concat(
        [
            pd.DataFrame(
                {
                    "time": times[i],
                    "quantile": evgrid,
                    "treated": fit["target_q"][i],
                    "counterfactual": fit["cf_q"][i],
                    "effect": effects[i],
                }
            )
            for i in range(len(times))
        ],
        ignore_index=True,
    )
    treated_mean = np.array([float(np.mean(target[i])) for i in range(len(times))])
    synth_mean = np.array([float(np.mean(fit["cf_q"][i])) for i in range(len(times))])
    gap_df = pd.DataFrame(
        {
            "time": times,
            "treated": treated_mean,
            "synthetic": synth_mean,
            "gap": treated_mean - synth_mean,
        }
    )
    effects_df = pd.DataFrame(
        {
            "time": [times[i] for i in post_idx],
            "treated": treated_mean[post_idx],
            "counterfactual": synth_mean[post_idx],
            "effect": [float(np.mean(effects[i])) for i in post_idx],
        }
    )
    period_w = pd.DataFrame(
        np.vstack(fit["period_weights"]),
        index=pd.Index([times[i] for i in pre_idx], name=time),
        columns=donors,
    )
    model_info: Dict[str, Any] = {
        "estimator": "gunsilius_disco",
        "method_variant": method,
        "simplex": simplex,
        "M": M,
        "G": G,
        "n_quantiles": G,
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": len(post_idx),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "weights": dict(zip(donors, weights)),
        "period_weights": period_w,
        "tau_grid": evgrid,
        "quantile_effects": quantile_effects_df,
        "quantile_effects_by_period": by_period,
        "quantile_effect_summary": _disco_tea_table(effects, times, post_idx, G),
        "treated_quantiles": avg_tq,
        "counterfactual_quantiles": avg_cq,
        "wasserstein_sq": dict(zip(times, fit["dist"])),
        "pre_rmsqe": float(np.sqrt(np.mean([fit["dist"][i] for i in pre_idx]))),
        "gap_table": gap_df,
        "effects_by_period": effects_df,
        "Y_synth": synth_mean,
        "Y_treated": treated_mean,
        "times": times,
    }
    if placebo:
        model_info["placebo_atts"] = placebo_avg
        model_info["n_placebos"] = len(placebo_avg)
        model_info["placebo_quantile_effects"] = np.vstack(placebo_q_eff)
        model_info["permutation"] = {
            "p_value": pvalue,
            "rmspe_ratio": ratio_stats,
            "placebo_wasserstein_sq": np.vstack(perm_dist),
        }

    return CausalResult(
        method="Distributional Synthetic Controls (Gunsilius 2023)",
        estimand="Distributional ATT",
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info=model_info,
        _citation_key="discos",
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
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.discos(df, outcome='packspercapita', unit='state',
    ...                    time='year', treated_unit='California',
    ...                    treatment_time=1989)
    >>> out = sp.discos_test(result, test='ks')
    >>> out['test']
    'Kolmogorov-Smirnov'
    >>> bool('pvalue' in out)
    True
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
            f"test must be 'ks', 'cvm', or 'stochastic_dominance', " f"got '{test}'"
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
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.discos(df, outcome='packspercapita', unit='state',
    ...                    time='year', treated_unit='California',
    ...                    treatment_time=1989)
    >>> out = sp.stochastic_dominance(result, order=1)
    >>> bool('dominates' in out)
    True
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
    ax: Any = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = "#2C3E50",
    ci_alpha: float = 0.2,
    title: Optional[str] = None,
) -> Any:
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
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.discos(df, outcome='packspercapita', unit='state',
    ...                    time='year', treated_unit='California',
    ...                    treatment_time=1989)
    >>> fig, ax = sp.discos_plot(result, type='quantile_effect')
    >>> fig2, ax2 = sp.discos_plot(result, type='quantile_comparison')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "matplotlib required for plotting. " "Install: pip install matplotlib"
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

        ax.fill_between(
            tau,
            ci_lo,
            ci_hi,
            alpha=ci_alpha,
            color=color,
            label=f"{int(100 * (1 - result.alpha))}% CI",
        )
        ax.plot(tau, eff, color=color, linewidth=1.5, label="Δ(τ)")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.axhline(
            y=result.estimate,
            color="#E74C3C",
            linestyle=":",
            linewidth=1,
            alpha=0.6,
            label=f"Mean = {result.estimate:.4f}",
        )
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

        ax.plot(tau, Q_tr, color=color, linewidth=1.5, label="Treated (observed)")
        ax.plot(
            tau,
            Q_cf,
            color="#E74C3C",
            linewidth=1.5,
            linestyle="--",
            label="Counterfactual (DiSCo)",
        )
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

        ax.plot(times, gaps, color=color, linewidth=1.5, marker="o", markersize=4)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(
            x=treatment_time,
            color="#E74C3C",
            linestyle=":",
            linewidth=1,
            alpha=0.6,
            label="Treatment onset",
        )
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Gap (Treated - Synthetic)", fontsize=11)
        ax.set_title(title or "Gap Plot", fontsize=13)
        ax.legend(fontsize=9, frameon=False)

    elif type == "weights":
        weights = mi["weights"]
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        # Show donors with weight > 0.001
        sorted_w = [(k, v) for k, v in sorted_w if v > 0.001]
        if not sorted_w:
            sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]

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
    return np.asarray(np.quantile(y_clean, tau_grid))


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
    # Exact active-set solution of the simplex least-squares problem. The
    # previous finite-difference SLSQP ran to its iteration cap on this
    # badly scaled objective (~30 s for the Prop. 99 placebos) and stopped
    # ~1e-5 short of the optimum.
    return _eq_bounded_lsq(Q_donors.T, Q_treated, 0.0, 1.0)


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
    Qy = Q_donors @ Q_treated  # (J,)
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
        plac_cvm = np.mean(plac_arr**2, axis=1)
        pval = placebo_rank_pvalue(cvm_stat, plac_cvm)
    else:
        # Asymptotic: treat as chi-squared approximation
        # Under H0, n_q * CvM ~ sum of squared normals
        pval = float(
            sp_stats.chi2.sf(n_q * cvm_stat / max(np.var(Q_treated), 1e-10), df=n_q)
        )

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
        # H0: no dominance. p = rank of the observed min_gap among itself
        # and the placebo min_gaps, divided by J+1.
        pval = placebo_rank_pvalue(min_gap, plac_min_gaps)
    else:
        # Approximate: use KS test as fallback
        ks_stat, pval = sp_stats.ks_2samp(
            Q_treated, Q_counterfactual, alternative="less"
        )
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
        pval = placebo_rank_pvalue(min_cum_gap, plac_min_cum)
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
