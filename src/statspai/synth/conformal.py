"""
Conformal Inference for Synthetic Control Methods.

Provides distribution-free, finite-sample valid inference for
treatment effects in SCM settings using conformal prediction.

Instead of relying on large-sample asymptotics or ad-hoc placebo
permutations, conformal inference tests H0: τ_t = τ0 for each
post-treatment period by checking whether the residual from the
hypothesised effect is exchangeable with pre-treatment residuals.

References
----------
Chernozhukov, V., Wuthrich, K. and Zhu, Y. (2021).
"An Exact and Robust Conformal Inference Method for Counterfactual
and Synthetic Controls."
*Journal of the American Statistical Association*, 116(536), 1849-1864. [@chernozhukov2021exact]
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility


@accepts_aliases(id="unit", y="outcome")
def conformal_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    scm_method: str = "classic",
    grid_size: int = 101,
    grid_range: Optional[Tuple[float, float]] = None,
    alpha: float = 0.05,
    penalization: float = 0.0,
) -> CausalResult:
    """
    Conformal inference for synthetic control (Chernozhukov, Wüthrich & Zhu).

    Tests sharp null hypotheses about the treated unit's post-period
    effects by re-estimating the synthetic control *under each null* on all
    periods and permuting the resulting residuals over time, and builds
    confidence sets by inverting those tests on a grid. This is the
    procedure of the authors' R package ``scinference``
    (``estimation_method = "sc"``, moving-block permutations, ``q = 1``).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data (one row per unit and period).
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    scm_method : {'classic', 'ridge'}, default 'classic'
        Label of the counterfactual model. Both fit simplex-constrained
        least squares of the treated outcome on the donor outcomes (no
        intercept, ``w >= 0``, ``sum(w) = 1``) plus
        ``penalization * ||w||^2``; ``'classic'`` with ``penalization = 0``
        is ``scinference``'s ``sc`` estimator.
    grid_size : int, default 101
        Number of points in the hypothesis grid for CI inversion.
    grid_range : tuple of (float, float), optional
        (min, max) of the hypothesis grid. Default: the point estimate
        +/- max(5 x SD of the pre-period SC residuals, 3 x the largest
        absolute post-period gap). A confidence set reaching an end of the
        grid is truncated there (flagged in ``model_info['ci_truncated']``,
        with a warning). When ``alpha`` is below the smallest attainable
        p-value (``1/(T0+1)`` pointwise, ``1/(T0+T1)`` joint) nothing can be
        rejected and the set is ``(-inf, inf)``.
    alpha : float, default 0.05
        Significance level (``scinference`` defaults to 0.1).
    penalization : float, default 0.0
        Ridge penalty on the weights.

    Returns
    -------
    CausalResult
        * ``estimate`` -- average post-period gap of the synthetic control
          fitted on the pre-period (the usual SC estimate).
        * ``pvalue`` -- moving-block conformal p-value of the joint null of
          no effect in any post-period (``scinference(..., theta0 = 0)``):
          the SC is fitted on all ``T0 + T1`` periods, and the statistic
          ``sum_{post} |u_t|`` is compared with the same sum over all
          ``T0 + T1`` cyclic blocks of length ``T1``.
        * ``ci`` -- set of constant effects ``tau0`` (post outcomes shifted
          by ``tau0``) whose joint p-value exceeds ``alpha``, reported as
          ``(min, max)`` of the accepted grid points.
        * ``se`` -- ``ci`` width / (2 z_{1-alpha/2}); a CI-implied scale,
          not a standard error.
        * ``detail`` / ``model_info['period_results']`` -- per post-period
          effect (pre-fit gap), pointwise conformal p-value of ``tau_t = 0``
          and pointwise CI (``scinference``'s ``ci = TRUE``: for each period
          the SC is refit on the ``T0`` pre-periods plus that period under
          each grid value, and ``p = mean(|u_s| >= |u_{T0+1}|)``).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()  # state, year, packspercapita, treated
    >>> result = sp.conformal_synth(df, outcome='packspercapita',
    ...     unit='state', time='year', treated_unit='California',
    ...     treatment_time=1989)
    >>> print(result.summary())  # doctest: +SKIP

    References
    ----------
    [@chernozhukov2021exact]
    """
    if scm_method not in ("classic", "ridge"):
        raise MethodIncompatibility(
            f"scm_method must be 'classic' or 'ridge', got {scm_method!r}"
        )
    if grid_size < 2:
        raise MethodIncompatibility("grid_size must be at least 2")

    # --- Build panel ---
    if data.duplicated(subset=[time, unit]).any():
        raise MethodIncompatibility(
            "conformal_synth needs one row per (unit, time); found duplicates.",
            recovery_hint="Aggregate or drop duplicate (unit, time) rows.",
        )
    pivot = data.pivot(index=time, columns=unit, values=outcome).sort_index()
    times = pivot.index.values
    pre_mask = times < treatment_time
    post_mask = times >= treatment_time

    if pre_mask.sum() < 2:
        raise DataInsufficient(
            "Need at least 2 pre-treatment periods",
            recovery_hint=(
                "Conformal SC inference needs ≥ 2 pre-treatment observations "
                "to form residuals; consider sp.did or sp.causal_impact "
                "with fewer periods."
            ),
            diagnostics={"n_pre_periods": int(pre_mask.sum())},
            alternative_functions=["sp.did", "sp.causal_impact"],
        )
    if post_mask.sum() < 1:
        raise DataInsufficient(
            "Need at least 1 post-treatment period",
            recovery_hint=("Verify the treatment_time is before the panel's end."),
            diagnostics={"n_post_periods": int(post_mask.sum())},
            alternative_functions=[],
        )

    Y_treated = pivot[treated_unit].values.astype(np.float64)
    if np.isnan(Y_treated).any():
        raise DataInsufficient(
            "The treated unit has missing outcomes.",
            recovery_hint="Impute or drop the periods with a missing treated outcome.",
        )
    donor_cols = [c for c in pivot.columns if c != treated_unit]
    Y_donors = pivot[donor_cols].values.astype(np.float64)

    # Drop donors with any missing period (the conformal fits use all periods)
    valid = ~np.any(np.isnan(Y_donors), axis=0)
    if valid.sum() == 0:
        raise ValueError("No valid donor units")  # pragma: no cover
    if not valid.all():
        import warnings

        warnings.warn(
            f"Dropped {int((~valid).sum())} donor(s) with missing outcomes.",
            UserWarning,
            stacklevel=2,
        )
    Y_donors = Y_donors[:, valid]
    donor_cols = [donor_cols[i] for i in range(len(donor_cols)) if valid[i]]

    T0 = int(pre_mask.sum())
    T1 = int(post_mask.sum())
    # Order periods as pre then post (the moving blocks are over time order).
    order = np.concatenate([np.flatnonzero(pre_mask), np.flatnonzero(post_mask)])
    y_all = Y_treated[order]
    X_all = Y_donors[order]

    # --- Point estimate: SC fitted on the pre-period ---
    weights = _solve_weights(y_all[:T0], X_all[:T0], penalization)
    gap = y_all - X_all @ weights
    gap_pre = gap[:T0]
    gap_post = gap[T0:]
    att = float(np.mean(gap_post))

    # Determine grid
    if grid_range is None:
        pre_scale = float(np.std(gap_pre))
        pre_scale = pre_scale if pre_scale > 0 else 1.0
        half = max(5.0 * pre_scale, 3.0 * float(np.max(np.abs(gap_post))))
        grid_lo = att - half
        grid_hi = att + half
    else:
        grid_lo, grid_hi = grid_range
    tau_grid = np.linspace(grid_lo, grid_hi, grid_size)

    # --- Pointwise tests and confidence sets ---
    # The smallest attainable pointwise p-value is 1/(T0+1) and the smallest
    # joint one 1/(T0+T1): below those levels no value can be rejected and
    # the confidence set is the whole real line.
    pointwise_unbounded = 1.0 / (T0 + 1) > alpha
    joint_unbounded = 1.0 / (T0 + T1) > alpha
    truncated = False
    period_results = []
    for t in range(T1):
        idx = np.concatenate([np.arange(T0), [T0 + t]])
        y_t, X_t = y_all[idx], X_all[idx]
        p0 = _pointwise_pvalue(y_t, X_t, 0.0, penalization)
        if pointwise_unbounded:
            lo, hi = -np.inf, np.inf
        else:
            pv = np.array(
                [_pointwise_pvalue(y_t, X_t, g, penalization) for g in tau_grid]
            )
            lo, hi, trunc = _accepted_range(tau_grid, pv, alpha)
            truncated = truncated or trunc
        period_results.append(
            {
                "time": times[order[T0 + t]],
                "effect": float(gap_post[t]),
                "pvalue": p0,
                "ci_lower": lo,
                "ci_upper": hi,
            }
        )
    period_df = pd.DataFrame(period_results)

    # --- Joint moving-block test of a constant effect ---
    avg_pvalue = _movingblock_pvalue(y_all, X_all, T0, T1, 0.0, penalization)
    pv_joint = np.array(
        [_movingblock_pvalue(y_all, X_all, T0, T1, g, penalization) for g in tau_grid]
    )
    if joint_unbounded:
        avg_ci_lo, avg_ci_hi = -np.inf, np.inf
    else:
        avg_ci_lo, avg_ci_hi, trunc = _accepted_range(tau_grid, pv_joint, alpha)
        truncated = truncated or trunc
    if truncated:
        import warnings

        warnings.warn(
            "A conformal confidence set reaches an end of the hypothesis grid "
            f"[{grid_lo:.6g}, {grid_hi:.6g}] and is truncated there; widen "
            "grid_range.",
            UserWarning,
            stacklevel=2,
        )

    # CI-implied scale (not a standard error)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_approx = (avg_ci_hi - avg_ci_lo) / (2 * z_crit)

    model_info = {
        "inference_method": "conformal (moving block, q=1)",
        "scm_method": scm_method,
        "n_donors": len(donor_cols),
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "pre_treatment_rmse": float(np.sqrt(np.mean(gap_pre**2))),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "grid_size": grid_size,
        "grid_range": (grid_lo, grid_hi),
        "ci_truncated": bool(truncated),
        "pointwise_ci_unbounded": bool(pointwise_unbounded),
        "joint_ci_unbounded": bool(joint_unbounded),
        "joint_pvalue_grid": pd.DataFrame({"tau0": tau_grid, "pvalue": pv_joint}),
        "period_results": period_df,
        "weights": dict(zip(donor_cols, weights)),
        "Y_synth": Y_donors @ weights,
        "Y_treated": Y_treated,
        "times": times,
    }

    return CausalResult(
        method="Conformal Synthetic Control (Chernozhukov et al. 2021)",
        estimand="ATT",
        estimate=att,
        se=se_approx,
        pvalue=avg_pvalue,
        ci=(avg_ci_lo, avg_ci_hi),
        alpha=alpha,
        n_obs=len(Y_treated),
        detail=period_df,
        model_info=model_info,
        _citation_key="conformal_synth",
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #


def _solve_weights(
    y: np.ndarray,
    X: np.ndarray,
    penalization: float = 0.0,
) -> np.ndarray:
    """Standard SCM weights: min ||y - Xw||^2 + pen*||w||^2, w>=0, sum=1."""
    from ._core import solve_simplex_weights

    return solve_simplex_weights(y, X, penalization=penalization)


def _null_residuals(
    y: np.ndarray, X: np.ndarray, n_pre: int, tau0: float, penalization: float
) -> np.ndarray:
    """Residuals of the SC refit on all rows after imposing ``tau = tau0``.

    Rows ``n_pre:`` are post-periods; their outcome is shifted by ``tau0``
    before the fit (``scinference``: ``Y1_0[post] <- Y1[post] - theta0``).
    """
    y0 = np.array(y, dtype=np.float64, copy=True)
    y0[n_pre:] -= tau0
    w = _solve_weights(y0, X, penalization)
    return y0 - X @ w


def _pointwise_pvalue(
    y: np.ndarray, X: np.ndarray, tau0: float, penalization: float
) -> float:
    """``scinference`` pointwise p-value: rows = T0 pre-periods + one post.

    ``p = mean(|u_s| >= |u_{T0+1}|)`` over the ``T0 + 1`` null residuals.
    """
    n_pre = len(y) - 1
    u = np.abs(_null_residuals(y, X, n_pre, tau0, penalization))
    return float(np.mean(u >= u[n_pre]))


def _movingblock_pvalue(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    T1: int,
    tau0: float,
    penalization: float,
) -> float:
    """``scinference`` moving-block p-value (``q = 1``) of ``tau_t = tau0``.

    ``S_s = sum_{j=0}^{T1-1} |u_{(s+j) mod T}|`` for ``s = 0..T-1``
    (``T = T0 + T1``); ``p = mean(S_s >= S_{T0})``.
    """
    u = np.abs(_null_residuals(y, X, T0, tau0, penalization))
    uc = np.concatenate([u, u])
    T = T0 + T1
    S = np.array([uc[s : s + T1].sum() for s in range(T)])
    return float(np.mean(S >= S[T0]))


def _accepted_range(
    tau_grid: np.ndarray, pvals: np.ndarray, alpha: float
) -> Tuple[float, float, bool]:
    """(min, max) of grid points with ``p > alpha``; truncation flag.

    An empty accepted set gives ``(nan, nan)`` (every grid value rejected).
    """
    acc = tau_grid[pvals > alpha]
    if acc.size == 0:
        return float("nan"), float("nan"), False
    truncated = bool(pvals[0] > alpha or pvals[-1] > alpha)
    return float(acc.min()), float(acc.max()), truncated


# Citation
CausalResult._CITATIONS["conformal_synth"] = (
    "@article{chernozhukov2021exact,\n"
    "  title={An Exact and Robust Conformal Inference Method for\n"
    "  Counterfactual and Synthetic Controls},\n"
    '  author={Chernozhukov, Victor and W{\\"u}thrich, Kaspar '
    "and Zhu, Yinchu},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1849--1864},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
