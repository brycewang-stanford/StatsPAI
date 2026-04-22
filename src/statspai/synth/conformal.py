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
*Journal of the American Statistical Association*, 116(536), 1849-1864.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult


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
    Conformal inference for synthetic control.

    Constructs valid confidence intervals by inverting a sequence of
    conformal tests, one for each hypothesised treatment effect.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
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
    scm_method : str, default 'classic'
        Which SCM variant to use for weight estimation.
        Currently supports 'classic' (constrained) and 'ridge'.
    grid_size : int, default 101
        Number of points in the hypothesis grid for CI inversion.
    grid_range : tuple of (float, float), optional
        (min, max) of the hypothesis grid. If None, auto-determined
        from pre-treatment residual scale.
    alpha : float, default 0.05
        Significance level.
    penalization : float, default 0.0
        Ridge penalty (used when scm_method='ridge').

    Returns
    -------
    CausalResult
        With ``model_info`` containing per-period p-values,
        conformal confidence sets, and the full test inversion grid.

    Examples
    --------
    >>> result = sp.conformal_synth(df, outcome='gdp', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989)
    >>> print(result.summary())
    """
    # --- Build panel ---
    pivot = data.pivot_table(index=time, columns=unit, values=outcome)
    times = pivot.index.values
    pre_mask = times < treatment_time
    post_mask = times >= treatment_time

    if pre_mask.sum() < 2:
        from statspai.exceptions import DataInsufficient
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
        from statspai.exceptions import DataInsufficient
        raise DataInsufficient(
            "Need at least 1 post-treatment period",
            recovery_hint=(
                "Verify the treatment_time is before the panel's end."
            ),
            diagnostics={"n_post_periods": int(post_mask.sum())},
            alternative_functions=[],
        )

    Y_treated = pivot[treated_unit].values.astype(np.float64)
    donor_cols = [c for c in pivot.columns if c != treated_unit]
    Y_donors = pivot[donor_cols].values.astype(np.float64)

    # Drop donors with NaN
    pre_donors = Y_donors[pre_mask]
    valid = ~np.any(np.isnan(pre_donors), axis=0)
    if valid.sum() == 0:
        raise ValueError("No valid donor units")
    Y_donors = Y_donors[:, valid]
    donor_cols = [donor_cols[i] for i in range(len(donor_cols)) if valid[i]]

    T0 = int(pre_mask.sum())
    T1 = int(post_mask.sum())
    post_indices = np.where(post_mask)[0]
    pre_indices = np.where(pre_mask)[0]

    # --- Fit standard SCM for point estimate ---
    weights = _solve_weights(
        Y_treated[pre_mask], Y_donors[pre_mask], penalization
    )
    Y_synth = Y_donors @ weights
    gap = Y_treated - Y_synth
    gap_pre = gap[pre_mask]
    gap_post = gap[post_mask]
    att = float(np.mean(gap_post))

    # --- Conformal inference ---
    # For each post-treatment period t, test H0: τ_t = τ0
    # Residual under H0: u_t = Y_{1t} - τ0 - Y_synth_t
    # Compare |u_t| to pre-treatment residuals |u_s| for s < T0

    # Determine grid
    if grid_range is None:
        pre_scale = np.std(gap_pre) if np.std(gap_pre) > 0 else 1.0
        grid_lo = att - 5 * pre_scale
        grid_hi = att + 5 * pre_scale
    else:
        grid_lo, grid_hi = grid_range

    tau_grid = np.linspace(grid_lo, grid_hi, grid_size)

    # Per-period conformal p-values and confidence sets
    period_results = []

    for t_idx, t_pos in enumerate(post_indices):
        observed_post = Y_treated[t_pos]
        synth_post = Y_synth[t_pos]
        observed_gap = observed_post - synth_post  # = τ_t under H0: τ=0

        # Conformal test for point null H0: τ_t = 0
        pval_zero = _conformal_pvalue(gap_pre, observed_gap, 0.0)

        # Confidence set by test inversion
        ci_lo, ci_hi = _invert_conformal_test(
            gap_pre, observed_gap, tau_grid, alpha
        )

        period_results.append({
            "time": times[t_pos],
            "effect": float(observed_gap),
            "pvalue": pval_zero,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        })

    period_df = pd.DataFrame(period_results)

    # --- Aggregate: average effect with uniform conformal CI ---
    # Joint conformal test for average effect
    avg_pvalue = _conformal_avg_pvalue(gap_pre, gap_post, 0.0, T0, T1)

    # Aggregate CI: Bonferroni-corrected union, or direct average inversion
    avg_ci_lo, avg_ci_hi = _invert_conformal_avg(
        gap_pre, gap_post, tau_grid, alpha, T0, T1
    )

    # SE approximation from conformal CI width
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_width = avg_ci_hi - avg_ci_lo
    se_approx = ci_width / (2 * z_crit) if z_crit > 0 else ci_width / 4

    model_info = {
        "inference_method": "conformal",
        "scm_method": scm_method,
        "n_donors": len(donor_cols),
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "pre_treatment_rmse": round(float(np.sqrt(np.mean(gap_pre ** 2))), 6),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "grid_size": grid_size,
        "grid_range": (grid_lo, grid_hi),
        "period_results": period_df,
        "weights": dict(zip(donor_cols, weights)),
        "Y_synth": Y_synth,
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
    y: np.ndarray, X: np.ndarray, penalization: float = 0.0,
) -> np.ndarray:
    """Standard SCM weights: min ||y - Xw||^2 + pen*||w||^2, w>=0, sum=1."""
    from ._core import solve_simplex_weights
    return solve_simplex_weights(y, X, penalization=penalization)


def _conformal_pvalue(
    pre_residuals: np.ndarray,
    observed_gap: float,
    tau0: float,
) -> float:
    """
    Conformal p-value for H0: τ = τ0.

    p = (1 + #{s : |u_s| >= |u_t - τ0|}) / (T0 + 1)
    """
    adjusted = abs(observed_gap - tau0)
    n_extreme = np.sum(np.abs(pre_residuals) >= adjusted)
    return float((1 + n_extreme) / (len(pre_residuals) + 1))


def _conformal_avg_pvalue(
    pre_residuals: np.ndarray,
    post_gaps: np.ndarray,
    tau0: float,
    T0: int,
    T1: int,
) -> float:
    """
    Conformal p-value for the average effect.

    Uses the average of absolute residuals as test statistic.
    """
    # Test statistic: mean of adjusted post-period residuals
    stat_obs = abs(np.mean(post_gaps) - tau0)

    # Compare to leave-one-out statistics from pre-period
    n_extreme = 0
    for s in range(T0):
        # If this pre-period obs were "post-treatment"
        remaining = np.delete(pre_residuals, s)
        stat_s = abs(pre_residuals[s])
        if stat_s >= stat_obs:
            n_extreme += 1

    return float((1 + n_extreme) / (T0 + 1))


def _invert_conformal_test(
    pre_residuals: np.ndarray,
    observed_gap: float,
    tau_grid: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """Invert conformal test to get CI for a single post-period."""
    accepted = []
    for tau0 in tau_grid:
        pval = _conformal_pvalue(pre_residuals, observed_gap, tau0)
        if pval > alpha:
            accepted.append(tau0)

    if len(accepted) == 0:
        return (float(observed_gap), float(observed_gap))

    return (float(min(accepted)), float(max(accepted)))


def _invert_conformal_avg(
    pre_residuals: np.ndarray,
    post_gaps: np.ndarray,
    tau_grid: np.ndarray,
    alpha: float,
    T0: int,
    T1: int,
) -> Tuple[float, float]:
    """Invert conformal test for the average effect."""
    accepted = []
    for tau0 in tau_grid:
        pval = _conformal_avg_pvalue(pre_residuals, post_gaps, tau0, T0, T1)
        if pval > alpha:
            accepted.append(tau0)

    if len(accepted) == 0:
        avg_gap = float(np.mean(post_gaps))
        return (avg_gap, avg_gap)

    return (float(min(accepted)), float(max(accepted)))


# Citation
CausalResult._CITATIONS["conformal_synth"] = (
    "@article{chernozhukov2021exact,\n"
    "  title={An Exact and Robust Conformal Inference Method for\n"
    "  Counterfactual and Synthetic Controls},\n"
    "  author={Chernozhukov, Victor and W{\\\"u}thrich, Kaspar "
    "and Zhu, Yinchu},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1849--1864},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
