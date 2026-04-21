"""
Bridge: Kink-Bunching ≡ RDD (Lu-Wang-Xie 2025, arXiv 2404.09117).

A kink in a deterministic schedule combined with bunching in the
running variable identifies the same elasticity / treatment effect
as a (sharp or fuzzy) RDD/RKD around the kink. Reporting both lets
the user check whether bunching evidence corroborates the RDD.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .core import BridgeResult, _agreement_test, _dr_combine, _register


@_register("kink_rdd")
def kink_rdd_bridge(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float,
    bandwidth: Optional[float] = None,
    bin_width: Optional[float] = None,
    polynomial: int = 2,
    alpha: float = 0.05,
) -> BridgeResult:
    """
    Compare RKD slope-jump (path A) against bunching-based slope
    estimate (path B) on a one-dimensional running variable.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    running : str
        Running variable / forcing variable.
    cutoff : float
        Kink point.
    bandwidth : float, optional
        Local-polynomial bandwidth on each side. Defaults to the
        IQR of the running variable.
    bin_width : float, optional
        Bin width for bunching density estimation. Defaults to
        bandwidth / 20.
    polynomial : int, default 2
        Local polynomial order for the bunching counterfactual fit.
    alpha : float
    """
    df = data[[y, running]].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float)
    Y = df[y].to_numpy(float)
    n = len(df)
    if bandwidth is None:
        bandwidth = float(np.subtract(*np.percentile(R, [75, 25])))

    # ---------- Path A: RKD via local linear slope-difference ---------- #
    def _local_slope(R, Y, side):
        # Triangular kernel local linear at boundary; returns slope coef.
        if side == 'left':
            mask = (R < cutoff) & (R >= cutoff - bandwidth)
            r0 = cutoff
        else:
            mask = (R > cutoff) & (R <= cutoff + bandwidth)
            r0 = cutoff
        Rm = R[mask] - r0
        Ym = Y[mask]
        if len(Rm) < 5:
            return np.nan, np.nan
        w = np.maximum(1.0 - np.abs(Rm) / bandwidth, 0.0)
        Xd = np.column_stack([np.ones_like(Rm), Rm])
        WX = Xd * w[:, None]
        try:
            beta = np.linalg.solve(WX.T @ Xd, WX.T @ Ym)
            resid = Ym - Xd @ beta
            sigma2 = float(np.sum(w * resid ** 2) / max(w.sum() - 2, 1))
            cov = sigma2 * np.linalg.pinv(WX.T @ Xd)
            return float(beta[1]), float(np.sqrt(max(cov[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            return np.nan, np.nan

    slope_left, se_left = _local_slope(R, Y, 'left')
    slope_right, se_right = _local_slope(R, Y, 'right')
    if np.isnan(slope_left) or np.isnan(slope_right):
        rkd_est = np.nan
        rkd_se = np.nan
    else:
        rkd_est = float(slope_right - slope_left)
        rkd_se = float(np.sqrt(se_left ** 2 + se_right ** 2))

    # ---------- Path B: bunching mass ratio ---------- #
    if bin_width is None:
        bin_width = bandwidth / 20.0
    bins = np.arange(
        cutoff - bandwidth, cutoff + bandwidth + bin_width, bin_width
    )
    counts, edges = np.histogram(R, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    excluded = (centers > cutoff - bin_width) & (centers < cutoff + bin_width)
    fit_mask = ~excluded
    # Polynomial counterfactual on fit mask
    coef = np.polyfit(centers[fit_mask], counts[fit_mask], polynomial)
    counterfactual = np.polyval(coef, centers)
    excess_mass = float(np.sum(counts[excluded] - counterfactual[excluded]))
    # Convert excess mass into a bunching-implied slope-change estimate:
    # for a linear schedule of slope τ on the right-of-cutoff segment,
    # excess mass ≈ τ × bandwidth × density(cutoff). Solve for τ.
    f_at_cutoff = float(np.mean(counts[fit_mask]) / (n * bin_width))
    if f_at_cutoff > 0 and bandwidth > 0:
        bunch_est = float(excess_mass / (n * f_at_cutoff * bandwidth ** 2))
    else:
        bunch_est = np.nan

    # SE for bunching via Poisson approximation on counts
    bunch_var = float(np.sum(counts[excluded])) + float(np.sum(
        counterfactual[excluded] * (1 - counterfactual[excluded] / n)
    ))
    if f_at_cutoff > 0 and bandwidth > 0:
        bunch_se = float(np.sqrt(max(bunch_var, 0.0))
                         / (n * f_at_cutoff * bandwidth ** 2))
    else:
        bunch_se = np.nan

    if np.isnan(rkd_est):
        rkd_est, rkd_se = bunch_est, bunch_se
    if np.isnan(bunch_est):
        bunch_est, bunch_se = rkd_est, rkd_se

    diff, diff_se, diff_p = _agreement_test(
        rkd_est, rkd_se, bunch_est, bunch_se
    )
    est_dr, se_dr = _dr_combine(
        rkd_est, rkd_se, bunch_est, bunch_se, diff_p
    )

    return BridgeResult(
        kind="kink_rdd",
        path_a_name="RKD slope-jump",
        path_b_name="Bunching mass",
        estimate_a=float(rkd_est),
        estimate_b=float(bunch_est),
        se_a=float(rkd_se),
        se_b=float(bunch_se),
        diff=diff,
        diff_se=diff_se,
        diff_p=diff_p,
        estimate_dr=est_dr,
        se_dr=se_dr,
        n_obs=n,
        detail={
            "bandwidth": bandwidth, "bin_width": bin_width,
            "excess_mass": excess_mass,
        },
        reference="Lu, Wang, Xie (2025), arXiv 2404.09117",
    )
