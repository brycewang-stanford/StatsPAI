"""
Unified Kink + RDD + Bunching framework (Lu-Wang-Xie 2025,
arXiv 2404.09117).

Three "discontinuity-design" tools — RDD, RKD, Bunching — share a
common reduced-form estimand: the local treatment effect at the
boundary, normalised by the change in incentive. This module computes
all three on the same data and reports a unified summary so the user
can pick whichever has the strongest first-stage / cleanest design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class KinkUnifiedResult:
    """Joint RDD + RKD + Bunching estimate at a common cutoff."""
    rdd_effect: float
    rdd_se: float
    rkd_effect: float
    rkd_se: float
    bunching_elasticity: float
    bunching_se: float
    bandwidth: float
    n_obs: int

    def summary(self) -> str:
        z_crit = 1.96
        return (
            "Kink-Bunching-RDD Unified Estimator\n"
            "=" * 42 + "\n"
            f"  N            : {self.n_obs}, h = {self.bandwidth:.4f}\n"
            f"  RDD level    : {self.rdd_effect:+.4f} (SE {self.rdd_se:.4f})\n"
            f"  RKD slope    : {self.rkd_effect:+.4f} (SE {self.rkd_se:.4f})\n"
            f"  Bunching ε   : {self.bunching_elasticity:+.4f} "
            f"(SE {self.bunching_se:.4f})\n"
        )


def kink_unified(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    bin_width: Optional[float] = None,
    polynomial_order: int = 2,
    alpha: float = 0.05,
) -> KinkUnifiedResult:
    """
    Run RDD + RKD + Bunching on the same data.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable (used by RDD/RKD).
    running : str
        Running variable (also forms density for Bunching).
    cutoff : float
    bandwidth : float, optional
    bin_width : float, optional
    polynomial_order : int, default 2
    alpha : float

    Returns
    -------
    KinkUnifiedResult
    """
    from ..rd._core import _kernel_fn

    df = data[[y, running]].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float) - cutoff
    Y = df[y].to_numpy(float)
    n = len(df)
    if bandwidth is None:
        bandwidth = float(np.subtract(*np.percentile(R, [75, 25])))
    if bin_width is None:
        bin_width = bandwidth / 20.0

    treat = (R >= 0).astype(float)
    w = _kernel_fn(R / bandwidth, 'triangular')
    mask = w > 0

    # ----- RDD + RKD via local linear with treat × {1, R} ----- #
    Xb = np.column_stack([
        np.ones(mask.sum()), R[mask], treat[mask], R[mask] * treat[mask],
    ])
    Wd = np.diag(w[mask])
    try:
        beta = np.linalg.solve(Xb.T @ Wd @ Xb, Xb.T @ Wd @ Y[mask])
        resid = Y[mask] - Xb @ beta
        sigma2 = float((w[mask] * resid ** 2).sum()
                       / max(w[mask].sum() - Xb.shape[1], 1))
        cov = sigma2 * np.linalg.pinv(Xb.T @ Wd @ Xb)
        rdd_e, rdd_se = float(beta[2]), float(np.sqrt(max(cov[2, 2], 0.0)))
        rkd_e, rkd_se = float(beta[3]), float(np.sqrt(max(cov[3, 3], 0.0)))
    except np.linalg.LinAlgError:
        rdd_e = rdd_se = rkd_e = rkd_se = float('nan')

    # ----- Bunching via mass-excess + density ----- #
    bins = np.arange(-bandwidth, bandwidth + bin_width, bin_width)
    counts, edges = np.histogram(R, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    excluded = (centers > -bin_width) & (centers < bin_width)
    fit_mask = ~excluded
    if fit_mask.sum() > polynomial_order:
        coef = np.polyfit(centers[fit_mask], counts[fit_mask], polynomial_order)
        cf = np.polyval(coef, centers)
        excess = float(np.sum(counts[excluded] - cf[excluded]))
        f_at = float(np.mean(counts[fit_mask]) / max(n * bin_width, 1e-9))
        if f_at > 0 and bandwidth > 0:
            bunch_e = float(excess / (n * f_at * bandwidth ** 2))
            bunch_var = float(np.sum(counts[excluded]))
            bunch_se = float(np.sqrt(max(bunch_var, 0.0))
                              / (n * f_at * bandwidth ** 2))
        else:
            bunch_e = bunch_se = float('nan')
    else:
        bunch_e = bunch_se = float('nan')

    return KinkUnifiedResult(
        rdd_effect=rdd_e, rdd_se=rdd_se,
        rkd_effect=rkd_e, rkd_se=rkd_se,
        bunching_elasticity=bunch_e, bunching_se=bunch_se,
        bandwidth=float(bandwidth),
        n_obs=n,
    )
