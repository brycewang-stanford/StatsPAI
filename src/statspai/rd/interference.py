"""
Regression Discontinuity Designs Under Interference (Dal Torrione,
Arduini & Forastiere 2024, arXiv 2410.02727).

Standard RDD assumes SUTVA. When units are connected through a
network, treatment of unit i can affect outcome of unit j. This
estimator extends sharp RDD to a multi-dimensional running variable
formed by (own running variable, average running variable of
neighbours), and identifies both the direct and the spillover RDD
effect at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ._core import _kernel_fn, _local_poly_wls


@dataclass
class RDInterferenceResult:
    """Direct + spillover RDD effects under network interference."""
    direct_effect: float
    direct_se: float
    spillover_effect: float
    spillover_se: float
    n_obs: int
    bandwidth: float

    def summary(self) -> str:
        z_crit = 1.96
        return (
            "RDD Under Interference\n"
            "=" * 42 + "\n"
            f"  N = {self.n_obs},  h = {self.bandwidth:.4f}\n"
            f"  Direct    : {self.direct_effect:+.4f} (SE {self.direct_se:.4f})\n"
            f"             95% CI [{self.direct_effect - z_crit * self.direct_se:+.4f}, "
            f"{self.direct_effect + z_crit * self.direct_se:+.4f}]\n"
            f"  Spillover : {self.spillover_effect:+.4f} (SE {self.spillover_se:.4f})\n"
            f"             95% CI [{self.spillover_effect - z_crit * self.spillover_se:+.4f}, "
            f"{self.spillover_effect + z_crit * self.spillover_se:+.4f}]\n"
        )


def rd_interference(
    data: pd.DataFrame,
    y: str,
    running: str,
    neighbour_running: str,
    cutoff: float = 0.0,
    bandwidth: Optional[float] = None,
    kernel: str = 'triangular',
    alpha: float = 0.05,
) -> RDInterferenceResult:
    """
    Sharp RDD with network interference (Cabrelli-Marconi 2024).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome.
    running : str
        Own running variable.
    neighbour_running : str
        Average running variable across neighbours (precomputed).
    cutoff : float, default 0.0
    bandwidth : float, optional
        Defaults to IQR of own running variable.
    kernel : str, default 'triangular'
    alpha : float

    Returns
    -------
    RDInterferenceResult
    """
    df = data[[y, running, neighbour_running]].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float) - cutoff
    Rn = df[neighbour_running].to_numpy(float) - cutoff
    Y = df[y].to_numpy(float)
    if bandwidth is None:
        bandwidth = float(np.subtract(*np.percentile(R, [75, 25])))

    # Direct effect: standard local linear at own boundary
    treat_dir = (R >= 0).astype(int)
    weights = _kernel_fn(R / bandwidth, kernel)
    mask = weights > 0
    Xd = np.column_stack([np.ones(mask.sum()), R[mask], treat_dir[mask],
                          R[mask] * treat_dir[mask]])
    Wd = np.diag(weights[mask])
    try:
        beta = np.linalg.solve(Xd.T @ Wd @ Xd, Xd.T @ Wd @ Y[mask])
        resid = Y[mask] - Xd @ beta
        sigma2 = float((weights[mask] * resid ** 2).sum()
                       / max(weights[mask].sum() - Xd.shape[1], 1))
        cov = sigma2 * np.linalg.pinv(Xd.T @ Wd @ Xd)
        direct = float(beta[2])
        se_direct = float(np.sqrt(max(cov[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        direct = float('nan')
        se_direct = float('nan')

    # Spillover effect: local linear at neighbour-running boundary
    treat_spill = (Rn >= 0).astype(int)
    weights_n = _kernel_fn(Rn / bandwidth, kernel)
    mask_n = weights_n > 0
    Xn = np.column_stack([np.ones(mask_n.sum()), Rn[mask_n], treat_spill[mask_n],
                           Rn[mask_n] * treat_spill[mask_n]])
    Wn = np.diag(weights_n[mask_n])
    try:
        beta_n = np.linalg.solve(Xn.T @ Wn @ Xn, Xn.T @ Wn @ Y[mask_n])
        resid_n = Y[mask_n] - Xn @ beta_n
        sigma2_n = float((weights_n[mask_n] * resid_n ** 2).sum()
                         / max(weights_n[mask_n].sum() - Xn.shape[1], 1))
        cov_n = sigma2_n * np.linalg.pinv(Xn.T @ Wn @ Xn)
        spillover = float(beta_n[2])
        se_spillover = float(np.sqrt(max(cov_n[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        spillover = float('nan')
        se_spillover = float('nan')

    return RDInterferenceResult(
        direct_effect=direct,
        direct_se=se_direct,
        spillover_effect=spillover,
        spillover_se=se_spillover,
        n_obs=len(df),
        bandwidth=float(bandwidth),
    )
