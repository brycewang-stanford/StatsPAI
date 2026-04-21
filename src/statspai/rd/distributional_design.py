"""
Distributional Discontinuity Design — unifying RDD + RKD on the
distribution layer (arXiv 2602.19290, 2026).

Returns:
- Distributional RDD effect (jump in CDF at the cutoff).
- Distributional RKD effect (slope-jump in CDF at the cutoff).

Both at every quantile of Y. This unifies the four main "discontinuity
design" variants (sharp/fuzzy RDD, RKD) into a single quantile-by-
quantile interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ._core import _kernel_fn


@dataclass
class DDDResult:
    """Distributional discontinuity design output."""
    quantiles: np.ndarray
    rdd_effect: np.ndarray
    rkd_effect: np.ndarray
    bandwidth: float
    n_obs: int

    def summary(self) -> str:
        rows = [
            "Distributional Discontinuity Design",
            "=" * 42,
            f"  N = {self.n_obs}, h = {self.bandwidth:.4f}",
            "  Quantile  RDD effect  RKD effect",
        ]
        for q, e1, e2 in zip(self.quantiles, self.rdd_effect, self.rkd_effect):
            rows.append(f"  {q:.2f}     {e1:+.4f}      {e2:+.4f}")
        return "\n".join(rows)


def rd_distributional_design(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    quantiles: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
    kernel: str = 'triangular',
) -> DDDResult:
    """
    Joint RDD + RKD on the conditional distribution of Y.

    Parameters
    ----------
    data : pd.DataFrame
    y, running : str
    cutoff : float
    quantiles : array-like, optional
    bandwidth : float, optional
    kernel : str
    """
    if quantiles is None:
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    df = data[[y, running]].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float) - cutoff
    Y = df[y].to_numpy(float)
    n = len(df)
    if bandwidth is None:
        bandwidth = float(np.subtract(*np.percentile(R, [75, 25])))

    treat = (R >= 0).astype(float)
    w = _kernel_fn(R / bandwidth, kernel)
    mask = w > 0
    R_m = R[mask]
    Y_m = Y[mask]
    w_m = w[mask]
    treat_m = treat[mask]

    rdd = np.zeros(len(quantiles))
    rkd = np.zeros(len(quantiles))
    for j, q in enumerate(quantiles):
        y_q = float(np.quantile(Y, q))
        ind = (Y_m <= y_q).astype(float)
        # Local linear in R, with treat × {1, R} interaction
        Xb = np.column_stack([
            np.ones_like(R_m), R_m, treat_m, treat_m * R_m,
        ])
        Wd = np.diag(w_m)
        try:
            beta = np.linalg.solve(Xb.T @ Wd @ Xb, Xb.T @ Wd @ ind)
            rdd[j] = float(beta[2])  # level-jump
            rkd[j] = float(beta[3])  # slope-jump
        except np.linalg.LinAlgError:
            rdd[j] = np.nan
            rkd[j] = np.nan

    return DDDResult(
        quantiles=quantiles,
        rdd_effect=rdd,
        rkd_effect=rkd,
        bandwidth=float(bandwidth),
        n_obs=n,
    )
