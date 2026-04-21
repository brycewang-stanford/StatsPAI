"""
Distribution-Valued RDD (arXiv 2504.03992, 2025).

Estimates the RDD effect on the entire conditional distribution of Y
at the cutoff, returning the effect on each quantile of Y rather
than the mean. Equivalent to running the standard local-linear
estimator with the indicator 1{Y ≤ y} as the dependent variable for
a grid of y values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ._core import _kernel_fn


@dataclass
class DistRDResult:
    """RDD effect on each quantile."""
    quantiles: np.ndarray
    qte: np.ndarray
    se: np.ndarray
    bandwidth: float
    n_obs: int

    def summary(self) -> str:
        rows = ["Distribution-Valued RDD", "=" * 42, "  Quantile  RDD effect  SE"]
        for q, e, s in zip(self.quantiles, self.qte, self.se):
            rows.append(f"  {q:.2f}     {e:+.4f}      {s:.4f}")
        return "\n".join(rows)


def rd_distribution(
    data: pd.DataFrame,
    y: str,
    running: str,
    cutoff: float = 0.0,
    quantiles: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
    kernel: str = 'triangular',
    alpha: float = 0.05,
) -> DistRDResult:
    """
    Distribution-valued sharp RDD.

    Parameters
    ----------
    data : pd.DataFrame
    y, running : str
    cutoff : float
    quantiles : array-like, optional
        Defaults to (0.1, 0.25, 0.5, 0.75, 0.9).
    bandwidth : float, optional
    kernel : str
    alpha : float

    Returns
    -------
    DistRDResult
    """
    if quantiles is None:
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    df = data[[y, running]].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float) - cutoff
    Y = df[y].to_numpy(float)
    n = len(df)
    if bandwidth is None:
        bandwidth = float(np.subtract(*np.percentile(R, [75, 25])))

    treat = (R >= 0).astype(int)
    weights = _kernel_fn(R / bandwidth, kernel)
    mask = weights > 0

    qte = np.zeros(len(quantiles))
    se = np.zeros(len(quantiles))
    for j, q in enumerate(quantiles):
        y_q = float(np.quantile(Y, q))
        ind = (Y <= y_q).astype(float)
        try:
            Xb = np.column_stack([
                np.ones(mask.sum()), R[mask], treat[mask],
                R[mask] * treat[mask],
            ])
            Wd = np.diag(weights[mask])
            beta = np.linalg.solve(Xb.T @ Wd @ Xb, Xb.T @ Wd @ ind[mask])
            resid = ind[mask] - Xb @ beta
            sigma2 = float((weights[mask] * resid ** 2).sum()
                           / max(weights[mask].sum() - Xb.shape[1], 1))
            cov = sigma2 * np.linalg.pinv(Xb.T @ Wd @ Xb)
            qte[j] = float(beta[2])
            se[j] = float(np.sqrt(max(cov[2, 2], 0.0)))
        except np.linalg.LinAlgError:
            qte[j] = np.nan
            se[j] = np.nan

    return DistRDResult(
        quantiles=quantiles,
        qte=qte,
        se=se,
        bandwidth=float(bandwidth),
        n_obs=n,
    )
