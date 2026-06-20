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
    """Distributional discontinuity design output.

    Attributes
    ----------
    quantiles : np.ndarray
        Quantiles of ``y`` at which the effects are evaluated.
    rdd_effect : np.ndarray
        Level jump in the conditional CDF at the cutoff (RDD).
    rkd_effect : np.ndarray
        Slope jump in the conditional CDF at the cutoff (RKD).
    bandwidth : float
        Bandwidth used in the local-linear fit.
    n_obs : int
        Number of complete observations.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 1.0 + 0.5 * x + (x >= 0) * 0.8 + rng.normal(0, 1, n)
    >>> data = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.rd_distributional_design(data, y="y", running="x")
    >>> bool(res.rdd_effect.shape == res.quantiles.shape)
    True
    """

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
    kernel: str = "triangular",
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

    Returns
    -------
    DDDResult
        Quantile-by-quantile level jump (``rdd_effect``) and slope jump
        (``rkd_effect``) in the conditional CDF of ``y`` at the cutoff.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 1.0 + 0.5 * x + (x >= 0) * 0.8 + rng.normal(0, 1, n)
    >>> data = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.rd_distributional_design(data, y="y", running="x", cutoff=0.0)
    >>> res.n_obs
    400
    >>> len(res.rdd_effect)
    5
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
        Xb = np.column_stack(
            [
                np.ones_like(R_m),
                R_m,
                treat_m,
                treat_m * R_m,
            ]
        )
        Wd = np.diag(w_m)
        try:
            beta = np.linalg.solve(Xb.T @ Wd @ Xb, Xb.T @ Wd @ ind)
            rdd[j] = float(beta[2])  # level-jump
            rkd[j] = float(beta[3])  # slope-jump
        except np.linalg.LinAlgError:  # pragma: no cover
            rdd[j] = np.nan  # pragma: no cover
            rkd[j] = np.nan  # pragma: no cover

    return DDDResult(
        quantiles=quantiles,
        rdd_effect=rdd,
        rkd_effect=rkd,
        bandwidth=float(bandwidth),
        n_obs=n,
    )
