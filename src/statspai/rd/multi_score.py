"""
Multi-Score RDD (arXiv 2508.15692, 2025).

When eligibility is determined by multiple thresholds applied to
multiple running variables (e.g. SAT-math AND SAT-verbal must each
exceed a cutoff), the boundary is a multi-dimensional manifold.
This module estimates the local average treatment effect on the
boundary curve using a multivariate local linear estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ._core import _kernel_fn


@dataclass
class MultiScoreRDResult:
    """Multi-score RDD effect on the eligibility boundary.

    Attributes
    ----------
    boundary_effect : float
        Local average treatment effect on the boundary manifold.
    se : float
        Standard error of ``boundary_effect``.
    n_obs : int
        Number of observations used (after dropping missing rows).
    boundary_share : float
        Fraction of observations inside the kernel window.
    bandwidth : float
        Bandwidth applied to the (max-) distance to the boundary.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> m = rng.uniform(400, 800, n)
    >>> v = rng.uniform(400, 800, n)
    >>> treat = ((m >= 600) & (v >= 600)).astype(int)
    >>> y = 10.0 + 0.01 * m + 0.01 * v + 5.0 * treat + rng.normal(0, 2, n)
    >>> df = pd.DataFrame({"y": y, "math": m, "verbal": v})
    >>> res = sp.rd_multi_score(
    ...     df, y="y", running_vars=["math", "verbal"], cutoffs=[600.0, 600.0])
    >>> isinstance(res, sp.MultiScoreRDResult)
    True
    >>> res.n_obs
    2000
    >>> bool(res.se >= 0.0)
    True
    """

    boundary_effect: float
    se: float
    n_obs: int
    boundary_share: float
    bandwidth: float

    def summary(self) -> str:
        z_crit = 1.96
        return (
            "Multi-Score RDD\n"
            "=" * 42 + "\n"
            f"  N           : {self.n_obs}\n"
            f"  Boundary sh : {self.boundary_share:.3f}\n"
            f"  h           : {self.bandwidth:.4f}\n"
            f"  Effect      : {self.boundary_effect:+.4f} (SE {self.se:.4f})\n"
            f"  95% CI      : [{self.boundary_effect - z_crit * self.se:+.4f},"
            f" {self.boundary_effect + z_crit * self.se:+.4f}]\n"
        )


def rd_multi_score(
    data: pd.DataFrame,
    y: str,
    running_vars: List[str],
    cutoffs: List[float],
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
) -> MultiScoreRDResult:
    """
    Multi-score RDD: treatment if all running variables exceed cutoffs.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome.
    running_vars : list of str
        Multiple running variables.
    cutoffs : list of float
        One cutoff per running variable (same length).
    bandwidth : float, optional
        Defaults to median IQR across running vars.
    kernel : str
    alpha : float

    Returns
    -------
    MultiScoreRDResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> sat_math = rng.uniform(400, 800, n)
    >>> sat_verbal = rng.uniform(400, 800, n)
    >>> treat = ((sat_math >= 600) & (sat_verbal >= 600)).astype(int)
    >>> y = (10.0 + 0.01 * sat_math + 0.01 * sat_verbal
    ...      + 5.0 * treat + rng.normal(0, 2, n))
    >>> df = pd.DataFrame({"y": y, "math": sat_math, "verbal": sat_verbal})
    >>> res = sp.rd_multi_score(
    ...     df, y="y", running_vars=["math", "verbal"],
    ...     cutoffs=[600.0, 600.0])
    >>> res.n_obs
    2000
    >>> isinstance(res.boundary_effect, float)
    True
    >>> print(res.summary())  # doctest: +SKIP
    """
    if len(running_vars) != len(cutoffs):
        raise ValueError(
            f"len(running_vars)={len(running_vars)} != " f"len(cutoffs)={len(cutoffs)}"
        )
    df = data[[y] + list(running_vars)].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    R = df[list(running_vars)].to_numpy(float) - np.array(cutoffs)
    n = len(df)

    if bandwidth is None:
        bandwidth = float(
            np.median(
                [
                    np.subtract(*np.percentile(R[:, j], [75, 25]))
                    for j in range(R.shape[1])
                ]
            )
        )

    # Distance to boundary = max(distance to each cutoff)
    dist = np.max(R, axis=1)  # negative when not all crossed
    treat = (dist >= 0).astype(int)
    weights = _kernel_fn(np.abs(dist) / bandwidth, kernel)
    mask = weights > 0
    if mask.sum() < 5:
        raise ValueError(  # pragma: no cover
            f"Bandwidth {bandwidth:.4f} too small — only "
            f"{mask.sum()} obs in window."
        )

    # Local linear design: (1, R1, ..., Rk, treat, treat*R1, ..., treat*Rk).
    Xb = [np.ones(mask.sum())]
    for j in range(R.shape[1]):
        Xb.append(R[mask, j])
    Xb.append(treat[mask])
    for j in range(R.shape[1]):
        Xb.append(R[mask, j] * treat[mask])
    Xd = np.column_stack(Xb)
    Wd = np.diag(weights[mask])
    try:
        beta = np.linalg.solve(Xd.T @ Wd @ Xd, Xd.T @ Wd @ Y[mask])
        resid = Y[mask] - Xd @ beta
        sigma2 = float(
            (weights[mask] * resid**2).sum() / max(weights[mask].sum() - Xd.shape[1], 1)
        )
        cov = sigma2 * np.linalg.pinv(Xd.T @ Wd @ Xd)
        # Treatment coefficient is at index 1 + R.shape[1]
        idx_treat = 1 + R.shape[1]
        effect = float(beta[idx_treat])
        se = float(np.sqrt(max(cov[idx_treat, idx_treat], 0.0)))
    except np.linalg.LinAlgError:  # pragma: no cover
        effect = float("nan")  # pragma: no cover
        se = float("nan")  # pragma: no cover

    return MultiScoreRDResult(
        boundary_effect=effect,
        se=se,
        n_obs=n,
        boundary_share=float(mask.mean()),
        bandwidth=float(bandwidth),
    )
