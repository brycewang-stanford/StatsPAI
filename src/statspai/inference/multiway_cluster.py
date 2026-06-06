"""
General N-way cluster-robust variance estimator (Cameron-Gelbach-Miller 2011).

Extends the two-way clustering idea to N dimensions using the inclusion-
exclusion formula:

    V_{1..M} = Σ_{S ⊆ {1,..,M}, S≠∅} (-1)^{|S|+1} V_{∩_{m∈S} G_m}

where ``V_{·}`` is the standard Liang-Zeger one-way cluster sandwich using
the intersection grouping. The resulting matrix is projected onto the
PSD cone by zeroing negative eigenvalues.

Also exposes:

- :func:`multiway_cluster_vcov`: raw V computation from (X, resid, clusters).
- :func:`cluster_robust_se`: thin wrapper returning SE only.
- :func:`cr3_jackknife_vcov`: CR3 cluster-jackknife variance (delete-one-
  cluster) -- more conservative than CR1 and often preferred for
  few-cluster designs (Bell-McCaffrey 2002, Niccodemi et al. 2020).

References
----------
Cameron, A.C., Gelbach, J.B., Miller, D.L. (2011). "Robust Inference with
Multiway Clustering." JBES, 29(2), 238-249. [@cameron2011robust]

Bell, R.M., McCaffrey, D.F. (2002). "Bias reduction in standard errors
for linear regression with multi-stage samples." Survey Methodology.

MacKinnon, J.G., Nielsen, M.Ø., Webb, M.D. (2022). "Cluster-robust
inference: A guide to empirical practice." Journal of Econometrics. [@mackinnon2023cluster]
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ======================================================================
# Internals
# ======================================================================


def _factorize(arr) -> Tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(arr, sort=False, use_na_sentinel=True)
    if (codes < 0).any():
        raise ValueError("NaN in cluster variable.")
    return codes.astype(np.int64), len(uniq)


def _intersection_codes(cluster_list: List[np.ndarray], idx_subset: Sequence[int]) -> np.ndarray:
    """Build integer codes for the intersection cluster of the selected
    dimensions: one code per unique combination of the dimensions' levels.

    Each dimension is factorized to dense integer codes, then the per-row
    code tuples are relabelled via ``np.unique(..., axis=0)``. This is
    collision-free for any cluster dtype.

    Correctness fix (v1.17.0): the previous implementation joined the
    dimensions into a single string with a ``"\\0"`` separator, but NumPy
    fixed-width unicode strips the embedded NUL, so e.g. ``(1, 23)`` and
    ``(12, 3)`` both collapsed to ``"123"``. That undercounted the
    intersection clusters and biased the multiway-cluster VCOV (it no longer
    matched the canonical ``sandwich::vcovCL`` / ``sp.twoway_cluster``
    two-way result). See MIGRATION.md and tests/r_parity module 56.
    """
    if len(idx_subset) == 1:
        codes, _ = _factorize(cluster_list[idx_subset[0]])
        return codes
    cols = [_factorize(cluster_list[i])[0] for i in idx_subset]
    stacked = np.column_stack(cols)
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    return np.asarray(inv, dtype=np.int64).ravel()


def _one_way_sandwich(
    X: np.ndarray,
    resid: np.ndarray,
    codes: np.ndarray,
    XtX_inv: np.ndarray,
    df_adjust: bool = True,
    n_params: Optional[int] = None,
) -> np.ndarray:
    """Liang-Zeger sandwich with small-sample correction (CR1).

    Delegates to the canonical ``core._vcov.sandwich_vcov`` (CLAUDE.md §4):
    the CR1 correction is its ``'cr1'`` factor (G/max(G-1,1) * (n-1)/max(n-p,1)).
    Verified byte-identical to the prior hand-rolled sandwich for G >= 2.
    """
    from ..core._vcov import sandwich_vcov
    p = n_params if n_params is not None else X.shape[1]
    scores = X * resid[:, None]
    return sandwich_vcov(
        XtX_inv, scores, clusters=codes,
        correction="cr1" if df_adjust else "none", n_params=p,
    )


def _project_psd(V: np.ndarray) -> np.ndarray:
    """Zero negative eigenvalues of a symmetric matrix."""
    V_sym = 0.5 * (V + V.T)
    eigvals, eigvecs = np.linalg.eigh(V_sym)
    if (eigvals < 0).any():
        eigvals = np.maximum(eigvals, 0.0)
        V_sym = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return V_sym


# ======================================================================
# Public API
# ======================================================================


def multiway_cluster_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    clusters: Union[np.ndarray, List[np.ndarray]],
    df_adjust: bool = True,
    n_params: Optional[int] = None,
    psd_correct: bool = True,
) -> np.ndarray:
    """Compute N-way cluster-robust variance of an OLS coefficient vector.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix used in the regression.
    resid : ndarray, shape (n,)
        OLS residuals.
    clusters : ndarray or list of ndarrays
        One or more cluster variables, one per dimension. Non-numeric
        labels are supported.
    df_adjust : bool, default True
        If True, apply the G/(G-1) * (n-1)/(n-k) CR1 finite-sample
        correction per component variance. If False, uses raw sandwich
        (useful when the caller has already degreed-freedom adjusted).
    n_params : int, optional
        Override for the ``k`` used in DOF adjustment; useful when FEs
        have been absorbed (pass total absorbed DOF here).
    psd_correct : bool, default True
        Project V onto PSD cone by zeroing negative eigenvalues.

    Returns
    -------
    ndarray, shape (k, k)
        Multiway-cluster-robust variance-covariance matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    resid = np.asarray(resid, dtype=np.float64).ravel()
    if not isinstance(clusters, list):
        clusters = [clusters]
    cluster_list = [np.asarray(c) for c in clusters]
    n, k = X.shape
    if n != resid.size:
        raise ValueError("X and resid length mismatch.")
    if any(c.size != n for c in cluster_list):
        raise ValueError("Cluster length mismatch.")

    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)

    M = len(cluster_list)
    V = np.zeros((k, k))
    for r in range(1, M + 1):
        for combo in combinations(range(M), r):
            codes = _intersection_codes(cluster_list, combo)
            sign = (-1) ** (r + 1)
            V += sign * _one_way_sandwich(
                X, resid, codes, XtX_inv,
                df_adjust=df_adjust, n_params=n_params,
            )

    if psd_correct:
        V = _project_psd(V)
    return V


def cluster_robust_se(
    X: np.ndarray,
    resid: np.ndarray,
    clusters: Union[np.ndarray, List[np.ndarray]],
    **kwargs,
) -> np.ndarray:
    """Return cluster-robust standard errors (diagonal sqrt of vcov)."""
    V = multiway_cluster_vcov(X, resid, clusters, **kwargs)
    return np.sqrt(np.maximum(np.diag(V), 0.0))


# ======================================================================
# CR3 cluster jackknife
# ======================================================================


def cr3_jackknife_vcov(
    X: np.ndarray,
    y: np.ndarray,
    cluster: np.ndarray,
) -> np.ndarray:
    """CR3 cluster-jackknife variance (delete-one-cluster).

    Drops each cluster in turn, refits OLS on the reduced sample, and
    forms the empirical variance of the leave-one-out coefficient vector.
    Scales by (G-1)/G. Often more conservative than CR1/CR2 and useful
    when clusters are unbalanced (Bell-McCaffrey 2002).

    Complexity is O(G · k²) refits; suitable when G <= a few hundred.

    Parameters
    ----------
    X : ndarray, shape (n, k)
    y : ndarray, shape (n,)
    cluster : ndarray, shape (n,)

    Returns
    -------
    ndarray, shape (k, k)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    codes, G = _factorize(cluster)
    n, k = X.shape

    # Full-sample coefs
    coef_full = np.linalg.lstsq(X, y, rcond=None)[0]

    coefs = np.empty((G, k))
    for g in range(G):
        mask = codes != g
        Xg = X[mask]
        yg = y[mask]
        coefs[g] = np.linalg.lstsq(Xg, yg, rcond=None)[0]

    diff = coefs - coef_full
    V = (G - 1) / G * (diff.T @ diff)
    return _project_psd(V)


__all__ = [
    "multiway_cluster_vcov",
    "cluster_robust_se",
    "cr3_jackknife_vcov",
]
