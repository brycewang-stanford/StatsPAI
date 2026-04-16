"""
Shared low-level primitives for the RD module.

Centralizes kernel functions, kernel-specific constants, and the
canonical weighted-least-squares local polynomial regression used
across rdrobust/rd2d. Keeping them in one place guarantees numerical
consistency and removes a class of drift bugs.

Public surface (module-internal, underscore-prefixed to stay private
to statspai.rd):

    _kernel_fn(u, kernel)           -> kernel weights K(u)
    _kernel_constants(kernel)       -> dict with C_K, mu_2, nu_0
    _kernel_mse_constant(kernel)    -> float C_K (local-linear MSE constant)
    _local_poly_wls(y, x, h, p, kernel, cluster=None, covs=None)
        -> (beta, vcov, n_eff) with HC1 or cluster-robust variance;
           optional additive covariate augmentation; returned beta/vcov
           correspond to the polynomial part only (first p+1 entries).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats as _sp_stats


_KERNEL_TABLE = {
    'triangular':   {'C_K': 3.4375, 'mu_2': 1 / 6, 'nu_0': 2 / 3},
    'epanechnikov': {'C_K': 3.0,    'mu_2': 1 / 5, 'nu_0': 3 / 5},
    'uniform':      {'C_K': 2.7,    'mu_2': 1 / 3, 'nu_0': 1 / 2},
}


def _kernel_fn(u: np.ndarray, kernel: str) -> np.ndarray:
    """
    Kernel K(u). Triangular / epanechnikov / uniform have compact support
    on |u| <= 1; gaussian is the standard normal pdf (full support).
    """
    u = np.asarray(u, dtype=float)
    if kernel == 'triangular':
        return np.maximum(1 - np.abs(u), 0)
    elif kernel == 'uniform':
        return 0.5 * (np.abs(u) <= 1).astype(float)
    elif kernel == 'epanechnikov':
        return 0.75 * np.maximum(1 - u ** 2, 0)
    elif kernel == 'gaussian':
        return _sp_stats.norm.pdf(u)
    raise ValueError(f"Unknown kernel: {kernel}")


def _kernel_constants(kernel: str) -> dict:
    """
    Return kernel-specific constants for bandwidth selection.

    C_K  : MSE-optimal bandwidth constant for local linear (p=1).
    mu_2 : second moment of the kernel, int u^2 K(u) du.
    nu_0 : int K(u)^2 du (roughness).
    """
    return _KERNEL_TABLE[kernel]


def _kernel_mse_constant(kernel: str) -> float:
    """C_{1,1}: MSE-optimal bandwidth constant for local linear."""
    return _KERNEL_TABLE.get(kernel, _KERNEL_TABLE['triangular'])['C_K']


def _sandwich_variance(
    Xw: np.ndarray,
    yw: np.ndarray,
    beta: np.ndarray,
    resid: np.ndarray,
    n_eff: int,
    k: int,
    cluster_in_bw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    HC1 or cluster-robust sandwich variance for a WLS fit.

    Parameters
    ----------
    Xw : (n, k) array
        Square-root-weighted design matrix, i.e. X * sqrt(w).
    yw : (n,) array
        Square-root-weighted response, i.e. y * sqrt(w).
    beta : (k,) array
        Estimated coefficients from the weighted LS.
    resid : (n,) array
        Raw (unweighted) residuals y - X @ beta.
    n_eff : int
        Effective sample size (observations inside the bandwidth).
    k : int
        Number of columns in the design matrix.
    cluster_in_bw : (n,) array or None
        Cluster identifiers for observations inside the bandwidth.
        When None, HC1 is used.
    """
    XtWX = Xw.T @ Xw
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)

    if cluster_in_bw is not None:
        unique_cl = np.unique(cluster_in_bw)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for c_val in unique_cl:
            idx = cluster_in_bw == c_val
            score = (Xw[idx].T @ (yw[idx] - Xw[idx] @ beta)).ravel()
            meat += np.outer(score, score)
        corr = n_cl / (n_cl - 1) if n_cl > 1 else 1.0
        return corr * bread @ meat @ bread
    else:
        corr = n_eff / (n_eff - k) if n_eff > k else 1.0
        meat = Xw.T @ np.diag(resid ** 2 * corr) @ Xw
        return bread @ meat @ bread


def _local_poly_wls(
    y: np.ndarray,
    x: np.ndarray,
    h: float,
    p: int,
    kernel: str,
    cluster: Optional[np.ndarray] = None,
    covs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    WLS local polynomial regression evaluated at x = 0.

    When covs is provided, the design matrix is augmented:
    [1, x, x^2, ..., x^p, z1, z2, ..., zk]
    The treatment effect is still beta[0] (intercept) for deriv=0
    or beta[1] (slope) for deriv=1, etc. Covariates enter additively.

    Returns (beta, vcov, n_effective).
    The returned beta and vcov correspond to the polynomial part only
    (first p+1 elements), with covariate effects absorbed.
    """
    u = x / h
    w = _kernel_fn(u, kernel)
    in_bw = np.abs(u) <= 1
    n_eff = int(in_bw.sum())

    k_poly = p + 1

    if n_eff < k_poly + 2:
        return np.zeros(k_poly), np.eye(k_poly) * 1e10, 0

    y_bw = y[in_bw]
    x_bw = x[in_bw]
    w_bw = w[in_bw]

    # Design matrix [1, x, x^2, ..., x^p]
    X_poly = np.column_stack([x_bw ** j for j in range(k_poly)])

    # Augment with covariates if provided
    if covs is not None:
        Z_bw = covs[in_bw]
        X = np.column_stack([X_poly, Z_bw])
    else:
        X = X_poly

    k_total = X.shape[1]

    # WLS via square-root weights
    sqw = np.sqrt(w_bw)
    Xw = X * sqw[:, np.newaxis]
    yw = y_bw * sqw

    try:
        XtWX = Xw.T @ Xw
        beta_full = np.linalg.solve(XtWX, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta_full = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        XtWX = Xw.T @ Xw

    resid = y_bw - X @ beta_full

    cl_in_bw = cluster[in_bw] if cluster is not None else None
    vcov_full = _sandwich_variance(
        Xw, yw, beta_full, resid, n_eff, k_total, cl_in_bw
    )

    # Return only the polynomial part (first k_poly elements)
    beta = beta_full[:k_poly]
    vcov = vcov_full[:k_poly, :k_poly]

    return beta, vcov, n_eff
