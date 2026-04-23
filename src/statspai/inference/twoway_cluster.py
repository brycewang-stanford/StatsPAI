"""
Two-Way Cluster-Robust Standard Errors (Cameron, Gelbach & Miller 2011).

Many panel datasets exhibit correlation along two dimensions (e.g., firm and
year, state and industry). Standard one-way cluster-robust SEs handle only
one dimension. This module implements the multi-way clustering correction:

    V_twoway = V_cluster1 + V_cluster2 - V_intersection

where V_intersection clusters on the (dim1 x dim2) interaction.

Each component uses the standard Liang-Zeger (1986) sandwich estimator with
the finite-sample correction G/(G-1) * (n-1)/(n-k).

If the resulting matrix is not positive semi-definite, an eigenvalue
correction sets negative eigenvalues to zero (Cameron et al. 2011, p. 241).

References
----------
Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2011).
"Robust Inference with Multiway Clustering."
*Journal of Business & Economic Statistics*, 29(2), 238-249. [@cameron2011robust]

Liang, K.-Y. and Zeger, S.L. (1986).
"Longitudinal data analysis using generalized linear models."
*Biometrika*, 73(1), 13-22. [@liang1986longitudinal]
"""

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults


def _cluster_robust_variance(X: np.ndarray, residuals: np.ndarray,
                             clusters: np.ndarray) -> np.ndarray:
    """
    Compute one-way cluster-robust variance matrix (Liang-Zeger 1986).

    V = (G/(G-1)) * ((n-1)/(n-k)) * (X'X)^{-1} B (X'X)^{-1}

    where B = sum_g (u_g u_g') and u_g = sum_{i in g} X_i * e_i.

    Parameters
    ----------
    X : np.ndarray, shape (n, k)
        Design matrix.
    residuals : np.ndarray, shape (n,)
        OLS residuals.
    clusters : np.ndarray, shape (n,)
        Cluster labels.

    Returns
    -------
    np.ndarray, shape (k, k)
        Cluster-robust variance-covariance matrix.
    """
    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    XtX_inv = np.linalg.inv(X.T @ X)

    # Build the meat: B = sum_g u_g u_g'
    B = np.zeros((k, k))
    for g in unique_clusters:
        mask = clusters == g
        # u_g = X_g' e_g  (k-vector)
        u_g = X[mask].T @ residuals[mask]
        B += np.outer(u_g, u_g)

    # Finite-sample correction
    scale = (G / (G - 1)) * ((n - 1) / (n - k))

    V = scale * XtX_inv @ B @ XtX_inv
    return V


def _ensure_psd(V: np.ndarray) -> np.ndarray:
    """
    Eigenvalue correction for non-PSD matrices.

    Sets negative eigenvalues to zero and reconstructs the matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(V)
    if np.any(eigvals < 0):
        eigvals = np.maximum(eigvals, 0.0)
        V = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return V


def twoway_cluster(
    result: EconometricResults,
    data: pd.DataFrame,
    cluster1: str,
    cluster2: str,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Compute two-way cluster-robust standard errors.

    Implements Cameron, Gelbach & Miller (2011):

        V_twoway = V_cluster1 + V_cluster2 - V_intersection

    Parameters
    ----------
    result : EconometricResults
        Fitted OLS result. Must have ``data_info`` containing
        ``'X'`` (design matrix), ``'y'`` (response), and ``'residuals'``.
    data : pd.DataFrame
        Original data containing the cluster variables.
    cluster1 : str
        Column name for the first clustering dimension.
    cluster2 : str
        Column name for the second clustering dimension.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        New results object with the same point estimates but two-way
        clustered standard errors, t-statistics, p-values, and
        confidence intervals.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.regress("y ~ x1 + x2", data=df)
    >>> tw = sp.twoway_cluster(result, data=df, cluster1="firm", cluster2="year")
    >>> print(tw.summary())
    """
    # --- Extract estimation objects ---
    X = np.asarray(result.data_info['X'])
    residuals = np.asarray(result.data_info['residuals'])

    c1 = data[cluster1].values
    c2 = data[cluster2].values
    # Intersection cluster: unique (dim1, dim2) pairs
    c_inter = np.array([f"{a}_{b}" for a, b in zip(c1, c2)])

    # --- Three variance components ---
    V1 = _cluster_robust_variance(X, residuals, c1)
    V2 = _cluster_robust_variance(X, residuals, c2)
    V_inter = _cluster_robust_variance(X, residuals, c_inter)

    V_twoway = V1 + V2 - V_inter

    # Ensure positive semi-definiteness
    V_twoway = _ensure_psd(V_twoway)

    # --- Build new results ---
    se = pd.Series(np.sqrt(np.diag(V_twoway)), index=result.params.index)

    n = X.shape[0]
    G1 = len(np.unique(c1))
    G2 = len(np.unique(c2))
    df_resid = min(G1, G2) - 1  # Conservative DoF

    model_info = dict(result.model_info)
    model_info['se_type'] = 'twoway_cluster'
    model_info['cluster1'] = cluster1
    model_info['cluster2'] = cluster2
    model_info['n_clusters1'] = G1
    model_info['n_clusters2'] = G2

    data_info = dict(result.data_info)
    data_info['df_resid'] = df_resid
    data_info['vcov'] = V_twoway

    new_result = EconometricResults(
        params=result.params.copy(),
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=dict(result.diagnostics),
    )

    # Recompute CIs at requested alpha
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    new_result.conf_int_lower = new_result.params - t_crit * se
    new_result.conf_int_upper = new_result.params + t_crit * se

    return new_result
