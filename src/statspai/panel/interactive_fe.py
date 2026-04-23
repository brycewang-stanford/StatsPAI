"""
Interactive Fixed Effects estimator (Bai 2009).

Estimates panel models with unobserved interactive effects:

    Y_{it} = X_{it}'β + λ_i' f_t + ε_{it}

where λ_i are factor loadings and f_t are common factors.
This generalizes additive FE (α_i + γ_t) to allow unit-specific
responses to common shocks.

Equivalent to R's ``phtt`` package and Stata's ``ife``.

References
----------
Bai, J. (2009).
"Panel Data Models with Interactive Fixed Effects."
*Econometrica*, 77(4), 1229-1279.

Moon, H.R. & Weidner, M. (2015).
"Linear Regression for Panel with Unknown Number of Factors as
Interactive Fixed Effects." *Econometrica*, 83(4), 1543-1579. [@moon2015linear]
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults


def interactive_fe(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str = 'id',
    time: str = 'time',
    n_factors: int = 1,
    method: str = 'iterative',
    maxiter: int = 1000,
    tol: float = 1e-6,
    robust: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Interactive fixed effects estimator (Bai 2009).

    Estimates Y_{it} = X_{it}'β + λ_i' f_t + ε_{it}
    where λ_i (N×r) are unit loadings and f_t (T×r) are time factors.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data.
    y : str
        Dependent variable.
    x : list of str
        Regressors.
    id : str, default 'id'
        Unit identifier.
    time : str, default 'time'
        Time identifier.
    n_factors : int, default 1
        Number of interactive factors (r).
    method : str, default 'iterative'
        'iterative' (Bai 2009 CCE-type), 'pca' (principal components).
    maxiter : int, default 1000
    tol : float, default 1e-6
    robust : bool, default True
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.interactive_fe(df, y='gdp', x=['investment', 'trade'],
    ...                            id='country', time='year', n_factors=2)
    >>> print(result.summary())
    """
    df = data.copy()
    units = df[id].unique()
    times = df[time].unique()
    N = len(units)
    T = len(times)
    k = len(x)
    r = n_factors

    # Create unit and time indices
    unit_map = {u: i for i, u in enumerate(units)}
    time_map = {t: i for i, t in enumerate(times)}
    df['_uid'] = df[id].map(unit_map)
    df['_tid'] = df[time].map(time_map)
    df = df.sort_values(['_uid', '_tid'])

    # Reshape to panel matrices (N x T)
    Y_mat = np.full((N, T), np.nan)
    X_mats = [np.full((N, T), np.nan) for _ in range(k)]

    for _, row in df.iterrows():
        i, t = int(row['_uid']), int(row['_tid'])
        Y_mat[i, t] = row[y]
        for j, xvar in enumerate(x):
            X_mats[j][i, t] = row[xvar]

    # Handle unbalanced panel: create mask
    x_valid = np.all(np.array(
        [np.all(np.isfinite(xm), axis=1) for xm in X_mats]), axis=0)
    valid = np.all(np.isfinite(Y_mat), axis=1) & x_valid
    if not np.all(valid):
        Y_mat = Y_mat[valid]
        X_mats = [xm[valid] for xm in X_mats]
        N = Y_mat.shape[0]

    # Iterative estimation (Bai 2009 Algorithm)
    # Step 1: Initialize β with pooled OLS (ignoring factors)
    Y_vec = Y_mat.ravel()
    X_pool = np.column_stack([xm.ravel() for xm in X_mats])
    valid_obs = np.isfinite(Y_vec) & np.all(np.isfinite(X_pool), axis=1)
    beta = np.linalg.lstsq(X_pool[valid_obs], Y_vec[valid_obs], rcond=None)[0]

    for iteration in range(maxiter):
        beta_old = beta.copy()

        # Step 2: Given β, compute residual matrix E = Y - X*β
        E = Y_mat.copy()
        for j in range(k):
            E -= beta[j] * X_mats[j]

        # Step 3: Estimate factors via PCA of E
        # SVD of E: E = U S V'
        # factors f_t are first r columns of V (scaled), loadings λ_i from U
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        Lambda = U[:, :r] * S[:r]  # N x r (loadings)
        F = Vt[:r, :].T  # T x r (factors)

        # Step 4: Given factors, re-estimate β
        # Concentrate out factors: M_F = I - F(F'F)^{-1}F'
        FtF_inv = np.linalg.inv(F.T @ F)
        M_F = np.eye(T) - F @ FtF_inv @ F.T

        # Apply M_F to each row: Y_i * M_F, X_ij * M_F
        Y_proj = Y_mat @ M_F  # N x T
        X_proj = [xm @ M_F for xm in X_mats]

        # Stack and do OLS
        Y_stacked = Y_proj.ravel()
        X_stacked = np.column_stack([xp.ravel() for xp in X_proj])

        try:
            beta = np.linalg.lstsq(X_stacked, Y_stacked, rcond=None)[0]
        except np.linalg.LinAlgError:
            break

        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    # Final residuals
    resid_mat = Y_mat.copy()
    for j in range(k):
        resid_mat -= beta[j] * X_mats[j]
    resid_mat -= Lambda @ F.T

    residuals = resid_mat.ravel()
    sigma2 = np.sum(residuals**2) / (N * T - k - r * (N + T - r))

    # Standard errors (Bai 2009, Theorem 3)
    # Simplified: use sandwich estimator on projected data
    if robust:
        # Cluster by unit
        meat = np.zeros((k, k))
        for i in range(N):
            x_i = np.column_stack([xm[i] @ M_F for xm in X_mats])  # T x k
            e_i = resid_mat[i]  # T vector
            score_i = x_i.T @ e_i  # k vector
            meat += np.outer(score_i, score_i)

        bread = np.linalg.inv(X_stacked.T @ X_stacked)
        var_cov = bread @ meat @ bread
    else:
        var_cov = sigma2 * np.linalg.inv(X_stacked.T @ X_stacked)

    se = np.sqrt(np.diag(var_cov))

    params = pd.Series(beta, index=x)
    std_errors = pd.Series(se, index=x)

    # R-squared
    tss = np.sum((Y_mat - Y_mat.mean())**2)
    rss = np.sum(residuals**2)
    r2 = 1 - rss / tss

    # Eigenvalues of E'E / (NT) for factor diagnostics
    eigenvalues = (S[:min(10, len(S))]**2) / (N * T)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            'model_type': 'Interactive Fixed Effects (Bai 2009)',
            'n_factors': r,
            'method': method,
            'n_iterations': iteration + 1,
            'converged': iteration < maxiter - 1,
        },
        data_info={
            'n_obs': N * T,
            'n_units': N,
            'n_periods': T,
            'dep_var': y,
            'df_resid': N * T - k - r * (N + T - r),
        },
        diagnostics={
            'r_squared': r2,
            'sigma2': sigma2,
            'eigenvalues': eigenvalues.tolist(),
            'factors': F,
            'loadings': Lambda,
        },
    )
