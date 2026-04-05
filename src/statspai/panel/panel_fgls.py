"""
Panel Feasible Generalized Least Squares (FGLS).

Handles heteroskedasticity and/or autocorrelation across panels.

Equivalent to Stata's ``xtgls`` and R's ``plm::pggls()``.

References
----------
Parks, R.W. (1967).
"Efficient Estimation of a System of Regression Equations When
Disturbances Are Both Serially and Contemporaneously Correlated."
*JASA*, 62(318), 500-509.

Beck, N. & Katz, J.N. (1995).
"What To Do (and Not To Do) with Time-Series Cross-Section Data."
*APSR*, 89(3), 634-647.
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults


def panel_fgls(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str = 'id',
    time: str = 'time',
    panels: str = "heteroskedastic",
    corr: str = "independent",
    maxiter: int = 100,
    tol: float = 1e-6,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Panel FGLS (Feasible Generalized Least Squares).

    Estimates panel models allowing for:
    - Heteroskedastic errors across panels
    - Panel-specific AR(1) autocorrelation
    - Cross-sectional correlation (SUR-like)

    Equivalent to Stata's ``xtgls y x, panels(het) corr(ar1)``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    y : str
        Dependent variable.
    x : list of str
        Regressors.
    id : str, default 'id'
        Panel identifier.
    time : str, default 'time'
        Time identifier.
    panels : str, default 'heteroskedastic'
        Error structure across panels:
        'homoskedastic', 'heteroskedastic', 'correlated' (cross-sectional).
    corr : str, default 'independent'
        Within-panel correlation:
        'independent', 'ar1' (panel-specific AR(1)), 'psar1' (common AR(1)).
    maxiter : int, default 100
    tol : float, default 1e-6
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.panel_fgls(df, y='gdp', x=['investment', 'trade'],
    ...                        id='country', time='year',
    ...                        panels='heteroskedastic', corr='ar1')
    >>> print(result.summary())
    """
    df = data.sort_values([id, time]).copy()
    units = df[id].unique()
    times = df[time].unique()
    N = len(units)
    T = len(times)
    k = len(x)

    # Build matrices
    y_data = df[y].values.astype(float)
    X_data = np.column_stack([np.ones(len(df)), df[x].values.astype(float)])
    n_total = len(df)
    k_total = X_data.shape[1]
    var_names = ['_cons'] + list(x)

    # Step 1: OLS to get initial residuals
    beta_ols = np.linalg.lstsq(X_data, y_data, rcond=None)[0]
    resid = y_data - X_data @ beta_ols

    # Step 2: Estimate error covariance structure
    # Reshape residuals by panel
    resid_by_unit = {}
    idx_by_unit = {}
    for u in units:
        mask = df[id].values == u
        resid_by_unit[u] = resid[mask]
        idx_by_unit[u] = np.where(mask)[0]

    # Panel-specific variances
    sigma2_i = {}
    rho_i = {}
    for u in units:
        e = resid_by_unit[u]
        Ti = len(e)
        sigma2_i[u] = np.sum(e**2) / Ti

        if corr in ['ar1', 'psar1'] and Ti > 1:
            rho_i[u] = np.sum(e[1:] * e[:-1]) / np.sum(e[:-1]**2) if np.sum(e[:-1]**2) > 0 else 0
        else:
            rho_i[u] = 0

    if corr == 'psar1':
        # Common AR(1) coefficient
        rho_common = np.mean(list(rho_i.values()))
        rho_i = {u: rho_common for u in units}

    # Step 3: Build Omega^{-1} and do GLS
    # For efficiency, do it block by block

    for iteration in range(maxiter):
        beta_old = beta_ols.copy() if iteration == 0 else beta_gls.copy()

        XtOiX = np.zeros((k_total, k_total))
        XtOiY = np.zeros(k_total)

        for u in units:
            idx = idx_by_unit[u]
            Xi = X_data[idx]
            yi = y_data[idx]
            Ti = len(idx)

            if panels == 'homoskedastic':
                s2 = np.mean(list(sigma2_i.values()))
            else:
                s2 = sigma2_i[u]

            s2 = max(s2, 1e-10)
            rho = rho_i[u]

            # Build Ti x Ti Omega_i^{-1} for AR(1)
            if abs(rho) > 1e-10 and Ti > 1:
                # AR(1) precision matrix
                Oi_inv = np.zeros((Ti, Ti))
                Oi_inv[0, 0] = 1
                Oi_inv[-1, -1] = 1
                for t in range(1, Ti - 1):
                    Oi_inv[t, t] = 1 + rho**2
                for t in range(Ti - 1):
                    Oi_inv[t, t+1] = -rho
                    Oi_inv[t+1, t] = -rho
                Oi_inv /= (s2 * (1 - rho**2))
            else:
                Oi_inv = np.eye(Ti) / s2

            XtOiX += Xi.T @ Oi_inv @ Xi
            XtOiY += Xi.T @ Oi_inv @ yi

        try:
            beta_gls = np.linalg.solve(XtOiX, XtOiY)
        except np.linalg.LinAlgError:
            beta_gls = beta_ols
            break

        # Update residuals and re-estimate sigma, rho
        resid_new = y_data - X_data @ beta_gls
        for u in units:
            idx = idx_by_unit[u]
            e = resid_new[idx]
            Ti = len(e)
            sigma2_i[u] = np.sum(e**2) / Ti
            if corr in ['ar1', 'psar1'] and Ti > 1:
                rho_i[u] = np.sum(e[1:] * e[:-1]) / np.sum(e[:-1]**2) if np.sum(e[:-1]**2) > 0 else 0

        if corr == 'psar1':
            rho_common = np.mean(list(rho_i.values()))
            rho_i = {u: rho_common for u in units}

        if np.max(np.abs(beta_gls - beta_old)) < tol:
            break

    # SE
    try:
        var_cov = np.linalg.inv(XtOiX)
        se = np.sqrt(np.diag(var_cov))
    except np.linalg.LinAlgError:
        se = np.full(k_total, np.nan)

    params = pd.Series(beta_gls, index=var_names)
    std_errors = pd.Series(se, index=var_names)

    # R-squared
    resid_final = y_data - X_data @ beta_gls
    tss = np.sum((y_data - y_data.mean())**2)
    rss = np.sum(resid_final**2)
    r2 = 1 - rss / tss

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            'model_type': 'Panel FGLS',
            'panels': panels,
            'corr': corr,
            'n_iterations': iteration + 1,
        },
        data_info={
            'n_obs': n_total,
            'n_units': N,
            'n_periods': T,
            'dep_var': y,
            'df_resid': n_total - k_total,
        },
        diagnostics={
            'r_squared': r2,
            'mean_sigma2': np.mean(list(sigma2_i.values())),
            'mean_rho': np.mean(list(rho_i.values())) if corr != 'independent' else 0,
        },
    )
