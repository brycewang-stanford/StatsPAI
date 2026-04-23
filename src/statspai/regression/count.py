"""
Count data models: Poisson, Negative Binomial, and PPML (Pseudo-Poisson Maximum Likelihood)

Implements native numpy/scipy MLE estimation with robust and clustered standard errors,
following Stata-like API conventions for applied econometric work.

References
----------
- Cameron, A.C. & Trivedi, P.K. (2013). Regression Analysis of Count Data. 2nd ed.
- Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity." REStat.
- Correia, S., Guimaraes, P. & Zylkin, T. (2020). "Fast Poisson estimation with
  high-dimensional fixed effects." Stata Journal. [@cameron2013regression]
"""

from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from scipy import stats, optimize, special
import warnings

from ..core.results import EconometricResults
from ..core.utils import parse_formula, create_design_matrices, prepare_data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_formula_or_xy(
    formula, data, y, x, add_constant=True
):
    """Parse formula/data or y/x into arrays + variable names."""
    if formula is not None and data is not None:
        parsed = parse_formula(formula)
        dep_var = parsed['dependent']
        indep_vars = parsed['exogenous']
        fe_vars = parsed.get('fixed_effects', [])
        has_constant = parsed['has_constant']

        y_arr = data[dep_var].values.astype(np.float64)
        X_cols = indep_vars
        X_arr_parts = [data[v].values.astype(np.float64) for v in X_cols]
        if has_constant and add_constant:
            X_arr_parts = [np.ones(len(data))] + X_arr_parts
            var_names = ['_cons'] + X_cols
        else:
            var_names = list(X_cols)
        X_arr = np.column_stack(X_arr_parts) if X_arr_parts else np.ones((len(data), 1))
        if not X_arr_parts:
            var_names = ['_cons']
        return y_arr, X_arr, var_names, dep_var, fe_vars, data
    elif y is not None and x is not None and data is not None:
        dep_var = y
        y_arr = data[dep_var].values.astype(np.float64)
        X_cols = list(x)
        X_arr_parts = [data[v].values.astype(np.float64) for v in X_cols]
        if add_constant:
            X_arr_parts = [np.ones(len(data))] + X_arr_parts
            var_names = ['_cons'] + X_cols
        else:
            var_names = list(X_cols)
        X_arr = np.column_stack(X_arr_parts) if X_arr_parts else np.ones((len(data), 1))
        if not X_arr_parts:
            var_names = ['_cons']
        return y_arr, X_arr, var_names, dep_var, [], data
    else:
        raise ValueError("Must provide either (formula, data) or (y, x, data)")


def _safe_exp(eta, cap=700.0):
    """Exponentiate with overflow protection."""
    return np.exp(np.clip(eta, -cap, cap))


def _sandwich_vcov(X, mu, residuals, XtX_inv_bread=None):
    """Robust (HC0) sandwich variance-covariance."""
    n, k = X.shape
    W = mu  # Poisson weight
    if XtX_inv_bread is None:
        XtWX = X.T @ (X * W[:, None])
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
    else:
        XtWX_inv = XtX_inv_bread

    # Meat: sum of u_i^2 * x_i x_i'  where u_i = (y_i - mu_i)
    score_i = X * residuals[:, None]  # n x k
    meat = score_i.T @ score_i
    return XtWX_inv @ meat @ XtWX_inv


def _cluster_vcov(X, mu, residuals, cluster_arr):
    """Clustered sandwich variance-covariance."""
    n, k = X.shape
    W = mu
    XtWX = X.T @ (X * W[:, None])
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)

    clusters = np.unique(cluster_arr)
    n_clusters = len(clusters)
    meat = np.zeros((k, k))
    for c in clusters:
        idx = cluster_arr == c
        score_c = (X[idx] * residuals[idx, None]).sum(axis=0)
        meat += np.outer(score_c, score_c)

    # Finite-sample correction
    correction = n_clusters / (n_clusters - 1)
    return correction * XtWX_inv @ meat @ XtWX_inv


def _poisson_vcov(X, mu, residuals, robust, cluster_arr):
    """Compute variance-covariance for Poisson-family models."""
    n, k = X.shape
    W = mu
    XtWX = X.T @ (X * W[:, None])
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)

    if cluster_arr is not None:
        return _cluster_vcov(X, mu, residuals, cluster_arr)
    elif robust.lower() in ('robust', 'hc0', 'hc1', 'hc2', 'hc3'):
        vcov = _sandwich_vcov(X, mu, residuals, XtWX_inv)
        if robust.lower() == 'hc1':
            vcov *= n / (n - k)
        return vcov
    else:
        # Model-based (assumes Var(y) = mu)
        return XtWX_inv


def _poisson_loglik(y, mu):
    """Poisson log-likelihood (up to constant)."""
    # l = sum(y*log(mu) - mu - log(y!))
    return np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu - special.gammaln(y + 1))


# ---------------------------------------------------------------------------
# Poisson IRLS
# ---------------------------------------------------------------------------

def _poisson_irls(y, X, offset=None, weights=None, maxiter=100, tol=1e-8):
    """
    Poisson regression via Iteratively Reweighted Least Squares.

    Returns (beta, mu, converged, n_iter).
    """
    n, k = X.shape
    if offset is None:
        offset = np.zeros(n)
    if weights is None:
        weights = np.ones(n)

    # Initialize with log(y + 0.5)
    y_init = np.where(y > 0, y, 0.5)
    beta = np.linalg.lstsq(X, np.log(y_init) - offset, rcond=None)[0]

    converged = False
    for it in range(maxiter):
        eta = X @ beta + offset
        mu = _safe_exp(eta)

        # Working variable and weights
        w = weights * mu
        z = eta + (y - mu) / mu - offset  # working response without offset

        # Weighted least squares: (X'WX)^-1 X'Wz
        XtW = X.T * w[None, :]
        XtWX = XtW @ X
        XtWz = XtW @ z
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

        # Check convergence on relative parameter change
        delta = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12))
        beta = beta_new
        if delta < tol:
            converged = True
            break

    eta = X @ beta + offset
    mu = _safe_exp(eta)
    return beta, mu, converged, it + 1


# ---------------------------------------------------------------------------
# Negative Binomial MLE
# ---------------------------------------------------------------------------

def _nb2_loglik(y, mu, alpha):
    """NB2 log-likelihood: Var(y) = mu + alpha * mu^2."""
    inv_alpha = 1.0 / max(alpha, 1e-300)
    r = inv_alpha
    # l = sum( lgamma(y + r) - lgamma(r) - lgamma(y+1) + r*log(r/(r+mu)) + y*log(mu/(r+mu)) )
    ll = np.sum(
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(np.maximum(mu, 1e-300) / (r + mu))
    )
    return ll


def _nb1_loglik(y, mu, delta):
    """NB1 log-likelihood: Var(y) = mu + delta * mu  =>  Var = mu*(1+delta)."""
    # Parameterize as r = mu/delta  so Var = mu + delta*mu
    delta = max(delta, 1e-300)
    r = mu / delta
    ll = np.sum(
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(np.maximum(mu, 1e-300) / (r + mu))
    )
    return ll


def _nb2_fit(y, X, offset=None, weights=None, maxiter=100, tol=1e-8):
    """
    NB2 via iterating: Poisson IRLS for beta given alpha, then profile
    likelihood optimization for alpha.

    Returns (beta, mu, alpha, converged, n_iter).
    """
    n, k = X.shape
    if offset is None:
        offset = np.zeros(n)
    if weights is None:
        weights = np.ones(n)

    # Start with Poisson estimates
    beta, mu, _, _ = _poisson_irls(y, X, offset, weights, maxiter=50, tol=1e-6)

    # Initial alpha from moment estimator
    resid = y - mu
    pearson = np.sum(resid**2 / mu) / (n - k)
    alpha = max((pearson - 1) / (np.mean(mu)), 0.01)

    converged = False
    for outer in range(maxiter):
        # Given alpha, IRLS for beta with NB2 weights
        for inner in range(maxiter):
            eta = X @ beta + offset
            mu = _safe_exp(eta)

            inv_alpha = 1.0 / alpha
            # NB2 weight: mu / (1 + alpha*mu)
            w = weights * mu / (1 + alpha * mu)
            z = eta + (y - mu) / mu - offset

            XtW = X.T * w[None, :]
            XtWX = XtW @ X
            XtWz = XtW @ z
            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

            delta_b = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12))
            beta = beta_new
            if delta_b < tol:
                break

        eta = X @ beta + offset
        mu = _safe_exp(eta)

        # Profile likelihood for alpha
        def neg_profile_ll(log_alpha):
            a = np.exp(log_alpha)
            return -_nb2_loglik(y, mu, a)

        res = optimize.minimize_scalar(
            neg_profile_ll,
            bounds=(np.log(1e-8), np.log(1e4)),
            method='bounded',
        )
        alpha_new = np.exp(res.x)

        if abs(alpha_new - alpha) / (alpha + 1e-12) < tol:
            converged = True
            alpha = alpha_new
            break
        alpha = alpha_new

    eta = X @ beta + offset
    mu = _safe_exp(eta)
    return beta, mu, alpha, converged, outer + 1


def _nb1_fit(y, X, offset=None, weights=None, maxiter=100, tol=1e-8):
    """NB1 fit: Var(y) = mu * (1 + delta)."""
    n, k = X.shape
    if offset is None:
        offset = np.zeros(n)
    if weights is None:
        weights = np.ones(n)

    beta, mu, _, _ = _poisson_irls(y, X, offset, weights, maxiter=50, tol=1e-6)
    resid = y - mu
    pearson = np.sum(resid**2 / mu) / (n - k)
    delta = max(pearson - 1, 0.01)

    converged = False
    for outer in range(maxiter):
        for inner in range(maxiter):
            eta = X @ beta + offset
            mu = _safe_exp(eta)

            w = weights * mu / (1 + delta)
            z = eta + (y - mu) / mu - offset

            XtW = X.T * w[None, :]
            XtWX = XtW @ X
            XtWz = XtW @ z
            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

            delta_b = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12))
            beta = beta_new
            if delta_b < tol:
                break

        eta = X @ beta + offset
        mu = _safe_exp(eta)

        def neg_profile_ll(log_delta):
            d = np.exp(log_delta)
            return -_nb1_loglik(y, mu, d)

        res = optimize.minimize_scalar(
            neg_profile_ll,
            bounds=(np.log(1e-8), np.log(1e4)),
            method='bounded',
        )
        delta_new = np.exp(res.x)

        if abs(delta_new - delta) / (delta + 1e-12) < tol:
            converged = True
            delta = delta_new
            break
        delta = delta_new

    eta = X @ beta + offset
    mu = _safe_exp(eta)
    return beta, mu, delta, converged, outer + 1


# ---------------------------------------------------------------------------
# PPML helpers: fixed-effect absorption via alternating projection
# ---------------------------------------------------------------------------

def _demean_poisson(y, X, fe_indices_list, mu, maxiter_demean=500, tol_demean=1e-10):
    """
    Demean X (weighted by mu) by absorbing high-dimensional fixed effects
    via alternating projection (Gauss-Seidel on normal equations).

    Parameters
    ----------
    y : ndarray (n,)
    X : ndarray (n, k)
    fe_indices_list : list of ndarray, each (n,) integer-coded FE groups
    mu : ndarray (n,)
        Current Poisson fitted values (used as weights).

    Returns
    -------
    y_tilde, X_tilde : demeaned arrays
    """
    w = mu
    n, k = X.shape

    # Stack y and X for joint demeaning
    Z = np.column_stack([y.reshape(-1, 1) / mu.reshape(-1, 1) * w.reshape(-1, 1),
                         X * w[:, None]])  # weighted

    # Actually we need to demean the working response and weighted X.
    # Working response in IRLS: z_i = eta_i + (y_i - mu_i)/mu_i
    # We demean X*sqrt(w) and z*sqrt(w) for the WLS step.
    sqrt_w = np.sqrt(w)

    # We demean each column of Z by subtracting weighted group means iteratively
    Z_dm = Z.copy()
    for _ in range(maxiter_demean):
        max_change = 0.0
        for fe_idx in fe_indices_list:
            groups = np.unique(fe_idx)
            for g in groups:
                mask = fe_idx == g
                w_g = w[mask]
                w_sum = w_g.sum()
                if w_sum < 1e-300:
                    continue
                group_mean = (Z_dm[mask] * w_g[:, None]).sum(axis=0) / w_sum
                change = np.max(np.abs(group_mean))
                if change > max_change:
                    max_change = change
                Z_dm[mask] -= group_mean[None, :]
        if max_change < tol_demean:
            break

    return Z_dm[:, 0], Z_dm[:, 1:]


def _detect_separation(y, X, fe_indices_list=None):
    """
    Simple separation detection for PPML.

    Checks for regressors that perfectly predict y=0 (i.e., whenever x_j > 0,
    all observations have y=0, or vice versa). Also checks if any FE group has
    all zeros.

    Returns list of warning messages.
    """
    warnings_list = []
    n = len(y)
    zero_mask = y == 0

    # Check regressors
    for j in range(X.shape[1]):
        col = X[:, j]
        # Skip constant
        if np.all(col == col[0]):
            continue
        # Check if positive values of x perfectly predict y=0
        pos_mask = col > 0
        if pos_mask.sum() > 0 and np.all(zero_mask[pos_mask]):
            warnings_list.append(
                f"Possible separation: column {j} > 0 perfectly predicts y=0"
            )

    # Check FE groups
    if fe_indices_list:
        for fe_i, fe_idx in enumerate(fe_indices_list):
            for g in np.unique(fe_idx):
                mask = fe_idx == g
                if np.all(y[mask] == 0):
                    warnings_list.append(
                        f"Separation: FE group {g} (FE dim {fe_i}) has all-zero outcomes"
                    )

    return warnings_list


# ---------------------------------------------------------------------------
# PPML with HDFE via IRLS + alternating projection
# ---------------------------------------------------------------------------

def _ppml_hdfe_irls(y, X, fe_indices_list=None, weights=None,
                    maxiter=1000, tol=1e-8):
    """
    PPML estimation with high-dimensional fixed effects.

    If fe_indices_list is empty/None, reduces to standard Poisson IRLS.
    Otherwise, absorbs FEs via within-transformation at each IRLS step.

    Returns (beta, mu, converged, n_iter).
    """
    n, k = X.shape
    has_fe = fe_indices_list is not None and len(fe_indices_list) > 0
    if weights is None:
        weights = np.ones(n)

    # Initialize
    y_init = np.where(y > 0, y, 0.5)
    if has_fe:
        beta = np.zeros(k)
        # Initialize FE as group log-means
        eta = np.zeros(n)
        for fe_idx in fe_indices_list:
            for g in np.unique(fe_idx):
                mask = fe_idx == g
                gm = np.mean(y_init[mask])
                eta[mask] += np.log(max(gm, 0.1))
        mu = _safe_exp(eta)
    else:
        beta = np.linalg.lstsq(X, np.log(y_init), rcond=None)[0]
        mu = _safe_exp(X @ beta)

    converged = False
    for it in range(maxiter):
        # Working variable and weights
        w = weights * mu
        z = (y - mu) / mu  # working residual (without eta, for demeaning)

        if has_fe:
            # Demean z and X by FEs (weighted by w)
            # Build the full z including current linear predictor component
            z_full = X @ beta + z  # eta_X + working_residual

            # Demean z_full and X
            Z = np.column_stack([z_full.reshape(-1, 1), X])
            Z_dm = Z.copy()
            for _ in range(500):
                max_change = 0.0
                for fe_idx in fe_indices_list:
                    # Vectorized group demeaning
                    for g in np.unique(fe_idx):
                        mask = fe_idx == g
                        w_g = w[mask]
                        w_sum = w_g.sum()
                        if w_sum < 1e-300:
                            continue
                        group_mean = (Z_dm[mask] * w_g[:, None]).sum(axis=0) / w_sum
                        change = np.max(np.abs(group_mean))
                        if change > max_change:
                            max_change = change
                        Z_dm[mask] -= group_mean[None, :]
                if max_change < 1e-10:
                    break

            z_dm = Z_dm[:, 0]
            X_dm = Z_dm[:, 1:]

            # WLS on demeaned data
            XtW = X_dm.T * w[None, :]
            XtWX = XtW @ X_dm
            XtWz = XtW @ z_dm
        else:
            eta = X @ beta
            z = eta + (y - mu) / mu
            XtW = X.T * w[None, :]
            XtWX = XtW @ X
            XtWz = XtW @ z

        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

        # Update mu: need to recover FE contributions
        if has_fe:
            eta_X = X @ beta_new
            # Recover FE: weighted group mean of (log(y+delta) - X*beta_new)
            # Use current working values: eta_fe = weighted mean of (z_full - X*beta) for each group
            # More stable: just re-estimate FE from (y, mu_X) residual
            eta_fe = np.zeros(n)
            resid_from_x = np.log(np.maximum(y, 0.1)) - eta_X
            for fe_idx in fe_indices_list:
                for g in np.unique(fe_idx):
                    mask = fe_idx == g
                    eta_fe[mask] = np.mean(resid_from_x[mask])

            mu_new = _safe_exp(eta_X + eta_fe)
        else:
            mu_new = _safe_exp(X @ beta_new)

        delta = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12))
        beta = beta_new
        mu = mu_new

        if delta < tol:
            converged = True
            break

    return beta, mu, converged, it + 1


# ---------------------------------------------------------------------------
# Cameron-Trivedi overdispersion test
# ---------------------------------------------------------------------------

def _overdispersion_test(y, mu):
    """
    Cameron-Trivedi (1990) test for overdispersion.

    Regresses (y - mu)^2 - y on mu. Under H0 (equidispersion) the coefficient
    on mu is zero.

    Returns (test_stat, p_value).
    """
    dep = (y - mu) ** 2 - y
    X_test = np.column_stack([np.ones_like(mu), mu])
    beta_test = np.linalg.lstsq(X_test, dep, rcond=None)[0]
    resid_test = dep - X_test @ beta_test
    se_test = np.sqrt(np.sum(resid_test**2) / (len(y) - 2) *
                      np.linalg.inv(X_test.T @ X_test)[1, 1])
    t_stat = beta_test[1] / se_test
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
    return float(t_stat), float(p_val)


# ===========================================================================
# Public API
# ===========================================================================

def poisson(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    robust: str = "nonrobust",
    cluster: str = None,
    weights: str = None,
    offset: str = None,
    exposure: str = None,
    irr: bool = False,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Poisson regression via MLE (IRLS).

    Parameters
    ----------
    formula : str, optional
        Model formula, e.g. "y ~ x1 + x2".
    data : pd.DataFrame
        Data containing all variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Independent variable names (alternative to formula).
    robust : str, default "nonrobust"
        Standard error type: "nonrobust", "robust"/"hc0", "hc1".
    cluster : str, optional
        Variable name for clustered standard errors.
    weights : str, optional
        Frequency/analytic weight variable.
    offset : str, optional
        Offset variable (log of exposure already computed).
    exposure : str, optional
        Exposure variable (will be logged and used as offset).
    irr : bool, default False
        If True, report Incidence Rate Ratios (exp(beta)) instead of
        raw coefficients.
    maxiter : int, default 100
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        Fitted model with params, standard errors, diagnostics.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.poisson("num_awards ~ math + prog", data=df)
    >>> print(res.summary())
    >>> res_irr = sp.poisson("num_awards ~ math + prog", data=df,
    ...                       robust="robust", irr=True)
    """
    y_arr, X, var_names, dep_var, _, data = _parse_formula_or_xy(
        formula, data, y, x
    )
    n, k = X.shape

    # Offset / exposure
    offset_arr = np.zeros(n)
    if offset is not None:
        offset_arr = data[offset].values.astype(np.float64)
    if exposure is not None:
        offset_arr = np.log(data[exposure].values.astype(np.float64))

    # Weights
    w_arr = None
    if weights is not None:
        w_arr = data[weights].values.astype(np.float64)

    # Cluster variable
    cluster_arr = None
    if cluster is not None:
        cluster_arr = data[cluster].values

    # Fit
    beta, mu, converged, n_iter = _poisson_irls(
        y_arr, X, offset=offset_arr, weights=w_arr, maxiter=maxiter, tol=tol
    )
    if not converged:
        warnings.warn(f"Poisson IRLS did not converge in {maxiter} iterations")

    residuals = y_arr - mu

    # Variance-covariance
    vcov = _poisson_vcov(X, mu, residuals, robust, cluster_arr)
    se = np.sqrt(np.diag(vcov))

    # Log-likelihood
    ll = _poisson_loglik(y_arr, mu)

    # Null model (intercept only)
    mu_null = np.full(n, np.mean(y_arr))
    ll_null = _poisson_loglik(y_arr, mu_null)

    # LR chi2
    lr_chi2 = 2 * (ll - ll_null)
    lr_pvalue = 1 - stats.chi2.cdf(lr_chi2, k - 1)

    # Pseudo R-squared (McFadden)
    pseudo_r2 = 1 - ll / ll_null

    # AIC, BIC
    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(n) * k

    # Goodness of fit
    deviance = 2 * np.sum(
        np.where(y_arr > 0, y_arr * np.log(np.maximum(y_arr, 1e-300) / mu), 0) - (y_arr - mu)
    )
    pearson_chi2 = np.sum((y_arr - mu) ** 2 / mu)

    # Overdispersion test
    od_stat, od_pval = _overdispersion_test(y_arr, mu)

    # IRR transform
    if irr:
        params_report = np.exp(beta)
        # Delta method: se(exp(b)) = exp(b) * se(b)
        se_report = params_report * se
        coef_label = "IRR"
    else:
        params_report = beta
        se_report = se
        coef_label = "Coefficient"

    params_series = pd.Series(params_report, index=var_names)
    se_series = pd.Series(se_report, index=var_names)

    model_info = {
        'model_type': 'Poisson',
        'family': 'Poisson',
        'link': 'log',
        'method': 'IRLS (MLE)',
        'robust': robust,
        'cluster': cluster,
        'irr': irr,
        'coef_label': coef_label,
        'converged': converged,
        'iterations': n_iter,
        'll': ll,
        'll_null': ll_null,
        'lr_chi2': lr_chi2,
        'lr_pvalue': lr_pvalue,
        'pseudo_r2': pseudo_r2,
        'aic': aic,
        'bic': bic,
    }

    data_info = {
        'nobs': n,
        'df_model': k - 1,
        'df_resid': n - k,
        'dependent_var': dep_var,
        'fitted_values': mu,
        'residuals': residuals,
        'X': X,
        'y': y_arr,
        'var_cov': vcov,
        'var_names': var_names,
        'offset': offset,
        'exposure': exposure,
        'weights': weights,
    }

    diagnostics = {
        'Log-Likelihood': ll,
        'Log-Lik (null)': ll_null,
        'LR chi2': lr_chi2,
        'Prob > chi2': lr_pvalue,
        'Pseudo R2': pseudo_r2,
        'AIC': aic,
        'BIC': bic,
        'Deviance': deviance,
        'Pearson chi2': pearson_chi2,
        'Overdispersion test (C-T)': od_stat,
        'Overdispersion p-value': od_pval,
    }

    return EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def nbreg(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    robust: str = "nonrobust",
    cluster: str = None,
    weights: str = None,
    offset: str = None,
    exposure: str = None,
    irr: bool = False,
    dispersion: str = "mean",
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Negative binomial regression (NB2 or NB1).

    Parameters
    ----------
    formula : str, optional
        Model formula, e.g. "y ~ x1 + x2".
    data : pd.DataFrame
        Data containing all variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Independent variable names (alternative to formula).
    robust : str, default "nonrobust"
        Standard error type: "nonrobust", "robust"/"hc0", "hc1".
    cluster : str, optional
        Variable name for clustered standard errors.
    weights : str, optional
        Weight variable name.
    offset : str, optional
        Offset variable (log of exposure).
    exposure : str, optional
        Exposure variable (will be logged).
    irr : bool, default False
        Report Incidence Rate Ratios.
    dispersion : str, default "mean"
        Dispersion parameterization:
        - "mean" (NB2): Var(y) = mu + alpha * mu^2
        - "constant" (NB1): Var(y) = mu * (1 + delta)
    maxiter : int, default 100
        Maximum iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.nbreg("days_absent ~ math + prog", data=df, irr=True)
    >>> print(res.summary())
    """
    y_arr, X, var_names, dep_var, _, data = _parse_formula_or_xy(
        formula, data, y, x
    )
    n, k = X.shape

    # Offset / exposure
    offset_arr = np.zeros(n)
    if offset is not None:
        offset_arr = data[offset].values.astype(np.float64)
    if exposure is not None:
        offset_arr = np.log(data[exposure].values.astype(np.float64))

    w_arr = None
    if weights is not None:
        w_arr = data[weights].values.astype(np.float64)

    cluster_arr = None
    if cluster is not None:
        cluster_arr = data[cluster].values

    # Fit
    is_nb2 = dispersion.lower() == "mean"
    if is_nb2:
        beta, mu, disp_param, converged, n_iter = _nb2_fit(
            y_arr, X, offset=offset_arr, weights=w_arr,
            maxiter=maxiter, tol=tol,
        )
        disp_label = "alpha"
        nb_label = "NB2"
    else:
        beta, mu, disp_param, converged, n_iter = _nb1_fit(
            y_arr, X, offset=offset_arr, weights=w_arr,
            maxiter=maxiter, tol=tol,
        )
        disp_label = "delta"
        nb_label = "NB1"

    if not converged:
        warnings.warn(f"NegBin did not converge in {maxiter} outer iterations")

    residuals = y_arr - mu

    # Variance-covariance (using NB weights in the bread)
    if is_nb2:
        nb_w = mu / (1 + disp_param * mu)
    else:
        nb_w = mu / (1 + disp_param)

    # Build bread with NB weights
    XtWX = X.T @ (X * nb_w[:, None])
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)

    if cluster_arr is not None:
        vcov = _cluster_vcov(X, nb_w, residuals, cluster_arr)
    elif robust.lower() in ('robust', 'hc0', 'hc1'):
        vcov = _sandwich_vcov(X, nb_w, residuals, XtWX_inv)
        if robust.lower() == 'hc1':
            vcov *= n / (n - k)
    else:
        vcov = XtWX_inv

    se = np.sqrt(np.diag(vcov))

    # Log-likelihood
    if is_nb2:
        ll = _nb2_loglik(y_arr, mu, disp_param)
    else:
        ll = _nb1_loglik(y_arr, mu, disp_param)

    # Null model
    mu_null = np.full(n, np.mean(y_arr))
    if is_nb2:
        # Optimize alpha for null model
        def neg_null_ll(log_a):
            return -_nb2_loglik(y_arr, mu_null, np.exp(log_a))
        res_null = optimize.minimize_scalar(
            neg_null_ll, bounds=(np.log(1e-8), np.log(1e4)), method='bounded'
        )
        ll_null = -res_null.fun
    else:
        def neg_null_ll(log_d):
            return -_nb1_loglik(y_arr, mu_null, np.exp(log_d))
        res_null = optimize.minimize_scalar(
            neg_null_ll, bounds=(np.log(1e-8), np.log(1e4)), method='bounded'
        )
        ll_null = -res_null.fun

    # Poisson ll for LR test of dispersion
    ll_poisson = _poisson_loglik(y_arr, mu)
    lr_alpha = 2 * (ll - ll_poisson)
    # One-sided test (alpha >= 0), use chibar^2 (50:50 mixture of chi2_0 and chi2_1)
    lr_alpha_pvalue = 0.5 * (1 - stats.chi2.cdf(max(lr_alpha, 0), 1))

    lr_chi2 = 2 * (ll - ll_null)
    lr_pvalue = 1 - stats.chi2.cdf(lr_chi2, k - 1)
    pseudo_r2 = 1 - ll / ll_null

    aic = -2 * ll + 2 * (k + 1)  # +1 for dispersion
    bic = -2 * ll + np.log(n) * (k + 1)

    # IRR
    if irr:
        params_report = np.exp(beta)
        se_report = params_report * se
        coef_label = "IRR"
    else:
        params_report = beta
        se_report = se
        coef_label = "Coefficient"

    params_series = pd.Series(params_report, index=var_names)
    se_series = pd.Series(se_report, index=var_names)

    model_info = {
        'model_type': f'NegBin ({nb_label})',
        'family': 'Negative Binomial',
        'link': 'log',
        'method': 'MLE (IRLS + profile likelihood)',
        'dispersion_type': nb_label,
        'robust': robust,
        'cluster': cluster,
        'irr': irr,
        'coef_label': coef_label,
        'converged': converged,
        'iterations': n_iter,
        'dispersion': disp_param,
        'dispersion_label': disp_label,
        'll': ll,
        'll_null': ll_null,
        'lr_chi2': lr_chi2,
        'lr_pvalue': lr_pvalue,
        'pseudo_r2': pseudo_r2,
        'aic': aic,
        'bic': bic,
    }

    data_info = {
        'nobs': n,
        'df_model': k - 1,
        'df_resid': n - k,
        'dependent_var': dep_var,
        'fitted_values': mu,
        'residuals': residuals,
        'X': X,
        'y': y_arr,
        'var_cov': vcov,
        'var_names': var_names,
    }

    diagnostics = {
        'Log-Likelihood': ll,
        'Log-Lik (null)': ll_null,
        'LR chi2': lr_chi2,
        'Prob > chi2': lr_pvalue,
        'Pseudo R2': pseudo_r2,
        'AIC': aic,
        'BIC': bic,
        f'Dispersion ({disp_label})': disp_param,
        'LR test vs Poisson (chi2)': lr_alpha,
        'LR test vs Poisson (p)': lr_alpha_pvalue,
    }

    return EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def ppmlhdfe(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    absorb: str = None,
    robust: str = "robust",
    cluster: str = None,
    weights: str = None,
    separation: bool = True,
    maxiter: int = 1000,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Pseudo-Poisson Maximum Likelihood with high-dimensional fixed effects.

    Implements Santos Silva & Tenreyro (2006) PPML estimator, the standard
    approach for gravity models and other trade/economic settings where:
    - The dependent variable has zeros
    - Log-linearization would be inconsistent under heteroskedasticity
    - High-dimensional fixed effects (origin, destination, year) must be absorbed

    Parameters
    ----------
    formula : str, optional
        Model formula. Fixed effects can be specified via ``|``:
        ``"trade ~ dist + contig | origin + destination + year"``
    data : pd.DataFrame
        Data containing all variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Independent variable names (alternative to formula).
    absorb : str, optional
        Fixed effects to absorb, e.g. ``"origin + destination + year"``.
        Overrides any FE specification in the formula.
    robust : str, default "robust"
        Default is robust SE (as in Stata's ppmlhdfe). Options:
        "robust"/"hc0", "hc1", "nonrobust".
    cluster : str, optional
        Variable name for clustered standard errors (recommended for
        gravity models, e.g. cluster on country-pair).
    weights : str, optional
        Weight variable name.
    separation : bool, default True
        If True, check for separation (perfect prediction of zeros) and
        warn. Observations causing separation are not dropped automatically.
    maxiter : int, default 1000
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults

    Notes
    -----
    PPML is consistent under the assumption E[y|x] = exp(x'beta), regardless
    of the true conditional variance. With robust SE it is a quasi-MLE
    estimator and does not assume Poisson variance.

    References
    ----------
    Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity."
    Review of Economics and Statistics, 88(4), 641-658.

    Examples
    --------
    >>> import statspai as sp
    >>> # Basic gravity model
    >>> res = sp.ppmlhdfe("trade ~ dist + contig | origin + dest + year",
    ...                    data=df, cluster="pair_id")
    >>> print(res.summary())
    >>>
    >>> # With absorb parameter instead of formula FE
    >>> res = sp.ppmlhdfe("trade ~ dist + contig", data=df,
    ...                    absorb="origin + dest + year",
    ...                    cluster="pair_id")
    """
    y_arr, X, var_names, dep_var, formula_fe, data = _parse_formula_or_xy(
        formula, data, y, x, add_constant=True
    )
    n, k = X.shape

    # Parse fixed effects
    fe_indices_list = []
    fe_names = []

    # Absorb parameter takes priority over formula-parsed FE
    if absorb is not None:
        fe_names = [v.strip() for v in absorb.split('+')]
    elif formula_fe:
        fe_names = formula_fe

    for fe_var in fe_names:
        codes, _ = pd.factorize(data[fe_var].values)
        fe_indices_list.append(codes)

    # If we have FE, drop the constant from X (absorbed by FE)
    if fe_indices_list and var_names[0] == '_cons':
        X = X[:, 1:]
        var_names = var_names[1:]
        k = X.shape[1]

    # Weights
    w_arr = None
    if weights is not None:
        w_arr = data[weights].values.astype(np.float64)

    # Cluster variable
    cluster_arr = None
    if cluster is not None:
        cluster_arr = data[cluster].values

    # Separation detection
    sep_warnings = []
    if separation:
        sep_warnings = _detect_separation(y_arr, X, fe_indices_list if fe_indices_list else None)
        for sw in sep_warnings:
            warnings.warn(sw)

    # Fit
    beta, mu, converged, n_iter = _ppml_hdfe_irls(
        y_arr, X,
        fe_indices_list=fe_indices_list if fe_indices_list else None,
        weights=w_arr,
        maxiter=maxiter, tol=tol,
    )
    if not converged:
        warnings.warn(f"PPML did not converge in {maxiter} iterations")

    residuals = y_arr - mu

    # Variance-covariance (robust by default, as in Stata ppmlhdfe)
    vcov = _poisson_vcov(X, mu, residuals, robust, cluster_arr)
    se = np.sqrt(np.diag(vcov))

    # Log-likelihood (Poisson quasi-likelihood)
    ll = _poisson_loglik(y_arr, mu)

    # Null model
    mu_null = np.full(n, np.mean(y_arr))
    ll_null = _poisson_loglik(y_arr, mu_null)

    lr_chi2 = 2 * (ll - ll_null)
    lr_pvalue = 1 - stats.chi2.cdf(lr_chi2, max(k - 1, 1)) if k > 1 else np.nan
    pseudo_r2 = 1 - ll / ll_null

    # Deviance
    deviance = 2 * np.sum(
        np.where(y_arr > 0, y_arr * np.log(np.maximum(y_arr, 1e-300) / mu), 0) - (y_arr - mu)
    )
    pearson_chi2 = np.sum((y_arr - mu) ** 2 / np.maximum(mu, 1e-300))

    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(n) * k

    # Number of FE levels absorbed
    n_fe = sum(len(np.unique(idx)) for idx in fe_indices_list) if fe_indices_list else 0

    params_series = pd.Series(beta, index=var_names)
    se_series = pd.Series(se, index=var_names)

    model_info = {
        'model_type': 'PPML' + (' HDFE' if fe_indices_list else ''),
        'family': 'Poisson (Pseudo-MLE)',
        'link': 'log',
        'method': 'IRLS (Quasi-MLE)' + (' + Alternating Projection' if fe_indices_list else ''),
        'robust': robust,
        'cluster': cluster,
        'converged': converged,
        'iterations': n_iter,
        'll': ll,
        'll_null': ll_null,
        'lr_chi2': lr_chi2,
        'lr_pvalue': lr_pvalue,
        'pseudo_r2': pseudo_r2,
        'aic': aic,
        'bic': bic,
        'absorbed_fe': fe_names if fe_names else None,
        'n_fe_levels': n_fe,
        'separation_warnings': sep_warnings,
    }

    n_cluster = len(np.unique(cluster_arr)) if cluster_arr is not None else None
    data_info = {
        'nobs': n,
        'df_model': k,
        'df_resid': n - k - n_fe,
        'dependent_var': dep_var,
        'fitted_values': mu,
        'residuals': residuals,
        'X': X,
        'y': y_arr,
        'var_cov': vcov,
        'var_names': var_names,
        'n_clusters': n_cluster,
    }

    diagnostics = {
        'Log-Likelihood': ll,
        'Log-Lik (null)': ll_null,
        'Pseudo R2': pseudo_r2,
        'Deviance': deviance,
        'Pearson chi2': pearson_chi2,
        'AIC': aic,
        'BIC': bic,
        'N absorbed FE': n_fe,
    }
    if n_cluster is not None:
        diagnostics['N clusters'] = n_cluster

    return EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
