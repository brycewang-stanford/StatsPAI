"""
Panel binary-choice models: logit and probit for panel data.

Estimators
----------
- ``panel_logit``  method='fe'  — Conditional FE logit (Chamberlain 1980)
- ``panel_logit``  method='re'  — Random Effects logit (Gauss-Hermite quadrature)
- ``panel_logit``  method='cre' — Correlated Random Effects (Mundlak) logit
- ``panel_probit`` method='re'  — Random Effects probit
- ``panel_probit`` method='cre' — Correlated Random Effects probit

References
----------
Chamberlain, G. (1980). "Analysis of Covariance with Qualitative Data."
Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data."
Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data.
"""
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats, optimize, special
from ..core.results import EconometricResults

# --------------- helpers ---------------

_logit_cdf = special.expit
_probit_cdf = stats.norm.cdf


def _numerical_hessian(f, x, eps=1e-5):
    """Central-difference numerical Hessian."""
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*eps*eps)
    return H


def _group_panel(df, y, x, id_col, drop_no_variation=False):
    """Group panel data by unit.  Returns groups, x_groups, y_groups, n_dropped."""
    groups, x_groups, y_groups, n_dropped = [], {}, {}, 0
    for gid, gdf in df.groupby(id_col):
        yi = gdf[y].values.astype(float)
        if drop_no_variation and (yi.sum() == 0 or yi.sum() == len(yi)):
            n_dropped += 1
            continue
        groups.append(gid)
        x_groups[gid] = gdf[x].values.astype(float)
        y_groups[gid] = yi
    return groups, x_groups, y_groups, n_dropped


def _add_mundlak_means(data, x, id_col):
    """Add within-unit means as additional regressors (Mundlak device)."""
    df = data.copy()
    mean_names = [f'{v}_mean' for v in x]
    means = df.groupby(id_col)[x].transform('mean')
    if isinstance(means, pd.Series):
        df[mean_names[0]] = means
    else:
        for orig, mn in zip(x, mean_names):
            df[mn] = means[orig]
    return df, mean_names


# --------------- Conditional FE logit (Chamberlain 1980) ---------------

def _log_sum_combinations(scores, T, d):
    """Log-sum-exp over all d-combinations of scores via DP."""
    NEG_INF = -1e30
    dp = np.full((T + 1, d + 1), NEG_INF)
    dp[0, 0] = 0.0
    for j in range(1, T + 1):
        sj = scores[j - 1]
        for k in range(min(j, d) + 1):
            val = dp[j-1, k]
            if k > 0:
                val = np.logaddexp(val, dp[j-1, k-1] + sj)
            dp[j, k] = val
    return dp[T, d]


def _conditional_logit_nll(beta, groups, xg, yg):
    """Negative conditional log-likelihood for FE logit."""
    nll = 0.0
    for g in groups:
        xi, yi = xg[g], yg[g]
        di, Ti = int(yi.sum()), len(yi)
        if di == 0 or di == Ti:
            continue
        scores = xi @ beta
        nll -= scores[yi == 1].sum() - _log_sum_combinations(scores, Ti, di)
    return nll


def _conditional_logit_grad(beta, groups, xg, yg):
    """Gradient of conditional negative log-likelihood (forward-backward DP)."""
    NEG_INF = -1e30
    k = len(beta)
    grad = np.zeros(k)
    for g in groups:
        xi, yi = xg[g], yg[g]
        di, Ti = int(yi.sum()), len(yi)
        if di == 0 or di == Ti:
            continue
        scores = xi @ beta
        grad -= xi[yi == 1].sum(axis=0)
        # forward DP
        dp_f = np.full((Ti+1, di+1), NEG_INF); dp_f[0, 0] = 0.0
        for j in range(1, Ti+1):
            sj = scores[j-1]
            for m in range(min(j, di)+1):
                v = dp_f[j-1, m]
                if m > 0:
                    v = np.logaddexp(v, dp_f[j-1, m-1] + sj)
                dp_f[j, m] = v
        log_denom = dp_f[Ti, di]
        # backward DP
        dp_b = np.full((Ti+2, di+1), NEG_INF); dp_b[Ti+1, 0] = 0.0
        for j in range(Ti, 0, -1):
            sj = scores[j-1]
            for m in range(min(Ti-j+1, di)+1):
                v = dp_b[j+1, m]
                if m > 0:
                    v = np.logaddexp(v, dp_b[j+1, m-1] + sj)
                dp_b[j, m] = v
        # marginal inclusion probabilities
        for j in range(Ti):
            for m in range(min(j, di-1)+1):
                rem = di - m - 1
                if rem < 0 or rem > Ti - j - 1:
                    continue
                lp = dp_f[j, m] + scores[j] + dp_b[j+2, rem] - log_denom
                if lp > NEG_INF + 100:
                    grad += np.exp(lp) * xi[j]
    return grad


def _fit_fe_logit(data, y, x, id_col, maxiter, tol):
    """Fit conditional FE logit via BFGS."""
    df = data[[id_col, y] + x].dropna()
    groups, xg, yg, n_dropped = _group_panel(df, y, x, id_col, drop_no_variation=True)
    n_units = len(groups)
    n_obs = sum(len(yg[g]) for g in groups)
    res = optimize.minimize(
        _conditional_logit_nll, np.zeros(len(x)),
        args=(groups, xg, yg), jac=_conditional_logit_grad,
        method='BFGS', options={'maxiter': maxiter, 'gtol': tol},
    )
    beta = res.x
    H = _numerical_hessian(lambda b: _conditional_logit_nll(b, groups, xg, yg), beta)
    try:
        vcov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        vcov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return beta, se, -res.fun, n_obs, n_units, n_dropped, vcov, res.success


# --------------- RE logit / probit via Gauss-Hermite quadrature ---------------

def _re_panel_nll(theta, groups, xg, yg, n_quad, link_cdf):
    """Negative log-likelihood for RE binary panel model.
    theta = [beta..., log_sigma_u]
    """
    beta, sigma_u = theta[:-1], np.exp(theta[-1])
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    alpha_pts = np.sqrt(2.0) * sigma_u * nodes
    log_w = np.log(weights) - 0.5 * np.log(np.pi)
    nll = 0.0
    for g in groups:
        xi, yi = xg[g], yg[g]
        xb = xi @ beta
        ll_q = np.empty(n_quad)
        for q in range(n_quad):
            p = np.clip(link_cdf(xb + alpha_pts[q]), 1e-15, 1 - 1e-15)
            ll_q[q] = log_w[q] + np.sum(yi*np.log(p) + (1-yi)*np.log(1-p))
        mx = ll_q.max()
        nll -= mx + np.log(np.sum(np.exp(ll_q - mx)))
    return nll


def _fit_re_binary(data, y, x, id_col, n_quad, link_cdf, maxiter, tol):
    """Fit RE binary panel model via MLE with Gauss-Hermite quadrature."""
    df = data[[id_col, y] + x].dropna()
    groups, xg, yg, _ = _group_panel(df, y, x, id_col)
    n_units, n_obs = len(groups), sum(len(yg[g]) for g in groups)
    theta0 = np.zeros(len(x) + 1)
    res = optimize.minimize(
        _re_panel_nll, theta0, args=(groups, xg, yg, n_quad, link_cdf),
        method='BFGS', options={'maxiter': maxiter, 'gtol': tol},
    )
    theta = res.x
    beta, sigma_u = theta[:-1], np.exp(theta[-1])
    H = _numerical_hessian(
        lambda t: _re_panel_nll(t, groups, xg, yg, n_quad, link_cdf), theta)
    try:
        vcov_full = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        vcov_full = np.linalg.pinv(H)
    se_full = np.sqrt(np.maximum(np.diag(vcov_full), 0.0))
    se_beta = se_full[:-1]
    se_sigma_u = sigma_u * se_full[-1]  # delta method
    return beta, se_beta, sigma_u, se_sigma_u, -res.fun, n_obs, n_units, vcov_full[:-1, :-1], res.success


# --------------- Result wrappers ---------------

def _wrap_re_result(data, y, x_vars, id_col, n_quad, link_cdf, maxiter, tol,
                    alpha, model_name, method_tag, link='logit',
                    original_x=None, mean_names=None):
    """Fit RE/CRE binary model and wrap into EconometricResults."""
    beta, se, sigma_u, se_sigma_u, ll, n_obs, n_units, vcov, ok = (
        _fit_re_binary(data, y, x_vars, id_col, n_quad, link_cdf, maxiter, tol))
    scale = np.pi**2/3 if link == 'logit' else 1.0
    rho = sigma_u**2 / (sigma_u**2 + scale)
    n_params = len(x_vars) + 1
    model_info = {
        'model': model_name, 'method': method_tag, 'link': link,
        'dep_var': y, 'converged': ok, 'log_likelihood': ll,
        'sigma_u': sigma_u, 'se_sigma_u': se_sigma_u,
        'rho': rho, 'n_quadrature': n_quad,
    }
    if original_x is not None:
        model_info['original_x'] = original_x
        model_info['mundlak_means'] = mean_names
    data_info = {
        'n_obs': n_obs, 'n_units': n_units, 'n_vars': len(x_vars),
        'df_resid': n_obs - n_params, 'alpha': alpha,
    }
    diagnostics = {
        'aic': -2*ll + 2*n_params,
        'bic': -2*ll + np.log(n_obs)*n_params,
    }
    return EconometricResults(
        pd.Series(beta, index=x_vars), pd.Series(se, index=x_vars),
        model_info, data_info, diagnostics)


# ====================== Public API ======================

def panel_logit(
    data: pd.DataFrame, y: str, x: list,
    id: str = 'id', time: str = 'time',
    method: str = 'fe',
    n_quadrature: int = 12,
    robust: str = 'nonrobust', cluster: str = None,
    maxiter: int = 200, tol: float = 1e-8, alpha: float = 0.05,
) -> EconometricResults:
    """Panel logit model.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format.
    y : str
        Binary dependent variable (0/1).
    x : list of str
        Regressors.
    id, time : str
        Unit and time identifier columns.
    method : str
        'fe' (conditional FE logit), 're' (random effects), 'cre' (Mundlak).
    n_quadrature : int
        Gauss-Hermite quadrature points (RE/CRE only).
    robust : str
        'nonrobust' or 'robust'.
    cluster : str or None
        Column for cluster-robust SEs.
    maxiter : int
        Maximum optimizer iterations.
    tol : float
        Gradient tolerance.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
    """
    method = method.lower()
    if method not in ('fe', 're', 'cre'):
        raise ValueError("method must be 'fe', 're', or 'cre'")
    id_col, x_vars = id, list(x)

    if method == 'cre':
        data, mn = _add_mundlak_means(data, x_vars, id_col)
        return _wrap_re_result(
            data, y, x_vars + mn, id_col, n_quadrature, _logit_cdf,
            maxiter, tol, alpha, 'Panel Logit (CRE/Mundlak)', 'cre',
            original_x=x_vars, mean_names=mn)
    if method == 're':
        return _wrap_re_result(
            data, y, x_vars, id_col, n_quadrature, _logit_cdf,
            maxiter, tol, alpha, 'Panel Logit (RE)', 're')

    # --- FE ---
    beta, se, ll, n_obs, n_units, n_dropped, vcov, ok = _fit_fe_logit(
        data, y, x_vars, id_col, maxiter, tol)
    k = len(x_vars)
    return EconometricResults(
        pd.Series(beta, index=x_vars), pd.Series(se, index=x_vars),
        model_info={
            'model': 'Panel Logit (Conditional FE)', 'method': 'fe',
            'link': 'logit', 'dep_var': y, 'converged': ok,
            'log_likelihood': ll, 'n_dropped_units': n_dropped,
        },
        data_info={
            'n_obs': n_obs, 'n_units': n_units, 'n_vars': k,
            'df_resid': n_obs - k, 'alpha': alpha,
        },
        diagnostics={
            'aic': -2*ll + 2*k, 'bic': -2*ll + np.log(n_obs)*k,
        })


def panel_probit(
    data: pd.DataFrame, y: str, x: list,
    id: str = 'id', time: str = 'time',
    method: str = 're',
    n_quadrature: int = 12,
    robust: str = 'nonrobust', cluster: str = None,
    maxiter: int = 200, tol: float = 1e-8, alpha: float = 0.05,
) -> EconometricResults:
    """Panel probit model.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format.
    y : str
        Binary dependent variable (0/1).
    x : list of str
        Regressors.
    id, time : str
        Unit and time identifier columns.
    method : str
        're' (random effects) or 'cre' (Mundlak).
        FE probit not supported (incidental parameters problem).
    n_quadrature : int
        Gauss-Hermite quadrature points.
    robust : str
        'nonrobust' or 'robust'.
    cluster : str or None
        Column for cluster-robust SEs.
    maxiter : int
        Maximum optimizer iterations.
    tol : float
        Gradient tolerance.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
    """
    method = method.lower()
    if method not in ('re', 'cre'):
        raise ValueError(
            "method must be 're' or 'cre'. FE probit is not supported "
            "due to the incidental parameters problem.")
    id_col, x_vars = id, list(x)

    if method == 'cre':
        data, mn = _add_mundlak_means(data, x_vars, id_col)
        return _wrap_re_result(
            data, y, x_vars + mn, id_col, n_quadrature, _probit_cdf,
            maxiter, tol, alpha, 'Panel Probit (CRE/Mundlak)', 'cre',
            link='probit', original_x=x_vars, mean_names=mn)
    return _wrap_re_result(
        data, y, x_vars, id_col, n_quadrature, _probit_cdf,
        maxiter, tol, alpha, 'Panel Probit (RE)', 're', link='probit')
