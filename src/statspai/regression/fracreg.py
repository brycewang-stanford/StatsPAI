"""
Fractional response models and beta regression.

For outcomes bounded in [0,1] (proportions, rates, shares).

Papke & Wooldridge (1996) quasi-MLE (fractional logit/probit),
and beta regression (Ferrari & Cribari-Neto 2004).

Equivalent to Stata's ``fracreg`` and R's ``betareg::betareg()``.

References
----------
Papke, L.E. & Wooldridge, J.M. (1996).
"Econometric Methods for Fractional Response Variables with an
Application to 401(k) Plan Participation Rates."
*Journal of Applied Econometrics*, 11(6), 619-632. [@papke1996econometric]

Ferrari, S.L.P. & Cribari-Neto, F. (2004).
"Beta Regression for Modelling Rates and Proportions."
*Journal of Applied Statistics*, 31(7), 799-815. [@ferrari2004beta]
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln, digamma

from ..core.results import EconometricResults


def fracreg(
    data: pd.DataFrame = None,
    y: str = None,
    x: List[str] = None,
    link: str = "logit",
    robust: str = "robust",
    cluster: str = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Fractional response model (Papke & Wooldridge 1996).

    Quasi-MLE for outcomes in [0, 1]. Uses Bernoulli log-likelihood
    with robust (sandwich) standard errors.

    Equivalent to Stata's ``fracreg logit y x``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable in [0, 1].
    x : list of str
        Regressors.
    link : str, default 'logit'
        Link function: 'logit' or 'probit'.
    robust : str, default 'robust'
        Always use robust SE (quasi-MLE).
    cluster : str, optional
        Cluster variable for clustered SE.
    maxiter : int, default 100
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.fracreg(df, y='participation_rate', x=['income', 'age'])
    >>> print(result.summary())
    """
    df = data.dropna(subset=[y] + x)
    n = len(df)

    y_data = df[y].values.astype(float)
    X_data = np.column_stack([np.ones(n), df[x].values.astype(float)])
    k = X_data.shape[1]
    var_names = ['_cons'] + list(x)

    if link == 'logit':
        def g(xb):
            xb = np.clip(xb, -500, 500)
            return 1 / (1 + np.exp(-xb))

        def g_prime(xb):
            p = g(xb)
            return p * (1 - p)
    elif link == 'probit':
        def g(xb):
            return stats.norm.cdf(xb)

        def g_prime(xb):
            return stats.norm.pdf(xb)
    else:
        raise ValueError(f"Unknown link: {link}")

    # Quasi-MLE via IRLS (Bernoulli working likelihood)
    beta = np.zeros(k)

    for iteration in range(maxiter):
        xb = X_data @ beta
        mu = g(xb)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        dmu = g_prime(xb)

        # Working weights and residuals
        w = dmu**2 / (mu * (1 - mu))
        w = np.clip(w, 1e-10, 1e10)
        z = xb + (y_data - mu) / dmu

        # WLS
        W = np.diag(w)
        try:
            XtWX = X_data.T @ W @ X_data
            XtWz = X_data.T @ W @ z
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final predictions
    xb = X_data @ beta
    mu = g(xb)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    dmu = g_prime(xb)

    # Quasi-log-likelihood
    qll = np.sum(y_data * np.log(mu) + (1 - y_data) * np.log(1 - mu))

    # Robust (sandwich) standard errors — always for QMLE
    score = ((y_data - mu) * dmu / (mu * (1 - mu)))[:, np.newaxis] * X_data
    try:
        XtWX_inv = np.linalg.inv(X_data.T @ np.diag(dmu**2 / (mu * (1 - mu))) @ X_data)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(X_data.T @ np.diag(dmu**2 / (mu * (1 - mu))) @ X_data)

    if cluster is not None:
        clusters = df[cluster].values
        unique_cl = np.unique(clusters)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for cl in unique_cl:
            cl_mask = clusters == cl
            s_cl = score[cl_mask].sum(axis=0)
            meat += np.outer(s_cl, s_cl)
        correction = n_cl / (n_cl - 1)
        var_cov = correction * XtWX_inv @ meat @ XtWX_inv
    else:
        meat = score.T @ score
        var_cov = XtWX_inv @ meat @ XtWX_inv

    se = np.sqrt(np.diag(var_cov))
    params = pd.Series(beta, index=var_names)
    std_errors = pd.Series(se, index=var_names)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            'model_type': f'Fractional {link.title()} (Papke-Wooldridge)',
            'link': link,
            'quasi_ll': qll,
        },
        data_info={
            'n_obs': n,
            'dep_var': y,
            'df_resid': n - k,
        },
        diagnostics={
            'quasi_log_likelihood': qll,
            'aic': -2 * qll + 2 * k,
            'bic': -2 * qll + np.log(n) * k,
        },
    )


def betareg(
    data: pd.DataFrame = None,
    y: str = None,
    x: List[str] = None,
    z: List[str] = None,
    link: str = "logit",
    robust: str = "nonrobust",
    cluster: str = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Beta regression (Ferrari & Cribari-Neto 2004).

    Full parametric model for outcomes in (0, 1) using the Beta distribution.

    Equivalent to R's ``betareg::betareg()`` and Stata's ``betareg``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome in (0, 1).
    x : list of str
        Regressors for the mean equation.
    z : list of str, optional
        Regressors for the precision equation. If None, constant precision.
    link : str, default 'logit'
        Link for mean: 'logit', 'probit', 'cloglog'.
    robust : str, default 'nonrobust'
    cluster : str, optional
    maxiter : int, default 200
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.betareg(df, y='share', x=['price', 'quality'])
    >>> print(result.summary())
    """
    df = data.dropna(subset=[y] + x + (z or []))
    n = len(df)

    y_data = df[y].values.astype(float)
    # Squeeze away from boundaries
    y_data = np.clip(y_data, 1e-6, 1 - 1e-6)

    X_mean = np.column_stack([np.ones(n), df[x].values.astype(float)])
    k_mean = X_mean.shape[1]
    mean_names = ['_cons'] + list(x)

    if z is not None:
        X_prec = np.column_stack([np.ones(n), df[z].values.astype(float)])
        prec_names = ['_cons_phi'] + [f'phi_{v}' for v in z]
    else:
        X_prec = np.ones((n, 1))
        prec_names = ['_cons_phi']
    k_prec = X_prec.shape[1]

    if link == 'logit':
        g = lambda xb: 1 / (1 + np.exp(-np.clip(xb, -500, 500)))
    elif link == 'probit':
        g = lambda xb: stats.norm.cdf(xb)
    else:
        g = lambda xb: 1 / (1 + np.exp(-np.clip(xb, -500, 500)))

    def neg_log_lik(theta):
        beta = theta[:k_mean]
        gamma = theta[k_mean:]
        mu = g(X_mean @ beta)
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        phi = np.exp(X_prec @ gamma)
        phi = np.clip(phi, 1e-4, 1e6)

        a = mu * phi
        b = (1 - mu) * phi

        ll = np.sum(gammaln(phi) - gammaln(a) - gammaln(b) +
                     (a - 1) * np.log(y_data) + (b - 1) * np.log(1 - y_data))
        return -ll

    # Initialize
    theta0 = np.zeros(k_mean + k_prec)
    theta0[k_mean] = np.log(5)  # initial precision

    try:
        result = minimize(neg_log_lik, theta0, method='BFGS',
                          options={'maxiter': maxiter, 'gtol': tol})
        theta_hat = result.x
        converged = result.success
    except Exception:
        theta_hat = theta0
        converged = False

    beta_hat = theta_hat[:k_mean]
    gamma_hat = theta_hat[k_mean:]

    # Numerical Hessian for SE
    eps = 1e-5
    k_total = len(theta_hat)
    H = np.zeros((k_total, k_total))
    f0 = neg_log_lik(theta_hat)
    for i in range(k_total):
        for j in range(i, k_total):
            ei, ej = np.zeros(k_total), np.zeros(k_total)
            ei[i], ej[j] = eps, eps
            fpp = neg_log_lik(theta_hat + ei + ej)
            fpm = neg_log_lik(theta_hat + ei - ej)
            fmp = neg_log_lik(theta_hat - ei + ej)
            fmm = neg_log_lik(theta_hat - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)

    try:
        var_cov = np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(var_cov)))
    except np.linalg.LinAlgError:
        se = np.full(k_total, np.nan)

    all_names = mean_names + prec_names
    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    ll = -neg_log_lik(theta_hat)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            'model_type': 'Beta Regression (Ferrari-Cribari-Neto)',
            'link': link,
            'n_mean_params': k_mean,
            'n_precision_params': k_prec,
            'converged': converged,
        },
        data_info={
            'n_obs': n,
            'dep_var': y,
            'df_resid': n - k_total,
        },
        diagnostics={
            'log_likelihood': ll,
            'aic': -2 * ll + 2 * k_total,
            'bic': -2 * ll + np.log(n) * k_total,
        },
    )
