"""
Truncated regression model.

MLE estimation for outcomes that are only observed within a range
(left-truncated, right-truncated, or both).

Equivalent to Stata's ``truncreg`` and R's ``truncreg::truncreg()``.

References
----------
Hausman, J.A. & Wise, D.A. (1977).
"Social Experimentation, Truncated Distributions, and Efficient
Estimation." *Econometrica*, 45(4), 919-938. [@hausman1977social]
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults


def truncreg(
    data: pd.DataFrame = None,
    y: str = None,
    x: List[str] = None,
    ll: float = None,
    ul: float = None,
    robust: str = "nonrobust",
    cluster: str = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Truncated regression (MLE).

    For samples where the outcome is only observed if it falls
    within [ll, ul]. Different from censoring (Tobit): here the
    *observation itself* is missing outside the range.

    Equivalent to Stata's ``truncreg y x, ll(0)`` and
    R's ``truncreg::truncreg()``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x : list of str
        Regressors.
    ll : float, optional
        Lower truncation point. None = no lower truncation.
    ul : float, optional
        Upper truncation point. None = no upper truncation.
    robust : str, default 'nonrobust'
    cluster : str, optional
    maxiter : int, default 200
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> # Wages observed only if > 0 (truncated at 0)
    >>> result = sp.truncreg(df, y='wage', x=['education', 'experience'], ll=0)
    >>> print(result.summary())
    """
    if ll is not None and ul is not None and ll >= ul:
        raise ValueError(f"Lower limit ({ll}) must be < upper limit ({ul})")

    df = data.dropna(subset=[y] + x)
    n = len(df)

    y_data = df[y].values.astype(float)
    X_data = np.column_stack([np.ones(n), df[x].values.astype(float)])
    k = X_data.shape[1]
    var_names = ['_cons'] + list(x)

    def neg_log_lik(theta):
        beta = theta[:k]
        ln_sigma = theta[k]
        sigma = np.exp(ln_sigma)

        xb = X_data @ beta
        z = (y_data - xb) / sigma

        # Log density: log φ(z) - log σ
        log_pdf = stats.norm.logpdf(z) - ln_sigma

        # Truncation adjustment
        if ll is not None and ul is not None:
            z_ll = (ll - xb) / sigma
            z_ul = (ul - xb) / sigma
            log_denom = np.log(np.clip(stats.norm.cdf(z_ul) - stats.norm.cdf(z_ll), 1e-20, None))
        elif ll is not None:
            z_ll = (ll - xb) / sigma
            log_denom = np.log(np.clip(1 - stats.norm.cdf(z_ll), 1e-20, None))
        elif ul is not None:
            z_ul = (ul - xb) / sigma
            log_denom = np.log(np.clip(stats.norm.cdf(z_ul), 1e-20, None))
        else:
            log_denom = 0

        return -np.sum(log_pdf - log_denom)

    # Initialize with OLS
    beta_init = np.linalg.lstsq(X_data, y_data, rcond=None)[0]
    resid = y_data - X_data @ beta_init
    sigma_init = np.std(resid)
    theta0 = np.concatenate([beta_init, [np.log(max(sigma_init, 0.1))]])

    result = minimize(neg_log_lik, theta0, method='BFGS',
                      options={'maxiter': maxiter, 'gtol': tol})
    theta_hat = result.x

    beta_hat = theta_hat[:k]
    sigma_hat = np.exp(theta_hat[k])

    # Standard errors via numerical Hessian
    eps = 1e-5
    k_total = len(theta_hat)
    H = np.zeros((k_total, k_total))
    f0 = neg_log_lik(theta_hat)
    for i in range(k_total):
        ei = np.zeros(k_total)
        ei[i] = eps
        fp = neg_log_lik(theta_hat + ei)
        fm = neg_log_lik(theta_hat - ei)
        H[i, i] = (fp - 2 * f0 + fm) / eps**2
        for j in range(i + 1, k_total):
            ej = np.zeros(k_total)
            ej[j] = eps
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

    all_names = var_names + ['ln_sigma']
    all_params = np.concatenate([beta_hat, [theta_hat[k]]])

    params = pd.Series(all_params, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    ll_val = -neg_log_lik(theta_hat)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            'model_type': 'Truncated Regression',
            'lower_limit': ll,
            'upper_limit': ul,
            'sigma': sigma_hat,
            'converged': result.success,
        },
        data_info={
            'n_obs': n,
            'dep_var': y,
            'df_resid': n - k - 1,
        },
        diagnostics={
            'log_likelihood': ll_val,
            'sigma': sigma_hat,
            'aic': -2 * ll_val + 2 * k_total,
            'bic': -2 * ll_val + np.log(n) * k_total,
        },
    )
