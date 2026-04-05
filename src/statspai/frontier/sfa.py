"""
Stochastic Frontier Analysis (SFA).

Estimates production/cost frontiers with composed error:
    y_i = x_i'β + v_i - u_i  (production)
    y_i = x_i'β + v_i + u_i  (cost)

where v_i ~ N(0, σ_v²) is noise and u_i ≥ 0 is inefficiency.

Equivalent to Stata's ``frontier`` and R's ``sfa::sfa()``.

References
----------
Aigner, D., Lovell, C.A.K. & Schmidt, P. (1977).
"Formulation and Estimation of Stochastic Frontier Production
Function Models." *Journal of Econometrics*, 6(1), 21-37.

Battese, G.E. & Coelli, T.J. (1995).
"A Model for Technical Inefficiency Effects in a Stochastic
Frontier Production Function for Panel Data."
*Empirical Economics*, 20(2), 325-332.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults


class FrontierResult(EconometricResults):
    """Extended results for stochastic frontier models."""

    def efficiency(self) -> pd.Series:
        """Technical efficiency estimates E[exp(-u_i) | ε_i]."""
        if 'efficiency' in self.diagnostics:
            return pd.Series(self.diagnostics['efficiency'], name='efficiency')
        return pd.Series(dtype=float)


def frontier(
    data: pd.DataFrame = None,
    y: str = None,
    x: List[str] = None,
    dist: str = "half-normal",
    cost: bool = False,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> FrontierResult:
    """
    Stochastic frontier model.

    Estimates a production or cost frontier with composed error.

    Equivalent to Stata's ``frontier y x, dist(hnormal)`` and
    R's ``sfa::sfa()``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Output (production) or cost variable.
    x : list of str
        Input/cost variables.
    dist : str, default 'half-normal'
        Inefficiency distribution: 'half-normal', 'exponential', 'truncated-normal'.
    cost : bool, default False
        If True, estimate cost frontier (u_i enters positively).
    maxiter : int, default 200
    alpha : float, default 0.05

    Returns
    -------
    FrontierResult
        With .efficiency() method for unit-level TE estimates.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.frontier(df, y='log_output', x=['log_labor', 'log_capital'])
    >>> print(result.summary())
    >>> eff = result.efficiency()
    """
    df = data.dropna(subset=[y] + x)
    n = len(df)

    y_data = df[y].values.astype(float)
    X_data = np.column_stack([np.ones(n), df[x].values.astype(float)])
    k = X_data.shape[1]
    var_names = ['_cons'] + list(x)

    sign = 1 if cost else -1  # sign of u in composed error

    if dist == 'half-normal':
        def neg_log_lik(theta):
            beta = theta[:k]
            ln_sigma_v = theta[k]
            ln_sigma_u = theta[k + 1]
            sigma_v = np.exp(ln_sigma_v)
            sigma_u = np.exp(ln_sigma_u)
            sigma = np.sqrt(sigma_v**2 + sigma_u**2)
            lam = sigma_u / sigma_v

            eps = y_data - X_data @ beta
            z = sign * eps * lam / sigma

            ll = np.sum(
                -0.5 * np.log(2 * np.pi) - np.log(sigma)
                - 0.5 * (eps / sigma)**2
                + np.log(2 * stats.norm.cdf(z))
            )
            return -ll

        n_extra = 2  # ln_sigma_v, ln_sigma_u

    elif dist == 'exponential':
        def neg_log_lik(theta):
            beta = theta[:k]
            ln_sigma_v = theta[k]
            ln_sigma_u = theta[k + 1]
            sigma_v = np.exp(ln_sigma_v)
            sigma_u = np.exp(ln_sigma_u)

            eps = y_data - X_data @ beta
            mu_star = -sign * eps - sigma_v**2 / sigma_u
            sigma_star = sigma_v

            ll = np.sum(
                -np.log(sigma_u) + 0.5 * (sigma_v / sigma_u)**2
                + sign * eps / sigma_u
                + np.log(np.clip(stats.norm.cdf(mu_star / sigma_star), 1e-20, None))
            )
            return -ll

        n_extra = 2

    elif dist == 'truncated-normal':
        def neg_log_lik(theta):
            beta = theta[:k]
            ln_sigma_v = theta[k]
            ln_sigma_u = theta[k + 1]
            mu = theta[k + 2]  # mean of truncated normal
            sigma_v = np.exp(ln_sigma_v)
            sigma_u = np.exp(ln_sigma_u)
            sigma = np.sqrt(sigma_v**2 + sigma_u**2)
            lam = sigma_u / sigma_v

            eps = y_data - X_data @ beta
            mu_star = (sign * eps * sigma_u**2 - mu * sigma_v**2) / sigma**2 * (-1)
            # Actually: mu_i* = (-sign*eps*sigma_u^2 + mu*sigma_v^2) / sigma^2
            mu_star2 = (-sign * eps * sigma_u**2 + mu * sigma_v**2) / sigma**2
            sigma_star = sigma_v * sigma_u / sigma

            ll = np.sum(
                -0.5 * np.log(2 * np.pi) - np.log(sigma)
                - 0.5 * ((eps + sign * mu) / sigma)**2
                + np.log(np.clip(stats.norm.cdf(mu_star2 / sigma_star), 1e-20, None))
                - np.log(np.clip(stats.norm.cdf(mu / sigma_u), 1e-20, None))
            )
            return -ll

        n_extra = 3
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Initialize with OLS
    beta_init = np.linalg.lstsq(X_data, y_data, rcond=None)[0]
    resid = y_data - X_data @ beta_init
    sigma_init = np.std(resid)
    theta0 = np.concatenate([
        beta_init,
        [np.log(sigma_init * 0.7), np.log(sigma_init * 0.7)],
    ])
    if dist == 'truncated-normal':
        theta0 = np.concatenate([theta0, [0.0]])

    result = minimize(neg_log_lik, theta0, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': tol})
    theta_hat = result.x

    beta_hat = theta_hat[:k]
    sigma_v = np.exp(theta_hat[k])
    sigma_u = np.exp(theta_hat[k + 1])
    sigma = np.sqrt(sigma_v**2 + sigma_u**2)
    lam = sigma_u / sigma_v

    # SE via numerical Hessian
    k_total = len(theta_hat)
    eps_h = 1e-5
    H = np.zeros((k_total, k_total))
    f0 = neg_log_lik(theta_hat)
    for i in range(k_total):
        ei = np.zeros(k_total)
        ei[i] = eps_h
        fp = neg_log_lik(theta_hat + ei)
        fm = neg_log_lik(theta_hat - ei)
        H[i, i] = (fp - 2 * f0 + fm) / eps_h**2
        for j in range(i + 1, k_total):
            ej = np.zeros(k_total)
            ej[j] = eps_h
            fpp = neg_log_lik(theta_hat + ei + ej)
            fpm = neg_log_lik(theta_hat + ei - ej)
            fmp = neg_log_lik(theta_hat - ei + ej)
            fmm = neg_log_lik(theta_hat - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps_h**2)

    try:
        var_cov = np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(var_cov)))
    except np.linalg.LinAlgError:
        se = np.full(k_total, np.nan)

    # Technical efficiency: E[exp(-u_i) | ε_i]  (Jondrow et al. 1982)
    eps_hat = y_data - X_data @ beta_hat
    if dist == 'half-normal':
        mu_star = -sign * eps_hat * sigma_u**2 / sigma**2
        sigma_star = sigma_v * sigma_u / sigma
        # E[u|ε] = μ* + σ* × φ(μ*/σ*) / Φ(μ*/σ*)
        ratio = mu_star / sigma_star
        E_u = mu_star + sigma_star * stats.norm.pdf(ratio) / np.clip(stats.norm.cdf(ratio), 1e-20, None)
        efficiency = np.exp(-E_u)
    else:
        efficiency = np.full(n, np.nan)

    efficiency = np.clip(efficiency, 0, 1)

    # Build results
    extra_names = ['ln_sigma_v', 'ln_sigma_u']
    if dist == 'truncated-normal':
        extra_names.append('mu')

    all_names = var_names + extra_names
    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    ll_val = -neg_log_lik(theta_hat)

    return FrontierResult(
        params=params,
        std_errors=std_errors,
        model_info={
            'model_type': f"Stochastic Frontier ({'Cost' if cost else 'Production'})",
            'inefficiency_dist': dist,
            'sigma_v': sigma_v,
            'sigma_u': sigma_u,
            'lambda': lam,
            'mean_efficiency': float(np.nanmean(efficiency)),
            'converged': result.success,
        },
        data_info={
            'n_obs': n,
            'dep_var': y,
            'df_resid': n - k_total,
        },
        diagnostics={
            'log_likelihood': ll_val,
            'sigma_v': sigma_v,
            'sigma_u': sigma_u,
            'sigma': sigma,
            'lambda': lam,
            'aic': -2 * ll_val + 2 * k_total,
            'bic': -2 * ll_val + np.log(n) * k_total,
            'efficiency': efficiency,
            'mean_efficiency': float(np.nanmean(efficiency)),
        },
    )
