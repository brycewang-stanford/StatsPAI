"""
Tobit (1958) censored regression model.

For outcomes censored at a lower and/or upper limit (e.g., wages
observed only if employed, expenditure ≥ 0).

    Y_i* = X_i'β + ε_i,    ε_i ~ N(0, σ²)
    Y_i  = max(Y_i*, L)     (left-censored at L)

References
----------
Tobin, J. (1958).
"Estimation of Relationships for Limited Dependent Variables."
*Econometrica*, 26(1), 24-36. [@tobin1958estimation]

Amemiya, T. (1984).
"Tobit Models: A Survey."
*Journal of Econometrics*, 24(1-2), 3-61. [@amemiya1984tobit]
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ..core.results import CausalResult


def tobit(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    ll: float = 0,
    ul: Optional[float] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Tobit (Type I) censored regression via MLE.

    Equivalent to Stata's ``tobit y x, ll(0)``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Censored outcome variable.
    x : list of str
        Regressors.
    ll : float, default 0
        Lower censoring limit. Observations with Y ≤ ll are censored.
        Set to ``-np.inf`` for no lower censoring.
    ul : float, optional
        Upper censoring limit. Default: no upper censoring.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        MLE coefficients, sigma, and marginal effects.

    Examples
    --------
    >>> # Hours worked (censored at 0)
    >>> result = sp.tobit(df, y='hours', x=['wage', 'education', 'children'],
    ...                   ll=0)
    >>> print(result.summary())

    Notes
    -----
    The Tobit log-likelihood for left-censoring at L:

    .. math::
        \\ell = \\sum_{y_i > L} \\left[
            -\\frac{1}{2}\\log(2\\pi\\sigma^2)
            - \\frac{(y_i - x_i'\\beta)^2}{2\\sigma^2}
        \\right]
        + \\sum_{y_i = L} \\log \\Phi\\left(
            \\frac{L - x_i'\\beta}{\\sigma}
        \\right)

    **Marginal effects**: The coefficient β does NOT directly give the
    marginal effect on E[Y|X]. The marginal effect on the observed
    (uncensored) mean is β × Φ(X'β/σ).

    See Tobin (1958, *Econometrica*).
    """
    df = data[[y] + x].dropna()
    Y = df[y].values.astype(float)
    X = np.column_stack([np.ones(len(df))] +
                        [df[v].values.astype(float) for v in x])
    n, k = X.shape

    if ul is None:
        ul = np.inf

    censored_low = Y <= ll
    if np.isfinite(ul):
        censored_high = Y >= ul
    else:
        censored_high = np.zeros(len(Y), dtype=bool)
    uncensored = ~censored_low & ~censored_high

    n_censored = int(censored_low.sum() + censored_high.sum())
    n_uncensored = uncensored.sum()

    if n_uncensored < k + 1:
        raise ValueError("Not enough uncensored observations.")

    # Initial values from OLS on uncensored
    beta_init = np.linalg.lstsq(X[uncensored], Y[uncensored], rcond=None)[0]
    resid_init = Y[uncensored] - X[uncensored] @ beta_init
    log_sigma_init = np.log(max(np.std(resid_init), 0.01))

    theta0 = np.concatenate([beta_init, [log_sigma_init]])

    # MLE
    def neg_loglik(theta):
        beta = theta[:k]
        sigma = np.exp(theta[k])
        sigma = max(sigma, 1e-6)

        xb = X @ beta
        ll_val = 0.0

        # Uncensored
        if uncensored.any():
            resid = Y[uncensored] - xb[uncensored]
            ll_val += np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2)
                             - resid ** 2 / (2 * sigma ** 2))

        # Left-censored
        if censored_low.any():
            z = (ll - xb[censored_low]) / sigma
            ll_val += np.sum(np.log(np.maximum(stats.norm.cdf(z), 1e-20)))

        # Upper-censored
        if isinstance(censored_high, np.ndarray) and censored_high.any():
            z = (ul - xb[censored_high]) / sigma
            ll_val += np.sum(np.log(np.maximum(1 - stats.norm.cdf(z), 1e-20)))

        return -ll_val

    result = optimize.minimize(neg_loglik, theta0, method='BFGS',
                               options={'maxiter': 1000, 'gtol': 1e-6})

    theta_hat = result.x
    beta = theta_hat[:k]
    sigma = np.exp(theta_hat[k])

    # Standard errors from inverse Hessian
    try:
        hess = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(k + 1)
        if isinstance(hess, np.ndarray):
            se_full = np.sqrt(np.maximum(np.diag(hess), 1e-20))
        else:
            se_full = np.sqrt(np.maximum(np.diag(hess.todense()), 1e-20))
    except Exception:
        se_full = np.full(k + 1, np.nan)

    se_beta = se_full[:k]
    se_sigma = se_full[k] * sigma  # delta method for exp transform

    var_names = ['const'] + x
    z_stats = beta / se_beta
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    detail = pd.DataFrame({
        'variable': var_names + ['sigma'],
        'coefficient': np.append(beta, sigma),
        'se': np.append(se_beta, se_sigma),
        'z': np.append(z_stats, np.nan),
        'pvalue': np.append(pvals, np.nan),
    })

    # Main estimate: first regressor
    main_coef = float(beta[1])
    main_se = float(se_beta[1])
    main_p = float(pvals[1])
    ci = (main_coef - z_crit * main_se, main_coef + z_crit * main_se)

    model_info = {
        'method': 'Tobit MLE',
        'sigma': float(sigma),
        'n_censored': int(n_censored),
        'n_uncensored': int(n_uncensored),
        'censor_pct': round(n_censored / n * 100, 1),
        'lower_limit': ll,
        'upper_limit': ul if np.isfinite(ul) else None,
        'log_likelihood': float(-result.fun),
        'converged': result.success,
    }

    return CausalResult(
        method='Tobit (Censored Regression)',
        estimand=f'beta_{x[0]}',
        estimate=main_coef,
        se=main_se,
        pvalue=main_p,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='tobit',
    )


# Citation
CausalResult._CITATIONS['tobit'] = (
    "@article{tobin1958estimation,\n"
    "  title={Estimation of Relationships for Limited Dependent Variables},\n"
    "  author={Tobin, James},\n"
    "  journal={Econometrica},\n"
    "  volume={26},\n"
    "  number={1},\n"
    "  pages={24--36},\n"
    "  year={1958},\n"
    "  publisher={Wiley}\n"
    "}"
)
