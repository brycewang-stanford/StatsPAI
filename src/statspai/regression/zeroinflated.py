"""
Zero-Inflated and Hurdle Count Models.

Implements Zero-Inflated Poisson (ZIP), Zero-Inflated Negative Binomial
(ZINB), and Hurdle models for count data with excess zeros.

Zero-Inflated models assume two data-generating processes:
  1. A binary process (logit) that generates structural zeros with
     probability π_i = Λ(z_i'γ).
  2. A count process (Poisson or NB2) that generates counts (including
     sampling zeros) with mean μ_i = exp(x_i'β).

Hurdle models differ: zeros come from ONE process only (the logit gate),
and positive counts come from a truncated-at-zero count distribution.

References
----------
Lambert, D. (1992).
"Zero-Inflated Poisson Regression, with an Application to Defects in
Manufacturing." *Technometrics*, 34(1), 1-14. [@lambert1992zero]

Vuong, Q.H. (1989).
"Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses."
*Econometrica*, 57(2), 307-333. [@vuong1989likelihood]

Cameron, A.C. and P.K. Trivedi (2013).
*Regression Analysis of Count Data*, 2nd ed. Cambridge University Press. [@cameron2013regression]

Mullahy, J. (1986).
"Specification and Testing of Some Modified Count Data Models."
*Journal of Econometrics*, 33(3), 341-365. [@mullahy1986specification]
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats, optimize, special

from ..core.results import EconometricResults
from ..core.utils import parse_formula, create_design_matrices, prepare_data


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _logit(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, numerically stable."""
    return special.expit(z)


def _log_poisson_pmf(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Log P(Y=y | mu) for Poisson, vectorized."""
    return y * np.log(np.maximum(mu, 1e-20)) - mu - special.gammaln(y + 1)


def _log_nb2_pmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """
    Log P(Y=y | mu, alpha) for NB2 parameterization.

    Var(Y) = mu + alpha * mu^2.  Let r = 1/alpha.
    P(Y=y) = Gamma(y+r)/(Gamma(r)*y!) * (r/(r+mu))^r * (mu/(r+mu))^y
    """
    r = 1.0 / max(alpha, 1e-10)
    log_p = (
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(np.maximum(mu, 1e-20) / (r + mu))
    )
    return log_p


def _build_matrices(
    data: pd.DataFrame,
    formula: Optional[str],
    y: Optional[str],
    x: Optional[List[str]],
    inflate: Optional[List[str]],
):
    """
    Parse inputs and return (Y, X_count, X_inflate, count_names, inflate_names, dep_var).

    X matrices include a constant column.
    """
    if formula is not None:
        parsed = parse_formula(formula)
        dep_var = parsed['dependent']
        x_vars = parsed['exogenous']
    else:
        if y is None or x is None:
            raise ValueError("Provide either `formula` or both `y` and `x`.")
        dep_var = y
        x_vars = list(x)

    if inflate is None:
        inflate_vars = list(x_vars)
    else:
        inflate_vars = list(inflate)

    all_vars = list(set([dep_var] + x_vars + inflate_vars))
    df = data[all_vars].dropna()

    Y = df[dep_var].values.astype(float)
    if np.any(Y < 0) or not np.all(Y == Y.astype(int)):
        raise ValueError("Dependent variable must contain non-negative integers.")
    Y = Y.astype(int)

    # Count equation design matrix (with constant)
    X_count = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in x_vars]
    )
    count_names = ['const'] + x_vars

    # Inflate equation design matrix (with constant)
    X_inflate = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in inflate_vars]
    )
    inflate_names = ['inflate_const'] + [f'inflate_{v}' for v in inflate_vars]

    return Y, X_count, X_inflate, count_names, inflate_names, dep_var, df


def _robust_se(score_obs: np.ndarray, hessian_inv: np.ndarray) -> np.ndarray:
    """HC0 (White) robust standard errors via sandwich formula."""
    meat = score_obs.T @ score_obs
    sandwich = hessian_inv @ meat @ hessian_inv
    return np.sqrt(np.maximum(np.diag(sandwich), 1e-20))


def _cluster_se(
    score_obs: np.ndarray, hessian_inv: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    """Clustered standard errors (Liang-Zeger)."""
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    n = score_obs.shape[0]
    k = score_obs.shape[1]

    # Sum scores within clusters
    cluster_scores = np.zeros((G, k))
    for i, c in enumerate(unique_clusters):
        mask = clusters == c
        cluster_scores[i] = score_obs[mask].sum(axis=0)

    meat = cluster_scores.T @ cluster_scores
    # Small-sample correction: G/(G-1) * n/(n-k)
    correction = (G / max(G - 1, 1)) * (n / max(n - k, 1))
    sandwich = correction * (hessian_inv @ meat @ hessian_inv)
    return np.sqrt(np.maximum(np.diag(sandwich), 1e-20))


def _numerical_hessian(func, x0, eps=1e-5):
    """Compute numerical Hessian of func at x0."""
    n = len(x0)
    H = np.zeros((n, n))
    f0 = func(x0)
    for i in range(n):
        for j in range(i, n):
            x_pp = x0.copy()
            x_pm = x0.copy()
            x_mp = x0.copy()
            x_mm = x0.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm[i] -= eps
            x_mm[j] -= eps
            H[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return H


def _numerical_score(neg_loglik, theta, n_obs, eps=1e-5):
    """Compute per-observation numerical score (gradient of log-lik contribution)."""
    # This is an approximation — compute gradient of total neg_loglik
    k = len(theta)
    grad = np.zeros(k)
    for j in range(k):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps
        grad[j] = (neg_loglik(theta_p) - neg_loglik(theta_m)) / (2 * eps)
    return grad


def _vuong_test(
    loglik_model1: np.ndarray, loglik_model2: np.ndarray
) -> Dict[str, float]:
    """
    Vuong (1989) non-nested likelihood ratio test.

    H0: models are equivalent.
    V > 1.96 favours model 1; V < -1.96 favours model 2.

    Parameters
    ----------
    loglik_model1, loglik_model2 : array of per-obs log-likelihoods.

    Returns
    -------
    dict with vuong_stat, vuong_p.
    """
    m = loglik_model1 - loglik_model2
    n = len(m)
    m_bar = m.mean()
    s_m = m.std(ddof=1)
    if s_m < 1e-15:
        return {'vuong_stat': 0.0, 'vuong_p': 1.0}
    V = np.sqrt(n) * m_bar / s_m
    p = 2 * (1 - stats.norm.cdf(np.abs(V)))
    return {'vuong_stat': float(V), 'vuong_p': float(p)}


# ===================================================================
# ZIP — Zero-Inflated Poisson
# ===================================================================

def zip_model(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    inflate: list = None,
    robust: str = "nonrobust",
    cluster: str = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Zero-Inflated Poisson (ZIP) regression via MLE.

    Two-part model:
      - Inflate equation: logit model for P(structural zero) = Λ(z'γ)
      - Count equation:  Poisson model with mean μ = exp(x'β)

    Equivalent to Stata's ``zip y x, inflate(z)``.

    Parameters
    ----------
    formula : str, optional
        Patsy-style formula for the count equation, e.g. "y ~ x1 + x2".
    data : pd.DataFrame
        Dataset.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Count-equation regressors (alternative to formula).
    inflate : list of str, optional
        Inflation-equation regressors. Default: same as count regressors.
    robust : str, default "nonrobust"
        "nonrobust", "HC0", "HC1", etc.
    cluster : str, optional
        Cluster variable name for clustered standard errors.
    maxiter : int, default 200
        Maximum iterations for optimizer.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        Coefficients for both equations, Vuong test, diagnostics.

    Examples
    --------
    >>> result = sp.zip_model(data=df, y='doctor_visits', x=['age', 'income'],
    ...                       inflate=['age', 'chronic'])
    >>> print(result.summary())

    Notes
    -----
    Log-likelihood for ZIP:

    .. math::
        y_i = 0: \\log[\\pi_i + (1-\\pi_i) e^{-\\mu_i}]
        y_i > 0: \\log(1-\\pi_i) + y_i \\log\\mu_i - \\mu_i - \\log(y_i!)

    where π_i = Λ(z_i'γ) and μ_i = exp(x_i'β).

    See Lambert (1992, *Technometrics*).
    """
    Y, X_count, X_inflate, count_names, inflate_names, dep_var, df = _build_matrices(
        data, formula, y, x, inflate
    )
    n = len(Y)
    k_count = X_count.shape[1]
    k_inflate = X_inflate.shape[1]
    k_total = k_count + k_inflate

    # --- Negative log-likelihood ---
    def neg_loglik(theta):
        beta = theta[:k_count]
        gamma = theta[k_count:]

        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)

        # y == 0
        zero_mask = Y == 0
        ll = np.zeros(n)
        ll[zero_mask] = np.log(
            np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-mu[zero_mask]), 1e-20)
        )
        # y > 0
        pos_mask = ~zero_mask
        ll[pos_mask] = (
            np.log(np.maximum(1 - pi[pos_mask], 1e-20))
            + _log_poisson_pmf(Y[pos_mask], mu[pos_mask])
        )
        return -ll.sum()

    def neg_loglik_obs(theta):
        """Per-observation negative log-likelihood (for robust SE)."""
        beta = theta[:k_count]
        gamma = theta[k_count:]
        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)
        zero_mask = Y == 0
        ll = np.zeros(n)
        ll[zero_mask] = np.log(
            np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-mu[zero_mask]), 1e-20)
        )
        pos_mask = ~zero_mask
        ll[pos_mask] = (
            np.log(np.maximum(1 - pi[pos_mask], 1e-20))
            + _log_poisson_pmf(Y[pos_mask], mu[pos_mask])
        )
        return ll

    # --- Initial values ---
    # Count: log-linear OLS on log(y+1)
    beta0 = np.linalg.lstsq(X_count, np.log(Y + 1), rcond=None)[0]
    gamma0 = np.zeros(k_inflate)

    theta0 = np.concatenate([beta0, gamma0])

    # --- Optimise ---
    result = optimize.minimize(
        neg_loglik, theta0, method='BFGS',
        options={'maxiter': maxiter, 'gtol': tol},
    )

    theta_hat = result.x
    beta_hat = theta_hat[:k_count]
    gamma_hat = theta_hat[k_count:]

    ll_zip = float(-result.fun)

    # --- Standard errors ---
    H = _numerical_hessian(neg_loglik, theta_hat)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    if cluster is not None:
        clusters = df[cluster].values if cluster in df.columns else data.loc[df.index, cluster].values
        score_obs = _compute_zip_score_obs(theta_hat, Y, X_count, X_inflate, k_count, n)
        se = _cluster_se(score_obs, H_inv, clusters)
    elif robust != "nonrobust":
        score_obs = _compute_zip_score_obs(theta_hat, Y, X_count, X_inflate, k_count, n)
        se = _robust_se(score_obs, H_inv)
    else:
        se = np.sqrt(np.maximum(np.diag(H_inv), 1e-20))

    # --- Vuong test: ZIP vs plain Poisson ---
    mu_hat = np.exp(np.clip(X_count @ beta_hat, -20, 20))
    pi_hat = _logit(X_inflate @ gamma_hat)

    ll_zip_obs = neg_loglik_obs(theta_hat)
    ll_poisson_obs = _log_poisson_pmf(Y, mu_hat)
    vuong = _vuong_test(ll_zip_obs, ll_poisson_obs)

    # --- Predicted values ---
    pred_structural_zero = pi_hat
    pred_count = mu_hat
    pred_overall = (1 - pi_hat) * mu_hat

    # --- Assemble results ---
    all_names = count_names + inflate_names
    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    model_info = {
        'model_type': 'zip',
        'method': 'Zero-Inflated Poisson (MLE)',
        'dependent_var': dep_var,
        'll': ll_zip,
        'aic': -2 * ll_zip + 2 * k_total,
        'bic': -2 * ll_zip + np.log(n) * k_total,
        'vuong_stat': vuong['vuong_stat'],
        'vuong_p': vuong['vuong_p'],
        'converged': result.success,
        'robust': robust if cluster is None else f'cluster({cluster})',
        'n_zeros': int((Y == 0).sum()),
        'pct_zeros': float((Y == 0).mean() * 100),
    }

    data_info = {
        'n_obs': n,
        'dependent_var': dep_var,
        'df_resid': n - k_total,
        'k_count': k_count,
        'k_inflate': k_inflate,
        'count_names': count_names,
        'inflate_names': inflate_names,
    }

    diagnostics = {
        'predicted_structural_zero': pred_structural_zero,
        'predicted_count': pred_count,
        'predicted_overall': pred_overall,
        'vuong_stat': vuong['vuong_stat'],
        'vuong_p': vuong['vuong_p'],
        'll': ll_zip,
        'aic': model_info['aic'],
        'bic': model_info['bic'],
    }

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _compute_zip_score_obs(theta, Y, X_count, X_inflate, k_count, n):
    """Compute per-observation score for ZIP (numerical)."""
    k_total = len(theta)
    eps = 1e-5
    score = np.zeros((n, k_total))

    beta = theta[:k_count]
    gamma = theta[k_count:]
    mu = np.exp(np.clip(X_count @ beta, -20, 20))
    pi = _logit(X_inflate @ gamma)

    zero_mask = Y == 0
    pos_mask = ~zero_mask

    # Per-obs log-likelihood
    ll_obs = np.zeros(n)
    ll_obs[zero_mask] = np.log(
        np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-mu[zero_mask]), 1e-20)
    )
    ll_obs[pos_mask] = (
        np.log(np.maximum(1 - pi[pos_mask], 1e-20))
        + _log_poisson_pmf(Y[pos_mask], mu[pos_mask])
    )

    for j in range(k_total):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps

        beta_p = theta_p[:k_count]
        gamma_p = theta_p[k_count:]
        mu_p = np.exp(np.clip(X_count @ beta_p, -20, 20))
        pi_p = _logit(X_inflate @ gamma_p)
        ll_p = np.zeros(n)
        ll_p[zero_mask] = np.log(
            np.maximum(pi_p[zero_mask] + (1 - pi_p[zero_mask]) * np.exp(-mu_p[zero_mask]), 1e-20)
        )
        ll_p[pos_mask] = (
            np.log(np.maximum(1 - pi_p[pos_mask], 1e-20))
            + _log_poisson_pmf(Y[pos_mask], mu_p[pos_mask])
        )

        beta_m = theta_m[:k_count]
        gamma_m = theta_m[k_count:]
        mu_m = np.exp(np.clip(X_count @ beta_m, -20, 20))
        pi_m = _logit(X_inflate @ gamma_m)
        ll_m = np.zeros(n)
        ll_m[zero_mask] = np.log(
            np.maximum(pi_m[zero_mask] + (1 - pi_m[zero_mask]) * np.exp(-mu_m[zero_mask]), 1e-20)
        )
        ll_m[pos_mask] = (
            np.log(np.maximum(1 - pi_m[pos_mask], 1e-20))
            + _log_poisson_pmf(Y[pos_mask], mu_m[pos_mask])
        )

        score[:, j] = (ll_p - ll_m) / (2 * eps)

    return score


# ===================================================================
# ZINB — Zero-Inflated Negative Binomial
# ===================================================================

def zinb(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    inflate: list = None,
    robust: str = "nonrobust",
    cluster: str = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Zero-Inflated Negative Binomial (ZINB) regression via MLE.

    Two-part model:
      - Inflate equation: logit for P(structural zero) = Λ(z'γ)
      - Count equation:  NB2 with mean μ = exp(x'β), Var = μ + α·μ²

    Equivalent to Stata's ``zinb y x, inflate(z)``.

    Parameters
    ----------
    formula : str, optional
        Patsy-style formula for the count equation.
    data : pd.DataFrame
        Dataset.
    y : str, optional
        Dependent variable name.
    x : list of str, optional
        Count-equation regressors.
    inflate : list of str, optional
        Inflation-equation regressors. Default: same as count regressors.
    robust : str, default "nonrobust"
        Standard error type.
    cluster : str, optional
        Cluster variable name.
    maxiter : int, default 200
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        Coefficients for count, inflate, and dispersion parameter.

    Examples
    --------
    >>> result = sp.zinb(data=df, y='doctor_visits', x=['age', 'income'],
    ...                  inflate=['age', 'chronic'])
    >>> print(result.summary())

    Notes
    -----
    The NB2 parameterization uses dispersion parameter α so that
    Var(Y|μ) = μ + α·μ². When α → 0 the model collapses to ZIP.

    See Cameron & Trivedi (2013, Ch. 4).
    """
    Y, X_count, X_inflate, count_names, inflate_names, dep_var, df = _build_matrices(
        data, formula, y, x, inflate
    )
    n = len(Y)
    k_count = X_count.shape[1]
    k_inflate = X_inflate.shape[1]
    # theta = [beta, gamma, log_alpha]
    k_total = k_count + k_inflate + 1

    def neg_loglik(theta):
        beta = theta[:k_count]
        gamma = theta[k_count:k_count + k_inflate]
        log_alpha = theta[-1]
        disp = np.exp(np.clip(log_alpha, -10, 10))

        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)

        zero_mask = Y == 0
        ll = np.zeros(n)

        # NB2 pmf at y=0
        nb_zero = _log_nb2_pmf(np.zeros_like(mu[zero_mask]), mu[zero_mask], disp)
        ll[zero_mask] = np.log(
            np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(nb_zero), 1e-20)
        )

        pos_mask = ~zero_mask
        ll[pos_mask] = (
            np.log(np.maximum(1 - pi[pos_mask], 1e-20))
            + _log_nb2_pmf(Y[pos_mask], mu[pos_mask], disp)
        )
        return -ll.sum()

    def neg_loglik_obs(theta):
        beta = theta[:k_count]
        gamma = theta[k_count:k_count + k_inflate]
        log_alpha = theta[-1]
        disp = np.exp(np.clip(log_alpha, -10, 10))
        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)
        zero_mask = Y == 0
        ll = np.zeros(n)
        nb_zero = _log_nb2_pmf(np.zeros_like(mu[zero_mask]), mu[zero_mask], disp)
        ll[zero_mask] = np.log(
            np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(nb_zero), 1e-20)
        )
        pos_mask = ~zero_mask
        ll[pos_mask] = (
            np.log(np.maximum(1 - pi[pos_mask], 1e-20))
            + _log_nb2_pmf(Y[pos_mask], mu[pos_mask], disp)
        )
        return ll

    # Initial values
    beta0 = np.linalg.lstsq(X_count, np.log(Y + 1), rcond=None)[0]
    gamma0 = np.zeros(k_inflate)
    log_alpha0 = np.array([0.0])  # alpha = 1 initial guess
    theta0 = np.concatenate([beta0, gamma0, log_alpha0])

    result = optimize.minimize(
        neg_loglik, theta0, method='BFGS',
        options={'maxiter': maxiter, 'gtol': tol},
    )

    theta_hat = result.x
    beta_hat = theta_hat[:k_count]
    gamma_hat = theta_hat[k_count:k_count + k_inflate]
    alpha_hat = np.exp(theta_hat[-1])
    ll_zinb = float(-result.fun)

    # Standard errors
    H = _numerical_hessian(neg_loglik, theta_hat)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    if cluster is not None:
        clusters = df[cluster].values if cluster in df.columns else data.loc[df.index, cluster].values
        score_obs = _compute_zi_score_obs(neg_loglik_obs, theta_hat, Y, X_count, X_inflate, k_count, k_inflate, n, nb=True)
        se = _cluster_se(score_obs, H_inv, clusters)
    elif robust != "nonrobust":
        score_obs = _compute_zi_score_obs(neg_loglik_obs, theta_hat, Y, X_count, X_inflate, k_count, k_inflate, n, nb=True)
        se = _robust_se(score_obs, H_inv)
    else:
        se = np.sqrt(np.maximum(np.diag(H_inv), 1e-20))

    # Vuong test: ZINB vs plain NB
    mu_hat = np.exp(np.clip(X_count @ beta_hat, -20, 20))
    ll_zinb_obs = neg_loglik_obs(theta_hat)
    ll_nb_obs = _log_nb2_pmf(Y, mu_hat, alpha_hat)
    vuong = _vuong_test(ll_zinb_obs, ll_nb_obs)

    # Predicted values
    pi_hat = _logit(X_inflate @ gamma_hat)
    pred_structural_zero = pi_hat
    pred_count = mu_hat
    pred_overall = (1 - pi_hat) * mu_hat

    # Assemble
    all_names = count_names + inflate_names + ['ln_alpha']
    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    model_info = {
        'model_type': 'zinb',
        'method': 'Zero-Inflated Negative Binomial (MLE)',
        'dependent_var': dep_var,
        'll': ll_zinb,
        'aic': -2 * ll_zinb + 2 * k_total,
        'bic': -2 * ll_zinb + np.log(n) * k_total,
        'alpha_dispersion': float(alpha_hat),
        'vuong_stat': vuong['vuong_stat'],
        'vuong_p': vuong['vuong_p'],
        'converged': result.success,
        'robust': robust if cluster is None else f'cluster({cluster})',
        'n_zeros': int((Y == 0).sum()),
        'pct_zeros': float((Y == 0).mean() * 100),
    }

    data_info = {
        'n_obs': n,
        'dependent_var': dep_var,
        'df_resid': n - k_total,
        'k_count': k_count,
        'k_inflate': k_inflate,
        'count_names': count_names,
        'inflate_names': inflate_names,
    }

    diagnostics = {
        'predicted_structural_zero': pred_structural_zero,
        'predicted_count': pred_count,
        'predicted_overall': pred_overall,
        'alpha_dispersion': float(alpha_hat),
        'vuong_stat': vuong['vuong_stat'],
        'vuong_p': vuong['vuong_p'],
        'll': ll_zinb,
        'aic': model_info['aic'],
        'bic': model_info['bic'],
    }

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _compute_zi_score_obs(neg_loglik_obs_fn, theta, Y, X_count, X_inflate, k_count, k_inflate, n, nb=False):
    """Numerical per-observation score for ZI models."""
    k_total = len(theta)
    eps = 1e-5
    score = np.zeros((n, k_total))
    ll_base = neg_loglik_obs_fn(theta)

    for j in range(k_total):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps
        ll_p = neg_loglik_obs_fn(theta_p)
        ll_m = neg_loglik_obs_fn(theta_m)
        score[:, j] = (ll_p - ll_m) / (2 * eps)

    return score


# ===================================================================
# Hurdle Model
# ===================================================================

def hurdle(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    count_model: str = "poisson",
    robust: str = "nonrobust",
    cluster: str = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Hurdle (two-part) model for count data.

    Part 1 (binary): logit model for P(Y > 0).
    Part 2 (count):  truncated-at-zero Poisson or Negative Binomial for
                     the distribution of Y | Y > 0.

    Unlike zero-inflated models, ALL zeros come from the binary process.

    Equivalent to R's ``pscl::hurdle()``.

    Parameters
    ----------
    formula : str, optional
        Patsy-style formula.
    data : pd.DataFrame
        Dataset.
    y : str, optional
        Dependent variable name.
    x : list of str, optional
        Regressors (used for both hurdle and count parts).
    count_model : str, default "poisson"
        Count distribution: "poisson" or "negbin".
    robust : str, default "nonrobust"
    cluster : str, optional
    maxiter : int, default 200
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> result = sp.hurdle(data=df, y='doctor_visits', x=['age', 'income'],
    ...                    count_model='negbin')
    >>> print(result.summary())

    Notes
    -----
    The hurdle log-likelihood decomposes as:

    .. math::
        \\ell = \\sum_{y_i=0} \\log(1-p_i) + \\sum_{y_i>0} [\\log p_i
        + \\log f(y_i|\\mu_i) - \\log(1 - f(0|\\mu_i))]

    where p_i = Λ(x_i'δ) is the hurdle probability.

    See Mullahy (1986, *Journal of Econometrics*).
    """
    if formula is not None:
        parsed = parse_formula(formula)
        dep_var = parsed['dependent']
        x_vars = parsed['exogenous']
    else:
        if y is None or x is None:
            raise ValueError("Provide either `formula` or both `y` and `x`.")
        dep_var = y
        x_vars = list(x)

    all_vars = list(set([dep_var] + x_vars))
    df = data[all_vars].dropna()

    Y = df[dep_var].values.astype(float)
    if np.any(Y < 0) or not np.all(Y == Y.astype(int)):
        raise ValueError("Dependent variable must contain non-negative integers.")
    Y = Y.astype(int)

    X = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in x_vars]
    )
    var_names = ['const'] + x_vars
    hurdle_names = ['hurdle_const'] + [f'hurdle_{v}' for v in x_vars]
    n, k = X.shape

    zero_mask = Y == 0
    pos_mask = ~zero_mask
    Y_pos = Y[pos_mask]
    X_pos = X[pos_mask]

    use_negbin = count_model.lower() in ('negbin', 'nb', 'nbreg')
    # k_hurdle params + k_count params (+ 1 if negbin for log_alpha)
    k_hurdle = k
    k_count = k
    k_total = k_hurdle + k_count + (1 if use_negbin else 0)

    def neg_loglik(theta):
        delta = theta[:k_hurdle]
        beta = theta[k_hurdle:k_hurdle + k_count]

        p = _logit(X @ delta)  # P(Y > 0)
        mu = np.exp(np.clip(X @ beta, -20, 20))

        # Part 1: binary
        ll_binary = np.zeros(n)
        ll_binary[zero_mask] = np.log(np.maximum(1 - p[zero_mask], 1e-20))
        ll_binary[pos_mask] = np.log(np.maximum(p[pos_mask], 1e-20))

        # Part 2: truncated count for positive obs
        if use_negbin:
            disp = np.exp(np.clip(theta[-1], -10, 10))
            log_f = _log_nb2_pmf(Y_pos, mu[pos_mask], disp)
            log_f0 = _log_nb2_pmf(np.zeros(pos_mask.sum()), mu[pos_mask], disp)
        else:
            log_f = _log_poisson_pmf(Y_pos, mu[pos_mask])
            log_f0 = -mu[pos_mask]  # log P(Y=0|mu) for Poisson

        # Truncated: f(y) / (1 - f(0))
        ll_count = log_f - np.log(np.maximum(1 - np.exp(log_f0), 1e-20))

        total = ll_binary.sum() + ll_count.sum()
        return -total

    def neg_loglik_obs(theta):
        delta = theta[:k_hurdle]
        beta = theta[k_hurdle:k_hurdle + k_count]
        p = _logit(X @ delta)
        mu = np.exp(np.clip(X @ beta, -20, 20))

        ll = np.zeros(n)
        ll[zero_mask] = np.log(np.maximum(1 - p[zero_mask], 1e-20))

        if use_negbin:
            disp = np.exp(np.clip(theta[-1], -10, 10))
            log_f = _log_nb2_pmf(Y[pos_mask], mu[pos_mask], disp)
            log_f0 = _log_nb2_pmf(np.zeros(pos_mask.sum()), mu[pos_mask], disp)
        else:
            log_f = _log_poisson_pmf(Y[pos_mask], mu[pos_mask])
            log_f0 = -mu[pos_mask]

        ll[pos_mask] = (
            np.log(np.maximum(p[pos_mask], 1e-20))
            + log_f
            - np.log(np.maximum(1 - np.exp(log_f0), 1e-20))
        )
        return ll

    # Initial values
    delta0 = np.zeros(k_hurdle)
    beta0 = np.linalg.lstsq(X_pos, np.log(Y_pos), rcond=None)[0]
    if use_negbin:
        theta0 = np.concatenate([delta0, beta0, [0.0]])
    else:
        theta0 = np.concatenate([delta0, beta0])

    result = optimize.minimize(
        neg_loglik, theta0, method='BFGS',
        options={'maxiter': maxiter, 'gtol': tol},
    )

    theta_hat = result.x
    delta_hat = theta_hat[:k_hurdle]
    beta_hat = theta_hat[k_hurdle:k_hurdle + k_count]
    ll_hurdle = float(-result.fun)

    # Standard errors
    H = _numerical_hessian(neg_loglik, theta_hat)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    if cluster is not None:
        clusters = df[cluster].values if cluster in df.columns else data.loc[df.index, cluster].values
        score_obs = _compute_hurdle_score_obs(neg_loglik_obs, theta_hat, n)
        se = _cluster_se(score_obs, H_inv, clusters)
    elif robust != "nonrobust":
        score_obs = _compute_hurdle_score_obs(neg_loglik_obs, theta_hat, n)
        se = _robust_se(score_obs, H_inv)
    else:
        se = np.sqrt(np.maximum(np.diag(H_inv), 1e-20))

    # Predicted values
    p_hat = _logit(X @ delta_hat)
    mu_hat = np.exp(np.clip(X @ beta_hat, -20, 20))
    if use_negbin:
        alpha_hat = np.exp(theta_hat[-1])
        f0 = np.exp(_log_nb2_pmf(np.zeros(n), mu_hat, alpha_hat))
    else:
        alpha_hat = None
        f0 = np.exp(-mu_hat)

    pred_hurdle_prob = p_hat  # P(Y > 0)
    # E[Y] = P(Y>0) * E[Y | Y>0] = p * mu / (1 - f(0))
    pred_overall = p_hat * mu_hat / np.maximum(1 - f0, 1e-20)

    # Assemble names
    all_names = hurdle_names + count_names_from_vars(var_names)
    if use_negbin:
        all_names.append('ln_alpha')

    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    model_info = {
        'model_type': 'hurdle',
        'method': f'Hurdle ({count_model.title()}, MLE)',
        'dependent_var': dep_var,
        'count_dist': count_model,
        'll': ll_hurdle,
        'aic': -2 * ll_hurdle + 2 * k_total,
        'bic': -2 * ll_hurdle + np.log(n) * k_total,
        'converged': result.success,
        'robust': robust if cluster is None else f'cluster({cluster})',
        'n_zeros': int(zero_mask.sum()),
        'pct_zeros': float(zero_mask.mean() * 100),
    }
    if use_negbin:
        model_info['alpha_dispersion'] = float(alpha_hat)

    data_info = {
        'n_obs': n,
        'dependent_var': dep_var,
        'df_resid': n - k_total,
        'k_hurdle': k_hurdle,
        'k_count': k_count,
        'hurdle_names': hurdle_names,
        'count_names': count_names_from_vars(var_names),
    }

    diagnostics = {
        'predicted_hurdle_prob': pred_hurdle_prob,
        'predicted_count_mean': mu_hat,
        'predicted_overall': pred_overall,
        'll': ll_hurdle,
        'aic': model_info['aic'],
        'bic': model_info['bic'],
    }
    if use_negbin:
        diagnostics['alpha_dispersion'] = float(alpha_hat)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def count_names_from_vars(var_names: List[str]) -> List[str]:
    """Prefix count-equation variable names."""
    return ['count_' + v for v in var_names]


def _compute_hurdle_score_obs(neg_loglik_obs_fn, theta, n):
    """Numerical per-observation score for hurdle models."""
    k_total = len(theta)
    eps = 1e-5
    score = np.zeros((n, k_total))

    for j in range(k_total):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps
        ll_p = neg_loglik_obs_fn(theta_p)
        ll_m = neg_loglik_obs_fn(theta_m)
        score[:, j] = (ll_p - ll_m) / (2 * eps)

    return score
