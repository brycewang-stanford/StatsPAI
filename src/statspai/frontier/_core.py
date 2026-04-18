"""
Shared primitives for stochastic frontier analysis (SFA).

Provides log-density kernels, Jondrow-style posterior moments of u|eps,
Battese-Coelli exp(-u) technical efficiency, numerical Hessian / design-matrix
helpers, and sign-convention utilities used by both cross-sectional
(``sfa.py``) and panel (``panel.py``) estimators.

Sign convention
---------------
Composed error: ``y = x'beta + v + sign * u``, with ``u >= 0``,
``sign = -1`` for production (output frontier), ``sign = +1`` for cost frontier.

Distributions supported
-----------------------
* Normal / half-normal (Aigner-Lovell-Schmidt 1977)
* Normal / exponential (Meeusen-van den Broeck 1977; Greene 2008)
* Normal / truncated-normal (Stevenson 1980)

Heteroskedasticity
------------------
All kernels accept vector-valued ``sigma_u`` and ``sigma_v`` so that the
caller can plug in ``sigma_u_i = exp(w_i' gamma)`` (Caudill-Ford-Gropper
1995, Hadri 1999) or ``sigma_v_i = exp(r_i' eta)`` (Wang 2002).  The mean
parameter ``mu`` of the truncated-normal may likewise be either a scalar
or a vector ``mu_i = z_i' delta`` (Battese-Coelli 1995, Kumbhakar-Ghosh-
McGuckin 1991).

References
----------
Aigner, D., Lovell, C.A.K. & Schmidt, P. (1977). "Formulation and Estimation
    of Stochastic Frontier Production Function Models." J. Econometrics 6, 21-37.
Battese, G.E. & Coelli, T.J. (1988). "Prediction of Firm-level Technical
    Efficiencies with a Generalized Frontier Production Function and Panel
    Data." J. Econometrics 38, 387-399.
Battese, G.E. & Coelli, T.J. (1992). "Frontier Production Functions,
    Technical Efficiency and Panel Data: With Application to Paddy Farmers
    in India." J. Productivity Analysis 3, 153-169.
Battese, G.E. & Coelli, T.J. (1995). "A Model for Technical Inefficiency
    Effects in a Stochastic Frontier Production Function for Panel Data."
    Empirical Economics 20, 325-332.
Caudill, S.B., Ford, J.M. & Gropper, D.M. (1995). "Frontier Estimation and
    Firm-specific Inefficiency Measures in the Presence of Heteroscedasticity."
    J. Business & Economic Statistics 13, 105-111.
Greene, W.H. (2008). "The Econometric Approach to Efficiency Analysis." In
    Fried, H., Lovell, C.A.K. & Schmidt, S. (eds), The Measurement of
    Productive Efficiency and Productivity Growth, Oxford U.P.
Jondrow, J., Lovell, C.A.K., Materov, I.S. & Schmidt, P. (1982). "On the
    Estimation of Technical Inefficiency in the Stochastic Frontier
    Production Function Model." J. Econometrics 19, 233-238.
Meeusen, W. & van den Broeck, J. (1977). "Efficiency Estimation from
    Cobb-Douglas Production Functions With Composed Error." IER 18, 435-444.
Pitt, M.M. & Lee, L.-F. (1981). "The Measurement and Sources of Technical
    Inefficiency in the Indonesian Weaving Industry." J. Development
    Economics 9, 43-64.
Stevenson, R.E. (1980). "Likelihood Functions for Generalized Stochastic
    Frontier Estimation." J. Econometrics 13, 57-66.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


_LN_2PI = np.log(2.0 * np.pi)
_LOG_TWO = np.log(2.0)
_LOG_EPS = 1e-300          # floor for log(Phi) to avoid -inf
_PHI_FLOOR = 1e-300        # floor for Phi denominator in Jondrow


# ---------------------------------------------------------------------------
# log Phi helpers (numerically stable)
# ---------------------------------------------------------------------------


def _log_phi_cdf(x: np.ndarray) -> np.ndarray:
    """Numerically-stable log Phi(x)."""
    # scipy's logcdf is accurate in the left tail.
    return stats.norm.logcdf(x)


def _phi_over_Phi(x: np.ndarray) -> np.ndarray:
    """phi(x) / Phi(x), numerically stable via Mills ratio in the left tail.

    For very negative ``x``, ``Phi(x) -> 0`` and a naive ratio blows up.
    We use the standard Mills-ratio asymptotic expansion phi/Phi ~ -x when
    x << 0 via the identity ``phi(x)/Phi(x) = exp(log phi - log Phi)``.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    # Use direct formula in the safe region.
    safe = x > -30.0
    if np.any(safe):
        pdf = stats.norm.pdf(x[safe])
        cdf = np.clip(stats.norm.cdf(x[safe]), _PHI_FLOOR, None)
        out[safe] = pdf / cdf
    if np.any(~safe):
        # Mills-ratio tail: phi(x)/Phi(x) ~ -x(1 - 1/x^2 + 3/x^4 - ...) for x << 0.
        xt = x[~safe]
        out[~safe] = -xt * (1.0 - 1.0 / xt**2 + 3.0 / xt**4)
    return out


# ---------------------------------------------------------------------------
# Distribution-specific log likelihoods (per observation)
# ---------------------------------------------------------------------------


def loglik_halfnormal(
    eps: np.ndarray,
    sigma_v: np.ndarray,
    sigma_u: np.ndarray,
    sign: int,
) -> np.ndarray:
    """Normal / half-normal log density of ``eps = y - x'beta``.

    log f(eps) = log 2 - log sigma - 0.5 log(2 pi)
                 - 0.5 (eps/sigma)^2 + log Phi(sign * eps * lam / sigma)

    where sigma^2 = sigma_v^2 + sigma_u^2, lam = sigma_u / sigma_v.

    For production (sign = -1) large negative ``eps`` raises the density
    (left-skewed composed error).  For cost (sign = +1) mirror image.
    """
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    sigma2 = sigma_v**2 + sigma_u**2
    sigma = np.sqrt(sigma2)
    lam = sigma_u / sigma_v
    z = sign * eps * lam / sigma
    return (
        _LOG_TWO
        - np.log(sigma)
        - 0.5 * _LN_2PI
        - 0.5 * (eps / sigma) ** 2
        + _log_phi_cdf(z)
    )


def loglik_exponential(
    eps: np.ndarray,
    sigma_v: np.ndarray,
    sigma_u: np.ndarray,
    sign: int,
) -> np.ndarray:
    """Normal / exponential log density (Greene 2008, eq. 2.39).

    For production (sign = -1):
        log f(eps) = -log sigma_u + eps/sigma_u + (sigma_v/sigma_u)^2 / 2
                     + log Phi(-eps/sigma_v - sigma_v/sigma_u).
    Unified form:
        log f(eps) = -log sigma_u - sign*eps/sigma_u + (sigma_v/sigma_u)^2/2
                     + log Phi(sign*eps/sigma_v - sigma_v/sigma_u).
    """
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    arg = sign * eps / sigma_v - sigma_v / sigma_u
    return (
        -np.log(sigma_u)
        - sign * eps / sigma_u
        + 0.5 * (sigma_v / sigma_u) ** 2
        + _log_phi_cdf(arg)
    )


def loglik_truncated_normal(
    eps: np.ndarray,
    sigma_v: np.ndarray,
    sigma_u: np.ndarray,
    mu: np.ndarray,
    sign: int,
) -> np.ndarray:
    """Normal / truncated-normal log density (Stevenson 1980).

    ``u ~ N^+(mu, sigma_u^2)`` truncated at 0.  Density of eps:

        f(eps) = (1/sigma) phi((eps - sign * mu)/sigma) Phi(mu_* / sigma_*)
                 / Phi(mu / sigma_u)

    with ``mu_*`` and ``sigma_*`` as computed by :func:`jondrow_truncnormal`.
    """
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    mu = np.asarray(mu, dtype=float)

    sigma2 = sigma_v**2 + sigma_u**2
    sigma = np.sqrt(sigma2)
    # Density of eps is (1/sigma) * phi((eps - sign*mu)/sigma) * Phi(mu*/sigma*) / Phi(mu/sigma_u).
    # For production (sign=-1): (eps - (-1)*mu) = eps + mu (composed error centred at -mu).
    # For cost       (sign=+1): (eps -  1 *mu) = eps - mu (composed error centred at +mu).
    centered = (eps - sign * mu) / sigma

    mu_star = (mu * sigma_v**2 + sign * eps * sigma_u**2) / sigma2
    sigma_star = sigma_v * sigma_u / sigma

    return (
        -0.5 * _LN_2PI
        - np.log(sigma)
        - 0.5 * centered**2
        + _log_phi_cdf(mu_star / sigma_star)
        - _log_phi_cdf(mu / sigma_u)
    )


# ---------------------------------------------------------------------------
# Jondrow-style posterior moments E[u | eps] and E[exp(-u) | eps]
# ---------------------------------------------------------------------------


def jondrow_halfnormal(
    eps: np.ndarray,
    sigma_v: np.ndarray,
    sigma_u: np.ndarray,
    sign: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior E[u|eps] and Battese-Coelli E[exp(-u)|eps] for half-normal.

    Derivation: u | eps ~ N^+(mu_*, sigma_*^2) with
        mu_*   = sign * eps * sigma_u^2 / sigma^2
        sigma_* = sigma_v * sigma_u / sigma
    """
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    sigma2 = sigma_v**2 + sigma_u**2
    sigma = np.sqrt(sigma2)

    mu_star = sign * eps * sigma_u**2 / sigma2
    sigma_star = sigma_v * sigma_u / sigma

    E_u = _posterior_truncnormal_mean(mu_star, sigma_star)
    TE_bc = _battese_coelli_te(mu_star, sigma_star)
    return E_u, TE_bc


def jondrow_exponential(
    eps: np.ndarray,
    sigma_v: np.ndarray,
    sigma_u: np.ndarray,
    sign: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior E[u|eps] and E[exp(-u)|eps] for normal/exponential.

    u | eps ~ N^+(mu_tilde, sigma_v^2) with
        mu_tilde = sign * eps - sigma_v^2 / sigma_u
        sigma_*  = sigma_v.
    """
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    mu_tilde = sign * eps - sigma_v**2 / sigma_u
    sigma_star = sigma_v
    E_u = _posterior_truncnormal_mean(mu_tilde, sigma_star)
    TE_bc = _battese_coelli_te(mu_tilde, sigma_star)
    return E_u, TE_bc


def jondrow_truncnormal(
    eps: np.ndarray,
    sigma_v: np.ndarray,
    sigma_u: np.ndarray,
    mu: np.ndarray,
    sign: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior E[u|eps] and E[exp(-u)|eps] for truncated normal.

    u | eps ~ N^+(mu_*, sigma_*^2) with
        mu_*    = (mu sigma_v^2 + sign * eps sigma_u^2) / sigma^2
        sigma_* = sigma_v sigma_u / sigma.
    """
    sigma_v = np.asarray(sigma_v, dtype=float)
    sigma_u = np.asarray(sigma_u, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma2 = sigma_v**2 + sigma_u**2
    sigma = np.sqrt(sigma2)
    mu_star = (mu * sigma_v**2 + sign * eps * sigma_u**2) / sigma2
    sigma_star = sigma_v * sigma_u / sigma
    E_u = _posterior_truncnormal_mean(mu_star, sigma_star)
    TE_bc = _battese_coelli_te(mu_star, sigma_star)
    return E_u, TE_bc


def _posterior_truncnormal_mean(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """E[X] for X ~ N^+(mu, sigma^2) truncated at 0.

    The closed-form ``mu + sigma * phi(mu/sigma)/Phi(mu/sigma)`` is
    mathematically non-negative but can dip slightly below zero at
    extreme ``mu/sigma << 0`` because of Mills-ratio series truncation.
    We clamp to ``>= 0`` to preserve the theoretical support.
    """
    ratio = mu / sigma
    return np.maximum(mu + sigma * _phi_over_Phi(ratio), 0.0)


def _battese_coelli_te(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """E[exp(-X)] for X ~ N^+(mu, sigma^2) truncated at 0.

    Battese-Coelli (1988): exp(-mu + 0.5 sigma^2) * Phi(mu/sigma - sigma) / Phi(mu/sigma).
    """
    num = np.exp(-mu + 0.5 * sigma**2)
    log_numer = _log_phi_cdf(mu / sigma - sigma)
    log_denom = _log_phi_cdf(mu / sigma)
    te = num * np.exp(log_numer - log_denom)
    return np.clip(te, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Numerical Hessian / variance helpers
# ---------------------------------------------------------------------------


def numerical_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    step: float = 1e-5,
) -> np.ndarray:
    """Central-difference numerical Hessian of a scalar function.

    Uses second-order central differences; cost O(k^2) evaluations.
    """
    x = np.asarray(x, dtype=float)
    k = x.size
    f0 = f(x)
    H = np.zeros((k, k))
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = step
        fp = f(x + ei)
        fm = f(x - ei)
        H[i, i] = (fp - 2.0 * f0 + fm) / step**2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = step
            fpp = f(x + ei + ej)
            fpm = f(x + ei - ej)
            fmp = f(x - ei + ej)
            fmm = f(x - ei - ej)
            val = (fpp - fpm - fmp + fmm) / (4.0 * step**2)
            H[i, j] = H[j, i] = val
    return H


def safe_invert_hessian(H: np.ndarray) -> np.ndarray:
    """Invert a Hessian with ridge regularization on failure."""
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(H.shape[0])
        for _ in range(6):
            try:
                return np.linalg.inv(H + ridge)
            except np.linalg.LinAlgError:
                ridge *= 10.0
        return np.full_like(H, np.nan)


# ---------------------------------------------------------------------------
# Design-matrix helpers
# ---------------------------------------------------------------------------


def build_design(
    data: pd.DataFrame,
    y: str,
    x: list[str],
    add_constant: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(y_vec, X_mat, names)`` with optional leading constant."""
    y_vec = data[y].to_numpy(dtype=float)
    X_block = data[x].to_numpy(dtype=float) if x else np.empty((len(data), 0))
    if add_constant:
        const = np.ones((len(data), 1))
        X_mat = np.concatenate([const, X_block], axis=1)
        names = ["_cons"] + list(x)
    else:
        X_mat = X_block
        names = list(x)
    return y_vec, X_mat, names


def build_optional_design(
    data: pd.DataFrame,
    cols: Optional[list[str]],
    include_constant: bool = True,
    prefix: str = "",
) -> tuple[Optional[np.ndarray], list[str]]:
    """Return (matrix, names) for optional heteroskedasticity / determinant design.

    If ``cols`` is None, returns ``(None, [])``.  Otherwise constructs a
    matrix ``[1, cols...]`` (or just cols when ``include_constant=False``)
    with names ``['<prefix>_cons', '<prefix><col>', ...]``.
    """
    if cols is None:
        return None, []
    block = data[cols].to_numpy(dtype=float) if cols else np.empty((len(data), 0))
    if include_constant:
        const = np.ones((len(data), 1))
        mat = np.concatenate([const, block], axis=1)
        names = [f"{prefix}_cons"] + [f"{prefix}{c}" for c in cols]
    else:
        mat = block
        names = [f"{prefix}{c}" for c in cols]
    return mat, names


def evaluate_sigma(
    gamma: np.ndarray,
    design: Optional[np.ndarray],
    fallback_log_sigma: float,
    n: int,
) -> np.ndarray:
    """Return sigma vector (length n).

    If ``design`` is None (homoskedastic), returns ``exp(fallback_log_sigma)`` broadcast
    to length n.  Otherwise returns ``exp(design @ gamma)``.
    """
    if design is None:
        return np.full(n, np.exp(fallback_log_sigma))
    return np.exp(design @ gamma)


# ---------------------------------------------------------------------------
# LR test helpers
# ---------------------------------------------------------------------------


def lr_test_statistic(ll_unrestricted: float, ll_restricted: float) -> float:
    """2 (ll_u - ll_r)."""
    return 2.0 * (ll_unrestricted - ll_restricted)


def mixed_chi_bar_pvalue(lr_stat: float, df_boundary: int = 1) -> float:
    """Mixed chi-bar-squared p-value for one-sided LR (boundary of parameter space).

    For a single boundary parameter sigma_u = 0, the asymptotic distribution
    of ``-2 log Lambda`` is 0.5 * chi2(0) + 0.5 * chi2(1) (Kodde-Palm 1986).
    """
    if lr_stat <= 0:
        return 1.0
    if df_boundary == 1:
        return 0.5 * (1.0 - stats.chi2.cdf(lr_stat, df=1))
    # Simpler fallback: use chi2(df_boundary).
    return 1.0 - stats.chi2.cdf(lr_stat, df=df_boundary)


def chi2_pvalue(lr_stat: float, df: int) -> float:
    if lr_stat <= 0:
        return 1.0
    return float(1.0 - stats.chi2.cdf(lr_stat, df=df))


# ---------------------------------------------------------------------------
# Skewness test for presence of inefficiency (Schmidt-Lin 1984, Coelli 1995)
# ---------------------------------------------------------------------------


def ols_residual_skewness(residuals: np.ndarray) -> float:
    """Sample skewness of OLS residuals (a quick specification diagnostic)."""
    r = residuals - residuals.mean()
    m2 = np.mean(r**2)
    m3 = np.mean(r**3)
    if m2 <= 0:
        return 0.0
    return float(m3 / m2**1.5)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def group_panel(
    data: pd.DataFrame,
    id_col: str,
    time_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Return ``(group_idx, time_vec, counts, unique_ids)``.

    ``group_idx`` is an integer array with each observation's group index,
    ``time_vec`` is the time column (or zeros when ``time_col`` is None),
    ``counts`` has per-group sizes ``T_i`` in the same order as ``unique_ids``.
    """
    ids = data[id_col].to_numpy()
    unique_ids, group_idx = np.unique(ids, return_inverse=True)
    time_vec = (
        data[time_col].to_numpy(dtype=float)
        if time_col is not None
        else np.zeros(len(data), dtype=float)
    )
    counts = np.bincount(group_idx)
    return group_idx, time_vec, counts, list(unique_ids)


__all__ = [
    "loglik_halfnormal",
    "loglik_exponential",
    "loglik_truncated_normal",
    "jondrow_halfnormal",
    "jondrow_exponential",
    "jondrow_truncnormal",
    "numerical_hessian",
    "safe_invert_hessian",
    "build_design",
    "build_optional_design",
    "evaluate_sigma",
    "lr_test_statistic",
    "mixed_chi_bar_pvalue",
    "chi2_pvalue",
    "ols_residual_skewness",
    "group_panel",
]
