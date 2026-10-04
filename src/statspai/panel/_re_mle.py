"""Random-effects panel regression by Gaussian maximum likelihood.

``y_it = x_it'b + u_i + e_it`` with ``u_i ~ N(0, sigma_u^2)`` and ``e_it ~
N(0, sigma_e^2)``. The point estimates are those of the random-intercept
mixed model (:func:`statspai.mixed` with ``method='ml'``), found here by
maximising the likelihood concentrated in the one ratio ``sigma_u^2 /
sigma_e^2``: given the ratio, ``b`` is a GLS fit on panel sums and
``sigma_e^2`` a mean square, so the cost does not grow with the number of
panels the way a general mixed-model fit does. The standard
errors come from the observed information of the full likelihood in ``(b,
sigma_u^2, sigma_e^2)``, which is what Stata's ``xtreg, mle`` reports; they
differ slightly from the GLS standard errors ``(X'V^{-1}X)^{-1}`` of the
mixed-model fit, which treat the variance components as known.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..exceptions import DataInsufficient

__all__ = ["fit_re_mle", "re_loglik", "re_information"]


def _panel_sums(
    resid: np.ndarray, codes: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    total = np.zeros(n)
    np.add.at(total, codes, resid)
    squares = np.zeros(n)
    np.add.at(squares, codes, resid**2)
    return total, squares


def re_loglik(
    y: np.ndarray,
    X: np.ndarray,
    codes: np.ndarray,
    beta: np.ndarray,
    s2u: float,
    s2e: float,
) -> float:
    """Log likelihood of the Gaussian random-effects model."""
    n = int(codes.max()) + 1
    T = np.bincount(codes, minlength=n).astype(float)
    S, SS = _panel_sums(y - X @ beta, codes, n)
    D = s2e + T * s2u
    ll = -0.5 * (
        T * np.log(2.0 * np.pi)
        + (T - 1.0) * np.log(s2e)
        + np.log(D)
        + (SS - s2u * S**2 / D) / s2e
    )
    return float(ll.sum())


def re_information(
    y: np.ndarray,
    X: np.ndarray,
    codes: np.ndarray,
    beta: np.ndarray,
    s2u: float,
    s2e: float,
) -> np.ndarray:
    """Observed information (minus the Hessian of the log likelihood) in
    ``(beta, sigma_u^2, sigma_e^2)``, evaluated at the given values."""
    n = int(codes.max()) + 1
    k = X.shape[1]
    T = np.bincount(codes, minlength=n).astype(float)
    S, SS = _panel_sums(y - X @ beta, codes, n)
    xsum = np.zeros((n, k))
    np.add.at(xsum, codes, X)
    D = s2e + T * s2u
    a = s2u / D
    H = np.zeros((k + 2, k + 2))
    # beta block: -X'V^{-1}X
    H[:k, :k] = -(X.T @ X - (xsum * a[:, None]).T @ xsum) / s2e
    # beta with the variances (the score in beta is zero at the optimum)
    H[:k, k] = H[k, :k] = -((s2e / D**2 * S)[:, None] * xsum).sum(axis=0) / s2e
    H[:k, k + 1] = H[k + 1, :k] = ((s2u / D**2 * S)[:, None] * xsum).sum(axis=0) / s2e
    # variance block
    H[k, k] = -0.5 * np.sum(-(T**2) / D**2 + 2.0 * T * S**2 / D**3)
    H[k, k + 1] = H[k + 1, k] = -0.5 * np.sum(-T / D**2 + 2.0 * S**2 / D**3)
    curve = 2.0 * (s2e * D - (D + s2e) ** 2) / (s2e**3 * D**3)
    H[k + 1, k + 1] = -0.5 * np.sum(
        -(T - 1.0) / s2e**2 - 1.0 / D**2 + 2.0 * SS / s2e**3 + s2u * S**2 * curve
    )
    return -H


def _concentrated_fit(
    y: np.ndarray, X: np.ndarray, codes: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Maximum-likelihood ``(beta, sigma_u^2, sigma_e^2)``.

    With ``r = sigma_u^2 / sigma_e^2`` the inverse covariance of panel ``i``
    is ``(I - a_i 11') / sigma_e^2``, ``a_i = r / (1 + T_i r)``, so the GLS
    normal equations need only ``X'X``, ``X'y`` and the panel sums.
    """
    n = int(codes.max()) + 1
    N, k = X.shape
    T = np.bincount(codes, minlength=n).astype(float)
    xsum = np.zeros((n, k))
    np.add.at(xsum, codes, X)
    ysum = np.zeros(n)
    np.add.at(ysum, codes, y)
    xtx, xty, yty = X.T @ X, X.T @ y, float(y @ y)

    def at(r: float) -> Tuple[np.ndarray, float]:
        a = r / (1.0 + T * r)
        A = xtx - (xsum * a[:, None]).T @ xsum
        b = xty - xsum.T @ (a * ysum)
        beta = np.linalg.solve(A, b)
        q = yty - float(a @ ysum**2) - float(beta @ b)
        return beta, max(q, np.finfo(float).tiny) / N

    def neg_profile(log_r: float) -> float:
        r = float(np.exp(log_r))
        _, s2e = at(r)
        return float(0.5 * (N * np.log(s2e) + np.log1p(T * r).sum()))

    best = optimize.minimize_scalar(
        neg_profile, bounds=(-30.0, 30.0), method="bounded", options={"xatol": 1e-13}
    )
    r = float(np.exp(best.x))
    _, s2e_pooled = at(0.0)
    if 0.5 * N * np.log(s2e_pooled) <= best.fun:
        r = 0.0  # the likelihood is highest on the boundary sigma_u = 0
    beta, s2e = at(r)
    return beta, r * s2e, s2e


def fit_re_mle(
    data: pd.DataFrame,
    dep_var: str,
    indep_vars: List[str],
    entity: str,
    alpha: float,
) -> Dict[str, Any]:
    """Estimates and the pieces a result is assembled from."""
    frame = data[[entity, dep_var] + list(indep_vars)].dropna()
    if frame[entity].nunique() < 2:
        raise DataInsufficient(
            "sp.panel(method='mle') needs at least two panels.",
            recovery_hint="Check the entity column.",
        )
    y = frame[dep_var].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(len(frame)), frame[list(indep_vars)].to_numpy(dtype=float)]
    )
    codes, _ = pd.factorize(frame[entity], sort=True)
    names = ["const"] + list(indep_vars)
    beta, s2u, s2e = _concentrated_fit(y, X, codes)
    k = len(names)
    ll = re_loglik(y, X, codes, beta, s2u, s2e)
    cov_full = np.linalg.inv(re_information(y, X, codes, beta, s2u, s2e))
    cov = cov_full[:k, :k]
    sigma_u, sigma_e = float(np.sqrt(s2u)), float(np.sqrt(s2e))
    # delta method from the variances to the standard deviations
    sigma_u_se = float(np.sqrt(cov_full[k, k]) / (2.0 * sigma_u)) if s2u > 0 else np.nan
    sigma_e_se = float(np.sqrt(cov_full[k + 1, k + 1]) / (2.0 * sigma_e))

    # comparison models: pooled OLS (sigma_u = 0) and the constant-only model
    resid_ols = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    n_obs = y.size
    ll_ols = (
        -0.5 * n_obs * (np.log(2.0 * np.pi * (resid_ols @ resid_ols) / n_obs) + 1.0)
    )
    ones = X[:, :1]
    ll_null = re_loglik(y, ones, codes, *_concentrated_fit(y, ones, codes))
    lr_model = 2.0 * (ll - ll_null)
    lr_sigma_u = max(2.0 * (ll - ll_ols), 0.0)
    return {
        "names": names,
        "beta": beta,
        "cov": cov,
        "nobs": int(n_obs),
        "n_groups": int(codes.max()) + 1,
        "fitted": X @ beta,
        "residuals": y - X @ beta,
        "model_info": {
            "sigma_u": sigma_u,
            "sigma_e": sigma_e,
            "sigma_u_se": sigma_u_se,
            "sigma_e_se": sigma_e_se,
            "rho": s2u / (s2u + s2e),
            "ll": ll,
            "ll_null": ll_null,
            "ll_ols": float(ll_ols),
            "lr_chi2": float(lr_model),
            "lr_df": len(indep_vars),
            "lr_pvalue": float(stats.chi2.sf(lr_model, len(indep_vars))),
            # H0: sigma_u = 0 is on the boundary: a 50:50 mixture of chi2(0)
            # and chi2(1)
            "lr_sigma_u": float(lr_sigma_u),
            "lr_sigma_u_pvalue": (
                float(0.5 * stats.chi2.sf(lr_sigma_u, 1)) if lr_sigma_u > 0 else 1.0
            ),
        },
    }
