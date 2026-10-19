"""Forecast-error variance of ``sp.causal_impact`` with parameter uncertainty.

The point forecast of ``sp.causal_impact`` is a fixed linear map of the
pre-period outcomes: with OLS ``beta`` (``B = X_o^+``) and the plug-in
state-space model's conditional-mean weights ``K = S_po S_oo^{-1}``,

    y_hat_post = L y_pre,   L = X_p B + K (I - X_o B).

Because ``L X_o = X_p``, the regression coefficients cancel from the error:
``e = y_post - L y_pre = u_post - L u_pre`` for the regression errors ``u``.
So the error variance is ``V = T_pp - L T_op - T_po L' + L T_oo L'`` with
``T`` the covariance of ``u``. That accounts exactly for the estimation of
``beta``. The plug-in parameters that build ``L`` (lag-1 correlation and
standard deviation of the OLS residuals) are not estimates of ``T``, so
using them for ``T`` as well (what the conditional variance did) understated
the error variance: 95% intervals covered 57% to 90% of null effects in
200-replication studies.

Here ``T`` is the AR(1)-plus-noise covariance
``sigma2 * (psi / (1 - rho^2) * rho^|i-j| + (1 - psi) I)`` fitted by
restricted maximum likelihood (REML) on the pre-period, with the
regression in the mean. The likelihood is evaluated with a scalar Kalman
filter (stationary start), first over a grid of ``(rho, psi)`` and then
refined. Uncertainty in ``T`` itself is not propagated; the interval uses
a t quantile with ``n_pre - p - 3`` degrees of freedom (``p`` regression
coefficients, three covariance parameters).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import optimize

_RHO_MAX = 0.98


def _whiten(
    Z: np.ndarray, rho: np.ndarray, psi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Kalman-whiten the columns of ``Z`` under unit-scale AR(1)+noise.

    ``rho``, ``psi`` have shape ``(G,)`` (a batch of parameter values).
    Returns the standardised innovations ``(G, n, m)`` and ``log|R|``
    ``(G,)`` for ``R = psi/(1-rho^2) rho^|i-j| + (1-psi) I``.
    """
    n, m = Z.shape
    G = rho.shape[0]
    s2s = psi
    s2e = 1.0 - psi
    P = s2s / (1.0 - rho**2)
    a = np.zeros((G, m))
    out = np.empty((G, n, m))
    logdet = np.zeros(G)
    for t in range(n):
        F = np.maximum(P + s2e, 1e-12)
        v = Z[t][None, :] - a
        out[:, t, :] = v / np.sqrt(F)[:, None]
        logdet += np.log(F)
        K = P / F
        a = rho[:, None] * (a + K[:, None] * v)
        P = rho**2 * P * (1.0 - K) + s2s
    return out, logdet


def _reml_objective(
    y: np.ndarray, X: np.ndarray, rho: np.ndarray, psi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Negative profile REML (scale concentrated out) and the scale."""
    n, k = X.shape
    W, logdet = _whiten(np.column_stack([y, X]), rho, psi)
    yw, Xw = W[:, :, 0], W[:, :, 1:]
    XtX = np.einsum("gni,gnj->gij", Xw, Xw)
    Xty = np.einsum("gni,gn->gi", Xw, yw)
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    resid = yw - np.einsum("gnk,gk->gn", Xw, beta)
    rss = np.einsum("gn,gn->g", resid, resid)
    dof = max(n - k, 1)
    s2 = np.maximum(rss / dof, 1e-300)
    sign, ld_xtx = np.linalg.slogdet(XtX)
    nll = 0.5 * (dof * np.log(s2) + logdet + ld_xtx)
    nll = np.where(sign > 0, nll, np.inf)
    return nll, s2


def reml_ar1_noise(y: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    """REML fit of ``y = X b + u``, ``u`` AR(1) plus white noise."""
    rho_g = np.linspace(-0.95, 0.95, 39)
    psi_g = np.linspace(0.0, 1.0, 21)
    R, Ps = np.meshgrid(rho_g, psi_g, indexing="ij")
    nll, _ = _reml_objective(y, X, R.ravel(), Ps.ravel())
    best = int(np.argmin(nll))
    x0 = np.array([R.ravel()[best], Ps.ravel()[best]])

    def f(th: np.ndarray) -> float:
        v, _ = _reml_objective(y, X, np.array([th[0]]), np.array([th[1]]))
        return float(v[0])

    res = optimize.minimize(
        f,
        x0,
        method="L-BFGS-B",
        bounds=[(-_RHO_MAX, _RHO_MAX), (0.0, 1.0)],
    )
    th = res.x if res.fun <= nll[best] else x0
    rho, psi = float(th[0]), float(th[1])
    _, s2 = _reml_objective(y, X, np.array([rho]), np.array([psi]))
    sigma2 = float(s2[0])
    return {
        "rho": rho,
        "sigma2_state": sigma2 * psi,
        "sigma2_obs": sigma2 * (1.0 - psi),
        "sigma2": sigma2,
        "noise_share": 1.0 - psi,
    }


def error_covariance(n: int, err: Dict[str, float]) -> np.ndarray:
    """Stationary covariance of AR(1)-plus-noise errors over ``n`` periods."""
    rho = err["rho"]
    idx = np.arange(n)
    lag = np.abs(idx[:, None] - idx[None, :])
    T = err["sigma2_state"] / (1.0 - rho**2) * rho**lag
    T[idx, idx] += err["sigma2_obs"]
    return T


def plugin_covariance(n: int, rho: float, sigma_obs: float, sigma_state: float):
    """Covariance implied by the plug-in filter (its non-stationary start)."""
    v = np.empty(n)
    v[0] = sigma_state**2
    for t in range(1, n):
        v[t] = rho**2 * v[t - 1] + sigma_state**2
    idx = np.arange(n)
    lo = np.minimum(idx[:, None], idx[None, :])
    S = rho ** np.abs(idx[:, None] - idx[None, :]) * v[lo]
    S[idx, idx] += sigma_obs**2
    return S


def predictor_map(X_aug: np.ndarray, pre: np.ndarray, model: Dict) -> np.ndarray:
    """``L`` with ``y_hat_post = L y_pre`` for the plug-in forecast."""
    n = X_aug.shape[0]
    S = plugin_covariance(
        n, float(model["rho"]), float(model["sigma_obs"]), float(model["sigma_state"])
    )
    Xo, Xp = X_aug[pre], X_aug[~pre]
    B = np.linalg.pinv(Xo)
    K = np.linalg.solve(S[np.ix_(pre, pre)], S[np.ix_(pre, ~pre)]).T
    return Xp @ B + K @ (np.eye(int(pre.sum())) - Xo @ B)


def forecast_error_cov(L: np.ndarray, T: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """``Var(u_post - L u_pre)`` for error covariance ``T``."""
    Too = T[np.ix_(pre, pre)]
    Tpp = T[np.ix_(~pre, ~pre)]
    Tpo = T[np.ix_(~pre, pre)]
    LTop = L @ Tpo.T
    return Tpp - LTop - LTop.T + L @ Too @ L.T
