"""Numerical core of :func:`statspai.timeseries.tvp_var.tvp_var`.

Two estimators of a VAR whose coefficients follow random walks.

``kalman``: every equation is a dynamic linear model, filtered and
smoothed by the kernels of :mod:`statspai.timeseries.dlm` (no second
Kalman filter is written here); this file adds the likelihood search for
the one-variance-per-equation restriction.

``forgetting``: the filter of Koop and Korobilis with a forgetting factor
for the state covariance and an exponentially weighted error covariance,
run in information form, which is exact under a diffuse start.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from .dlm import _kernels, dlm

_LOG_2PI = float(np.log(2.0 * np.pi))

__all__ = [
    "lag_design",
    "kalman_equation",
    "mle_free",
    "mle_common",
    "forgetting_filter",
    "ma_coefficients",
    "max_roots",
]


def lag_design(Y: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """Left- and right-hand side of a VAR(p): lags first, constant last.

    Column ``(l - 1) K + j`` of ``X`` is variable ``j`` at lag ``l``, the
    order of :func:`statspai.var`.
    """
    n = Y.shape[0]
    parts = [Y[lags - lag : n - lag] for lag in range(1, lags + 1)]
    X = np.column_stack(parts + [np.ones(n - lags)])
    return np.ascontiguousarray(Y[lags:]), np.ascontiguousarray(X)


def kalman_equation(
    y: np.ndarray,
    X: np.ndarray,
    V: float,
    W: np.ndarray,
    m0: np.ndarray,
    C0: np.ndarray,
) -> Dict[str, Any]:
    """Filter and smooth one equation at given variances."""
    kern = _kernels()
    w = np.ascontiguousarray(W, dtype=float)
    m, C, R, f, Q, ll = kern["filter"](
        np.ascontiguousarray(y),
        np.ascontiguousarray(X),
        float(V),
        w,
        np.ascontiguousarray(m0, dtype=float),
        np.ascontiguousarray(C0, dtype=float),
    )
    s, S = kern["smooth"](m, C, R)
    return {
        "filtered": m[1:],
        "filtered_var": np.clip(np.einsum("tii->ti", C[1:]), 0.0, None),
        "smoothed": s[1:],
        "smoothed_var": np.clip(np.einsum("tii->ti", S[1:]), 0.0, None),
        "forecast": f,
        "forecast_var": Q,
        "loglik": float(ll),
    }


def mle_free(
    y: np.ndarray,
    X: np.ndarray,
    m0: np.ndarray,
    C0: np.ndarray,
    obs_var: Optional[float],
    state_var: Optional[np.ndarray],
) -> Tuple[float, np.ndarray]:
    """Variances of one equation from :func:`statspai.dlm`.

    One state variance per coefficient (or the given ones, the
    observation variance alone being estimated). The regressors are passed
    to ``dlm`` under neutral names with the intercept first, its order,
    and the variances are returned in the order of ``X`` (constant last).
    """
    k = X.shape[1]
    cols = [f"x{j}" for j in range(k - 1)]
    frame = pd.DataFrame(X[:, : k - 1], columns=cols)
    frame["yy"] = y
    order = [k - 1] + list(range(k - 1))  # dlm puts the intercept first
    formula = "yy ~ " + (" + ".join(cols) if cols else "1")
    sv = None if state_var is None else np.asarray(state_var, dtype=float)[order]
    fit = dlm(
        formula,
        frame,
        obs_var=obs_var,
        state_var=sv,
        m0=m0[order],
        C0=C0[np.ix_(order, order)],
    )
    est = fit.variances["estimate"].to_numpy(dtype=float)
    W = np.empty(k)
    W[order] = est[1:]
    return float(est[0]), W


def mle_common(
    y: np.ndarray,
    X: np.ndarray,
    m0: np.ndarray,
    C0: np.ndarray,
    obs_var: Optional[float],
) -> Tuple[float, np.ndarray, bool, str]:
    """One state variance shared by every coefficient of the equation.

    The search mirrors ``dlm``: L-BFGS-B on the log variances from three
    starts, then a simplex polish (the likelihood is flat near its top).
    """
    k = X.shape[1]
    filt = _kernels()["filter"]
    yc, Xc = np.ascontiguousarray(y), np.ascontiguousarray(X)
    scale = float(np.var(y)) or 1.0
    n_v = 0 if obs_var is not None else 1

    def unpack(par: np.ndarray) -> Tuple[float, np.ndarray]:
        V = float(obs_var) if obs_var is not None else float(np.exp(par[0]))
        return V, np.full(k, float(np.exp(par[n_v])))

    def nll(par: np.ndarray) -> float:
        V, W = unpack(par)
        val = -filt(yc, Xc, V, W, m0, C0)[5]
        return float(val) if np.isfinite(val) else 1e300

    best: Any = None
    for start in (-2.0, -5.0, -9.0):
        p0 = np.concatenate(
            [np.full(n_v, np.log(scale * 0.5)), [np.log(scale) + start]]
        )
        res = optimize.minimize(
            nll,
            p0,
            method="L-BFGS-B",
            bounds=[(-40.0, 40.0)] * p0.size,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
        )
        if best is None or res.fun < best.fun:
            best = res
    pol = optimize.minimize(
        nll,
        best.x,
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 2000},
    )
    ok = bool(best.success) or "CONVERGENCE" in str(best.message).upper()
    message = str(best.message)
    if pol.fun < best.fun:
        best, ok = pol, True
    V, W = unpack(np.clip(best.x, -40.0, 40.0))
    return V, W, ok, message


def forgetting_filter(
    Y: np.ndarray,
    X: np.ndarray,
    lam: float,
    kappa: float,
    b0: np.ndarray,
    P0: np.ndarray,
    S0: np.ndarray,
    update: str = "filtered",
) -> Dict[str, Any]:
    """Forgetting-factor filter for ``y_t = B_t x_t + u_t``.

    The state is ``beta = (b_1', ..., b_K')'``, the rows of ``B`` stacked,
    so the observation matrix is ``I_K (x) x_t'``. With ``Omega`` the
    precision of the state and ``h = Omega beta``::

        prediction   beta_{t|t-1} = beta_{t-1},  P_{t|t-1} = P_{t-1} / lam
        update       Omega_t = lam Omega_{t-1} + S_{t-1}^{-1} (x) x_t x_t'
                     h_t     = lam h_{t-1} + (S_{t-1}^{-1} y_t) (x) x_t
        covariance   S_t = kappa S_{t-1} + (1 - kappa) e_t e_t'

    where ``e_t`` is ``y_t - B_t x_t`` (``update='filtered'``) or the
    one-step prediction error ``y_t - B_{t-1} x_t`` (``'predicted'``).

    Parameters
    ----------
    Y : (T, K) array
    X : (T, k) array
    lam, kappa : float
    b0 : (K, k) array
        Prior mean of the coefficients.
    P0 : (K k, K k) array
        Prior covariance of ``beta``.
    S0 : (K, K) array
        Error covariance used at the first date.
    update : {'filtered', 'predicted'}

    Returns
    -------
    dict
        ``coef`` (T, K, k), ``se`` (T, K, k), ``cov_last`` (K k, K k),
        ``sigma`` (T, K, K), ``pred_error`` (T, K), ``pred_cov``
        (T, K, K), ``loglik``.
    """
    T, K = Y.shape
    k = X.shape[1]
    Om = np.linalg.inv(P0)
    Om = 0.5 * (Om + Om.T)
    h = Om @ b0.reshape(-1)
    B = b0.copy()
    S = S0.copy()
    coef = np.empty((T, K, k))
    se = np.empty((T, K, k))
    sig = np.empty((T, K, K))
    perr = np.empty((T, K))
    pcov = np.empty((T, K, K))
    eye_K = np.eye(K)
    ll = 0.0
    P = np.linalg.inv(Om)
    for t in range(T):
        x = X[t]
        Z = np.kron(eye_K, x[None, :])
        e = Y[t] - B @ x
        F = Z @ (P / lam) @ Z.T + S
        F = 0.5 * (F + F.T)
        sign, logdet = np.linalg.slogdet(F)
        ll += -0.5 * (K * _LOG_2PI + logdet + float(e @ np.linalg.solve(F, e)))
        Sinv = np.linalg.inv(S)
        Om = lam * Om + np.kron(Sinv, np.outer(x, x))
        Om = 0.5 * (Om + Om.T)
        h = lam * h + np.kron(Sinv @ Y[t], x)
        P = np.linalg.inv(Om)
        P = 0.5 * (P + P.T)
        B = (P @ h).reshape(K, k)
        r = Y[t] - B @ x if update == "filtered" else e
        S = kappa * S + (1.0 - kappa) * np.outer(r, r)
        coef[t] = B
        se[t] = np.sqrt(np.clip(np.diag(P), 0.0, None)).reshape(K, k)
        sig[t] = S
        perr[t] = e
        pcov[t] = F
    return {
        "coef": coef,
        "se": se,
        "cov_last": P,
        "sigma": sig,
        "pred_error": perr,
        "pred_cov": pcov,
        "loglik": float(ll),
    }


def ma_coefficients(B: np.ndarray, lags: int, periods: int) -> np.ndarray:
    """Moving-average matrices ``Phi_0 .. Phi_periods`` of a VAR.

    ``B`` is (K, K p + 1) with the lags first; ``Phi_s[i, j]`` is the
    response of variable ``i`` after ``s`` periods to a unit innovation in
    variable ``j``.
    """
    K = B.shape[0]
    A: List[np.ndarray] = [B[:, lag * K : (lag + 1) * K] for lag in range(lags)]
    phi = np.zeros((periods + 1, K, K))
    phi[0] = np.eye(K)
    for s in range(1, periods + 1):
        for j in range(min(s, lags)):
            phi[s] += phi[s - j - 1] @ A[j]
    return phi


def max_roots(coef: np.ndarray, lags: int) -> np.ndarray:
    """Largest modulus of the companion eigenvalues at every date.

    ``coef`` is (T, K, K p + 1), lags first.
    """
    T, K, _ = coef.shape
    comp = np.zeros((T, K * lags, K * lags))
    comp[:, :K, :] = coef[:, :, : K * lags]
    if lags > 1:
        comp[:, K:, : K * (lags - 1)] = np.eye(K * (lags - 1))
    out: np.ndarray = np.abs(np.linalg.eigvals(comp)).max(axis=1)
    return out
