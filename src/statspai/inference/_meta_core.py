"""Shared numerics for summary-data meta-analysis and meta-regression.

The model is ``y_i = x_i' beta + u_i + e_i`` with ``e_i ~ N(0, v_i)``
(``v_i`` the squared standard error of study ``i``) and ``u_i ~ N(0,
tau2)``. With ``W = diag(1 / (v_i + tau2))`` the coefficient estimate is
weighted least squares, and every estimator of ``tau2`` used here is a
function of the residual projection

    P = W - W X (X' W X)^{-1} X' W.

``sp.meta_analysis`` is the intercept-only case; ``sp.meta_regression``
adds moderators; ``sp.trim_and_fill`` refits the intercept-only model on
trimmed and augmented sets of studies.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import optimize as _optimize

from ..exceptions import MethodIncompatibility

#: Between-study variance estimators accepted by the meta-regression core.
TAU2_METHODS: Tuple[str, ...] = ("DL", "REML", "ML", "PM", "fixed")


def projection(v: np.ndarray, X: np.ndarray, tau2: float) -> np.ndarray:
    """Residual projection ``P`` at ``tau2`` (a ``k x k`` matrix)."""
    w = 1.0 / (v + tau2)
    XtW = X.T * w
    A = XtW @ X
    return np.diag(w) - XtW.T @ np.linalg.solve(A, XtW)


def wls(
    y: np.ndarray, v: np.ndarray, X: np.ndarray, tau2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Coefficients and their model-based covariance at ``tau2``."""
    w = 1.0 / (v + tau2)
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    return cov @ (XtW @ y), cov


def _loglik(
    y: np.ndarray, v: np.ndarray, X: np.ndarray, t2: float, reml: bool
) -> float:
    w = 1.0 / (v + t2)
    P = projection(v, X, t2)
    ll = -0.5 * float(np.sum(np.log(v + t2))) - 0.5 * float(y @ P @ y)
    if reml:
        _sign, logdet = np.linalg.slogdet((X.T * w) @ X)
        ll -= 0.5 * float(logdet)
    return ll


def _scoring(
    y: np.ndarray, v: np.ndarray, X: np.ndarray, start: float, reml: bool
) -> float:
    """Fisher scoring for ``tau2`` on the (restricted) likelihood, truncated at 0.

    The step is halved while it lowers the likelihood, so the iteration
    cannot overshoot; it stops when the step no longer moves ``tau2`` at
    double precision relative to the within-study variances.
    """
    tau2 = max(float(start), 0.0)
    scale = float(np.mean(v))
    for _ in range(500):
        w = 1.0 / (v + tau2)
        P = projection(v, X, tau2)
        Py = P @ y
        if reml:
            score = float(Py @ Py) - float(np.trace(P))
            info = float(np.sum(P * P))
        else:
            score = float(Py @ Py) - float(np.sum(w))
            info = float(np.sum(w**2))
        if not info > 0:
            break
        new = max(tau2 + score / info, 0.0)
        ll_old = _loglik(y, v, X, tau2, reml)
        for _ in range(60):
            if _loglik(y, v, X, new, reml) >= ll_old - 1e-12 * max(1.0, abs(ll_old)):
                break
            new = 0.5 * (new + tau2)
        moved = abs(new - tau2)
        tau2 = new
        if moved <= 1e-14 * max(scale, tau2):
            break
    return float(tau2)


def _paule_mandel(y: np.ndarray, v: np.ndarray, X: np.ndarray) -> float:
    """Paule-Mandel: the ``tau2`` at which the generalised Q equals its df."""
    k, p = X.shape
    target = float(k - p)

    def gap(t2: float) -> float:
        return float(y @ projection(v, X, t2) @ y) - target

    if gap(0.0) <= 0.0:
        return 0.0
    hi = max(float(np.var(y)), float(np.mean(v)), 1e-12)
    for _ in range(200):
        if gap(hi) < 0.0:
            break
        hi *= 4.0
    else:  # pragma: no cover - Q decreases to 0 as tau2 grows
        return float(hi)
    return float(_optimize.brentq(gap, 0.0, hi, xtol=1e-300, rtol=8.9e-16))


def tau2_dl(y: np.ndarray, v: np.ndarray, X: np.ndarray) -> float:
    """DerSimonian-Laird (method-of-moments) estimate, truncated at 0."""
    k, p = X.shape
    P = projection(v, X, 0.0)
    q_e = float(y @ P @ y)
    tr = float(np.trace(P))
    return max(0.0, (q_e - (k - p)) / tr) if tr > 0 else 0.0


def estimate_tau2(y: np.ndarray, v: np.ndarray, X: np.ndarray, method: str) -> float:
    """Between-study variance of the model with design ``X`` (intercept included)."""
    if method == "fixed":
        return 0.0
    if method == "DL":
        return tau2_dl(y, v, X)
    if method == "PM":
        return _paule_mandel(y, v, X)
    if method in ("REML", "ML"):
        return _scoring(y, v, X, tau2_dl(y, v, X), reml=method == "REML")
    raise MethodIncompatibility(
        f"unknown between-study variance estimator {method!r}.",
        recovery_hint="Use one of 'DL', 'REML', 'ML', 'PM' or 'fixed'.",
    )


def typical_variance(v: np.ndarray, X: np.ndarray) -> float:
    """Higgins-Thompson typical within-study variance, ``(k - p) / tr(P_0)``."""
    k, p = X.shape
    return float((k - p) / np.trace(projection(v, X, 0.0)))
