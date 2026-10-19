"""Parametric survival-time models in the parametrisation of Stata's
``stteffects`` (and ``streg``'s accelerated-failure-time metric).

Each model has a location ``x'b`` and, except the exponential, a log scale
``eta`` (a constant; ``s = exp(eta)``):

========== ============================ ==============================
model      survivor ``S(t)``            mean ``E(T | x)``
========== ============================ ==============================
weibull    ``exp{-(t e^{-xb})^s}``      ``e^{xb} Gamma(1 + 1/s)``
exponential the Weibull with ``s = 1``  ``e^{xb}``
lognormal  ``1 - Phi{(ln t - xb)/s}``   ``exp(xb + s^2/2)``
========== ============================ ==============================

The functions return per-observation log densities / log survivors and
their analytic scores with respect to ``(b, eta)``; the estimators
weight and stack them.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import optimize, special, stats

from ..exceptions import ConvergenceFailure, MethodIncompatibility

DISTS = ("weibull", "exponential", "lognormal")


def check_dist(dist: str, what: str) -> str:
    key = str(dist).lower()
    if key in ("lnormal", "log-normal"):
        key = "lognormal"
    if key in ("exp",):
        key = "exponential"
    if key not in DISTS:
        raise MethodIncompatibility(
            f"{what} must be one of {list(DISTS)}; got {dist!r}.",
            recovery_hint="Use 'weibull' (Stata's default), 'exponential' or "
            "'lognormal'.",
        )
    return key


def n_params(dist: str, k: int) -> int:
    return k if dist == "exponential" else k + 1


def loglik_parts(
    theta: np.ndarray, X: np.ndarray, logt: np.ndarray, dist: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``log f``, ``log S`` and their scores (``n x p``) at ``theta``."""
    k = X.shape[1]
    xb = X @ theta[:k]
    if dist == "exponential":
        z = logt - xb
        ez = np.exp(z)
        lf = z - ez - logt
        lS = -ez
        sf = ((ez - 1.0)[:, None]) * X
        sS = ez[:, None] * X
        return lf, lS, sf, sS
    eta = theta[k]
    s = np.exp(eta)
    if dist == "weibull":
        z = s * (logt - xb)
        ez = np.exp(z)
        lf = eta - logt + z - ez
        lS = -ez
        sf = np.column_stack([(s * (ez - 1.0))[:, None] * X, 1.0 + z - z * ez])
        sS = np.column_stack([(s * ez)[:, None] * X, -z * ez])
        return lf, lS, sf, sS
    # lognormal
    u = (logt - xb) / s
    lf = -eta - logt - 0.5 * u**2 - 0.5 * np.log(2 * np.pi)
    lS = stats.norm.logsf(u)
    mills = np.exp(stats.norm.logpdf(u) - lS)
    sf = np.column_stack([(u / s)[:, None] * X, u**2 - 1.0])
    sS = np.column_stack([(mills / s)[:, None] * X, mills * u])
    return lf, lS, sf, sS


def mean_time(theta: np.ndarray, X: np.ndarray, dist: str) -> np.ndarray:
    k = X.shape[1]
    xb = X @ theta[:k]
    if dist == "exponential":
        return np.asarray(np.exp(xb))
    s = np.exp(theta[k])
    if dist == "weibull":
        return np.asarray(np.exp(xb + special.gammaln(1.0 + 1.0 / s)))
    return np.asarray(np.exp(xb + 0.5 * s**2))


def survivor(
    theta: np.ndarray, X: np.ndarray, logt: np.ndarray, dist: str
) -> np.ndarray:
    return np.asarray(np.exp(loglik_parts(theta, X, logt, dist)[1]))


def score_rows(
    theta: np.ndarray,
    X: np.ndarray,
    logt: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    dist: str,
    censored_term: bool,
) -> np.ndarray:
    """Weighted per-row scores: ``w (d s_f + (1 - d) s_S)`` (likelihood with
    the censored term) or ``w d s_f`` (uncensored rows only)."""
    _, _, sf, sS = loglik_parts(theta, X, logt, dist)
    rows = d[:, None] * sf
    if censored_term:
        rows = rows + (1.0 - d)[:, None] * sS
    return np.asarray(w[:, None] * rows)


def fit(
    X: np.ndarray,
    logt: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    dist: str,
    censored_term: bool,
    what: str,
) -> np.ndarray:
    """Maximise the weighted log likelihood (Newton on the analytic score,
    Hessian by central differences of it, with step halving)."""
    k = X.shape[1]
    use = w > 0
    if censored_term:
        n_ev = float((w * d).sum())
    else:
        n_ev = float((w * d).sum())
        use = use & (d == 1)
    if n_ev <= 0:
        raise ConvergenceFailure(
            f"{what}: no events carry weight, the model is not identified.",
            recovery_hint="Check the event indicator and the weights.",
        )
    Xu, lu, du, wu = X[use], logt[use], d[use], w[use]

    def negll(th: np.ndarray) -> float:
        lf, lS, _, _ = loglik_parts(th, Xu, lu, dist)
        val = du * lf
        if censored_term:
            val = val + (1.0 - du) * lS
        return float(-(wu * val).sum())

    def grad(th: np.ndarray) -> np.ndarray:
        return -score_rows(th, Xu, lu, du, wu, dist, censored_term).sum(axis=0)

    ev = du == 1
    beta0 = np.linalg.lstsq(
        Xu[ev] * np.sqrt(wu[ev])[:, None], lu[ev] * np.sqrt(wu[ev]), rcond=None
    )[0]
    theta = np.concatenate([beta0, [] if dist == "exponential" else [0.0]])
    res = optimize.minimize(
        negll, theta, jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 500}
    )
    theta = res.x
    p = len(theta)
    for _ in range(50):
        g = grad(theta)
        H = np.empty((p, p))
        for j in range(p):
            h = 1e-6 * max(1.0, abs(theta[j]))
            e = np.zeros(p)
            e[j] = h
            H[:, j] = (grad(theta + e) - grad(theta - e)) / (2 * h)
        H = 0.5 * (H + H.T)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceFailure(
                f"{what}: singular information matrix.",
                recovery_hint="Drop collinear covariates.",
            ) from exc
        f0 = negll(theta)
        t = 1.0
        while t > 1e-8 and negll(theta - t * step) > f0 + 1e-12 * abs(f0):
            t *= 0.5
        theta = theta - t * step
        if np.max(np.abs(t * step)) < 1e-11:
            break
    if not np.all(np.isfinite(theta)) or np.max(np.abs(grad(theta))) > 1e-5 * max(
        1.0, n_ev
    ):
        raise ConvergenceFailure(
            f"{what}: the {dist} model did not converge.",
            recovery_hint="Rescale the time variable or simplify the covariates.",
        )
    return np.asarray(theta)


def binary_choice(
    alpha: np.ndarray, Z: np.ndarray, link: str
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Probability of treatment and a function giving the score rows."""
    eta = Z @ alpha
    if link == "logit":
        p = special.expit(eta)

        def score(tr: np.ndarray) -> np.ndarray:
            return np.asarray((tr - p)[:, None] * Z)

    else:
        p = stats.norm.cdf(eta)
        dens = stats.norm.pdf(eta)
        p_c = np.clip(p, 1e-300, 1 - 1e-16)

        def score(tr: np.ndarray) -> np.ndarray:
            return np.asarray(((tr - p) * dens / (p_c * (1 - p_c)))[:, None] * Z)

    return np.asarray(p), score


def fit_binary(Z: np.ndarray, tr: np.ndarray, link: str) -> np.ndarray:
    def negll(a: np.ndarray) -> float:
        eta = Z @ a
        if link == "logit":
            return float(-(tr * eta - np.logaddexp(0.0, eta)).sum())
        return float(
            -(tr * stats.norm.logcdf(eta) + (1 - tr) * stats.norm.logsf(eta)).sum()
        )

    def grad(a: np.ndarray) -> np.ndarray:
        return -binary_choice(a, Z, link)[1](tr).sum(axis=0)

    a = np.zeros(Z.shape[1])
    res = optimize.minimize(
        negll, a, jac=grad, method="BFGS", options={"gtol": 1e-10, "maxiter": 1000}
    )
    a = res.x
    # Newton polish
    for _ in range(30):
        g = grad(a)
        p = len(a)
        H = np.empty((p, p))
        for j in range(p):
            h = 1e-6 * max(1.0, abs(a[j]))
            e = np.zeros(p)
            e[j] = h
            H[:, j] = (grad(a + e) - grad(a - e)) / (2 * h)
        step = np.linalg.solve(0.5 * (H + H.T), g)
        a = a - step
        if np.max(np.abs(step)) < 1e-12:
            break
    return np.asarray(a)
