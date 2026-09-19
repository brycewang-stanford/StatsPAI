"""
Hausman-Hall-Griliches panel negative-binomial models.

These are the models behind Stata's ``xtnbreg, fe`` and ``xtnbreg, re``
and R's ``pglm(family = negbin, model = "within" / "random")``.  Both
start from y_it | γ_it ~ Poisson(γ_it), γ_it | δ_i ~ gamma(λ_it, δ_i) with
λ_it = exp(x_it'β + offset_it), i.e. a negative binomial whose
variance-to-mean ratio 1 + δ_i is constant within a panel.

``fe`` -- conditional fixed effects.  Conditioning on Σ_t y_it removes δ_i:

    ln L_i = lnΓ(Λ_i) + lnΓ(Y_i + 1) − lnΓ(Λ_i + Y_i)
             + Σ_t [lnΓ(λ_it + y_it) − lnΓ(λ_it) − lnΓ(y_it + 1)],

with Λ_i = Σ_t λ_it and Y_i = Σ_t y_it.  Unlike the conditional Poisson,
the intercept is identified (the model is not invariant to scaling λ).
Panels with a single observation or with all-zero outcomes contribute
exactly zero to ln L and are dropped, as Stata does.

``re`` -- random effects with 1/(1 + δ_i) ~ Beta(r, s):

    ln L_i = lnΓ(r + s) + lnΓ(r + Λ_i) + lnΓ(s + Y_i) − lnΓ(r) − lnΓ(s)
             − lnΓ(r + s + Λ_i + Y_i)
             + Σ_t [lnΓ(λ_it + y_it) − lnΓ(λ_it) − lnΓ(y_it + 1)],

estimated over (β, ln r, ln s).

Both are fitted by Newton-Raphson on the analytic score, with the Hessian
taken as the central-difference Jacobian of that score (accurate to
~1e-10), and standard errors from its inverse (``vce(oim)``).  The
likelihoods are the ones printed in Stata's [XT] ``xtnbreg`` Methods and
formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import special

from ..exceptions import ConvergenceFailure, DataInsufficient


@dataclass
class HHGFit:
    params: np.ndarray
    vcov: np.ndarray
    loglik: float
    converged: bool
    iterations: int
    gradient_norm: float
    keep: np.ndarray  # row mask of the estimation sample
    n_groups: int
    n_dropped_groups: int


def _segment(values: np.ndarray, gidx: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(gidx, weights=values, minlength=n_groups)


def _hhg_parts(beta, X, y, off, gidx, n_groups):
    lam = np.exp(X @ beta + off)
    Lam = _segment(lam, gidx, n_groups)
    Y = _segment(y, gidx, n_groups)
    lnA = _segment(
        special.gammaln(lam + y) - special.gammaln(lam) - special.gammaln(y + 1.0),
        gidx,
        n_groups,
    )
    return lam, Lam, Y, lnA


def fe_loglik(theta, X, y, off, gidx, n_groups) -> float:
    lam, Lam, Y, lnA = _hhg_parts(theta, X, y, off, gidx, n_groups)
    lnB = special.gammaln(Lam) + special.gammaln(Y + 1.0) - special.gammaln(Lam + Y)
    return float(np.sum(lnA + lnB))


def fe_score(theta, X, y, off, gidx, n_groups) -> np.ndarray:
    lam, Lam, Y, _ = _hhg_parts(theta, X, y, off, gidx, n_groups)
    g = (
        special.digamma(lam + y)
        - special.digamma(lam)
        + (special.digamma(Lam) - special.digamma(Lam + Y))[gidx]
    )
    return X.T @ (g * lam)


def re_loglik(theta, X, y, off, gidx, n_groups) -> float:
    k = X.shape[1]
    beta = theta[:k]
    r, s = np.exp(theta[k]), np.exp(theta[k + 1])
    lam, Lam, Y, lnA = _hhg_parts(beta, X, y, off, gidx, n_groups)
    lnC = (
        special.gammaln(r + s)
        + special.gammaln(r + Lam)
        + special.gammaln(s + Y)
        - special.gammaln(r)
        - special.gammaln(s)
        - special.gammaln(r + s + Lam + Y)
    )
    return float(np.sum(lnA + lnC))


def re_score(theta, X, y, off, gidx, n_groups) -> np.ndarray:
    k = X.shape[1]
    beta = theta[:k]
    r, s = np.exp(theta[k]), np.exp(theta[k + 1])
    lam, Lam, Y, _ = _hhg_parts(beta, X, y, off, gidx, n_groups)
    big = special.digamma(r + s + Lam + Y)
    g = (
        special.digamma(lam + y)
        - special.digamma(lam)
        + (special.digamma(r + Lam) - big)[gidx]
    )
    d_r = np.sum(
        special.digamma(r + s) + special.digamma(r + Lam) - special.digamma(r) - big
    )
    d_s = np.sum(
        special.digamma(r + s) + special.digamma(s + Y) - special.digamma(s) - big
    )
    return np.concatenate([X.T @ (g * lam), [r * d_r, s * d_s]])


def _jacobian(
    score: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, h: float = 1e-5
):
    k = theta.size
    J = np.empty((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h * max(1.0, abs(theta[j]))
        J[:, j] = (score(theta + e) - score(theta - e)) / (2.0 * e[j])
    return 0.5 * (J + J.T)


def _newton(
    loglik: Callable[[np.ndarray], float],
    score: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    maxiter: int,
    tol: float,
) -> Tuple[np.ndarray, float, bool, int, np.ndarray]:
    theta = np.asarray(theta0, dtype=float).copy()
    f = loglik(theta)
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        g = score(theta)
        H = _jacobian(score, theta)
        # Newton direction on -H; fall back to a gradient step if -H is not
        # positive definite (far from the optimum).
        try:
            np.linalg.cholesky(-H)
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            step = g / max(1.0, float(np.max(np.abs(g))))
        t = 1.0
        for _ in range(40):
            cand = theta + t * step
            fc = loglik(cand)
            if np.isfinite(fc) and fc >= f - 1e-12 * max(1.0, abs(f)):
                break
            t *= 0.5
        else:
            break
        theta, f = cand, fc
        if float(np.max(np.abs(t * step))) < tol and float(np.max(np.abs(g))) < 1e-4:
            converged = True
            break
    g = score(theta)
    return theta, f, converged, it, g


def fit_hhg(
    model: str,
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    offset: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-10,
    beta_start: Optional[np.ndarray] = None,
) -> HHGFit:
    """Fit the HHG conditional-FE (``model='fe'``) or beta-RE (``'re'``) NB."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    _, gidx_all = np.unique(groups, return_inverse=True)
    n_all = int(gidx_all.max()) + 1

    keep = np.ones(len(y), dtype=bool)
    n_dropped = 0
    if model == "fe":
        counts = np.bincount(gidx_all, minlength=n_all)
        ysum = np.bincount(gidx_all, weights=y, minlength=n_all)
        bad = (counts < 2) | (ysum == 0)
        n_dropped = int(bad.sum())
        keep = ~bad[gidx_all]
    if not keep.any():
        raise DataInsufficient(
            "xtnbreg: no panel has at least two observations and a positive "
            "outcome total.",
            diagnostics={"n_groups": n_all},
        )
    y_k, X_k, off_k = y[keep], X[keep], off[keep]
    _, gidx = np.unique(np.asarray(groups)[keep], return_inverse=True)
    n_groups = int(gidx.max()) + 1

    args = (X_k, y_k, off_k, gidx, n_groups)
    if beta_start is None:
        beta_start = np.zeros(X_k.shape[1])
    if model == "fe":
        ll = lambda th: fe_loglik(th, *args)  # noqa: E731
        sc = lambda th: fe_score(th, *args)  # noqa: E731
        theta0 = np.asarray(beta_start, float)
    elif model == "re":
        ll = lambda th: re_loglik(th, *args)  # noqa: E731
        sc = lambda th: re_score(th, *args)  # noqa: E731
        theta0 = np.concatenate([np.asarray(beta_start, float), [0.0, 0.0]])
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(model)

    theta, f, converged, it, g = _newton(ll, sc, theta0, maxiter, tol)
    H = _jacobian(sc, theta)
    try:
        vcov = np.linalg.inv(-H)
    except np.linalg.LinAlgError as exc:
        raise ConvergenceFailure(
            "xtnbreg: the Hessian is singular at the optimum.",
            diagnostics={"model": model},
        ) from exc
    return HHGFit(
        params=theta,
        vcov=vcov,
        loglik=f,
        converged=converged,
        iterations=it,
        gradient_norm=float(np.max(np.abs(g))),
        keep=keep,
        n_groups=n_groups,
        n_dropped_groups=n_dropped,
    )
