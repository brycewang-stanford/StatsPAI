"""Analytic Newton-Raphson engines for ordered and multinomial models.

The likelihood of observation ``i`` in category ``j`` is
``P_i = F(a_i) - F(b_i)`` with ``a_i = kappa_j - x_i'beta`` and
``b_i = kappa_{j-1} - x_i'beta`` (``kappa_0 = -inf``, ``kappa_J = +inf``).
Its score and Hessian are closed-form in ``F``, ``f`` and ``f'``, so the
fit is Newton-Raphson on ``theta = (beta, kappa)`` -- Stata's ``oprobit`` /
``ologit`` parameterisation -- with step halving that keeps the
log-likelihood increasing and the cutpoints ordered.

The engine it replaces ran BFGS on finite-difference gradients and then
built the Hessian by complex-step differentiation, ``O(p^2)`` likelihood
evaluations: with a few hundred dummies a fit took minutes (Busting the
Princelings, QJE 2019, ~300 prefecture dummies; 260 s at 100 dummies).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from scipy import special, stats

from ..exceptions import NumericalInstability


def _link_funcs(
    link: str,
) -> Tuple[Callable[[np.ndarray], np.ndarray], ...]:
    """(cdf, survival, pdf, pdf') for the latent error distribution."""
    if link == "probit":

        def dpdf(z: np.ndarray) -> np.ndarray:
            return -z * stats.norm.pdf(z)

        return special.ndtr, lambda z: special.ndtr(-z), stats.norm.pdf, dpdf
    if link == "logit":

        def pdf(z: np.ndarray) -> np.ndarray:
            e = special.expit(z)
            return e * (1.0 - e)

        def dpdf(z: np.ndarray) -> np.ndarray:
            e = special.expit(z)
            return e * (1.0 - e) * (1.0 - 2.0 * e)

        return special.expit, lambda z: special.expit(-z), pdf, dpdf
    raise ValueError("link must be 'logit' or 'probit'.")


class OrderedLikelihood:
    """Per-observation log-likelihood, scores and Hessian of an ordered model."""

    def __init__(self, X: np.ndarray, y_idx: np.ndarray, n_cats: int, link: str):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y_idx, dtype=int)
        self.J = int(n_cats)
        self.n, self.k = self.X.shape
        self.n_cuts = self.J - 1
        self.cdf, self.sf, self.pdf, self.dpdf = _link_funcs(link)
        self.has_upper = self.y < self.n_cuts  # a = kappa_j - xb is finite
        self.has_lower = self.y > 0  # b = kappa_{j-1} - xb is finite

    def _parts(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        beta, kappa = theta[: self.k], theta[self.k :]
        xb = self.X @ beta
        up, lo = self.has_upper, self.has_lower
        a = np.where(up, kappa[np.minimum(self.y, self.n_cuts - 1)] - xb, np.inf)
        b = np.where(lo, kappa[np.maximum(self.y - 1, 0)] - xb, -np.inf)
        # P = F(a) - F(b), computed on the tail that avoids cancellation:
        # when both points are in the upper half use S(b) - S(a).
        upper_half = b > 0
        P = np.where(
            upper_half,
            self.sf(b) - self.sf(a),
            self.cdf(a) - self.cdf(b),
        )
        fa = np.where(up, self.pdf(np.where(up, a, 0.0)), 0.0)
        fb = np.where(lo, self.pdf(np.where(lo, b, 0.0)), 0.0)
        dfa = np.where(up, self.dpdf(np.where(up, a, 0.0)), 0.0)
        dfb = np.where(lo, self.dpdf(np.where(lo, b, 0.0)), 0.0)
        return dict(P=P, fa=fa, fb=fb, dfa=dfa, dfb=dfb)

    def loglik_obs(self, theta: np.ndarray) -> np.ndarray:
        P = self._parts(theta)["P"]
        with np.errstate(divide="ignore"):
            return np.log(P)

    def loglik(self, theta: np.ndarray) -> float:
        kappa = theta[self.k :]
        if self.n_cuts > 1 and np.any(np.diff(kappa) <= 0):
            return -np.inf
        ll = self.loglik_obs(theta)
        return float(np.sum(ll)) if np.all(np.isfinite(ll)) else -np.inf

    def derivatives(self, theta: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """(log-likelihood, per-observation scores n x p, Hessian p x p)."""
        q = self._parts(theta)
        P, fa, fb = q["P"], q["fa"], q["fb"]
        la = fa / P  # d l / d a
        lb = -fb / P  # d l / d b
        laa = q["dfa"] / P - la * la
        lbb = -q["dfb"] / P - lb * lb
        lab = -la * lb  # = fa fb / P^2
        n, k, m = self.n, self.k, self.n_cuts
        rows = np.arange(n)
        ju = np.minimum(self.y, m - 1)
        jl = np.maximum(self.y - 1, 0)
        up, lo = self.has_upper, self.has_lower

        S = np.zeros((n, k + m))
        S[:, :k] = -(la + lb)[:, None] * self.X
        S[rows[up], k + ju[up]] += la[up]
        S[rows[lo], k + jl[lo]] += lb[lo]

        H = np.zeros((k + m, k + m))
        w_bb = laa + 2.0 * lab + lbb
        H[:k, :k] = (self.X * w_bb[:, None]).T @ self.X
        # beta x kappa blocks: d2l / d beta d kappa_upper = -x (laa + lab),
        # d2l / d beta d kappa_lower = -x (lab + lbb).
        wu = np.where(up, -(laa + lab), 0.0)
        wl = np.where(lo, -(lab + lbb), 0.0)
        cross = np.zeros((k, m))
        for c in range(m):
            sel_u = up & (ju == c)
            sel_l = lo & (jl == c)
            cross[:, c] = self.X[sel_u].T @ wu[sel_u] + self.X[sel_l].T @ wl[sel_l]
        H[:k, k:] = cross
        H[k:, :k] = cross.T
        for c in range(m):
            H[k + c, k + c] = np.sum(laa[up & (ju == c)]) + np.sum(lbb[lo & (jl == c)])
            if c + 1 < m:
                # category c+1 has upper cut c+1 and lower cut c
                off = np.sum(lab[self.y == c + 1])
                H[k + c, k + c + 1] = off
                H[k + c + 1, k + c] = off
        ll = float(np.sum(np.log(P)))
        return ll, S, H


def fit_ordered(
    lik: OrderedLikelihood,
    theta0: np.ndarray,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, int, bool]:
    """Newton-Raphson with step halving.

    Returns ``(theta, loglik, scores, Hessian, iterations, converged)``.
    Convergence is Stata's scaled-gradient rule ``g'(-H)^{-1}g < tol``
    (``nrtolerance``), checked at a point where ``-H`` is positive definite.
    """
    theta = np.asarray(theta0, dtype=float).copy()
    ll, S, H = lik.derivatives(theta)
    if not np.isfinite(ll):
        raise NumericalInstability(
            "ordered model: non-finite log-likelihood at the starting values"
        )
    converged = False
    polish = 0
    it = 0
    for it in range(1, maxiter + 1):
        g = S.sum(axis=0)
        negH = -H
        try:
            L = np.linalg.cholesky(negH)
            step = np.linalg.solve(L.T, np.linalg.solve(L, g))
        except np.linalg.LinAlgError:
            # Not concave here: a gradient step scaled by the diagonal.
            d = np.abs(np.diag(negH))
            step = g / np.where(d > 0, d, 1.0)
            L = None
        if L is not None and float(g @ step) < tol:
            converged = True
            # ``tol`` decides convergence; the estimate is then polished with
            # up to two more (quadratically convergent) steps, so it sits at
            # the optimum to machine precision rather than ~sqrt(tol) SEs
            # away from it.
            if float(g @ step) < 1e-24 or polish >= 2:
                break
            polish += 1
        t = 1.0
        for _ in range(60):
            cand = theta + t * step
            ll_c = lik.loglik(cand)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12 * abs(ll):
                break
            t *= 0.5
        else:
            break
        theta = cand
        ll, S, H = lik.derivatives(theta)
    if not converged:
        try:
            np.linalg.cholesky(-H)
            g = S.sum(axis=0)
            converged = bool(g @ np.linalg.solve(-H, g) < tol)
        except np.linalg.LinAlgError:
            converged = False
    return theta, ll, S, H, it, converged


def logit_newton(
    X: np.ndarray, y: np.ndarray, maxiter: int = 100, tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary logit by Newton-Raphson: (coefficients, inverse information)."""
    b = np.zeros(X.shape[1])
    for _ in range(maxiter):
        p = special.expit(X @ b)
        g = X.T @ (y - p)
        info = (X * (p * (1.0 - p))[:, None]).T @ X
        step = np.linalg.solve(info, g)
        b = b + step
        if float(g @ step) < tol:
            break
    p = special.expit(X @ b)
    info = (X * (p * (1.0 - p))[:, None]).T @ X
    return b, np.linalg.inv(info)


def mlogit_newton(
    X: np.ndarray,
    y_idx: np.ndarray,
    n_cats: int,
    base: int,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, int, bool]:
    """Multinomial logit by Newton-Raphson with the analytic Hessian.

    ``theta`` stacks the ``k`` coefficients of each non-base category.
    Score of observation ``i``: ``(1[y_i = a] - P_ia) x_i`` in block ``a``;
    Hessian block ``(a, b)``: ``-sum_i P_ia (1[a = b] - P_ib) x_i x_i'``.
    The log-likelihood is globally concave, so Newton from zero converges;
    step halving guards the first steps. Returns
    ``(theta, loglik, scores, Hessian, iterations, converged)``.
    """
    n, k = X.shape
    non_base = [j for j in range(n_cats) if j != base]
    m = len(non_base)
    Y = np.zeros((n, n_cats))
    Y[np.arange(n), y_idx] = 1.0

    def parts(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        V = np.zeros((n, n_cats))
        for a, j in enumerate(non_base):
            V[:, j] = X @ theta[a * k : (a + 1) * k]
        V -= V.max(axis=1, keepdims=True)
        logP = V - np.log(np.exp(V).sum(axis=1, keepdims=True))
        return float(np.sum(logP[np.arange(n), y_idx])), np.exp(logP)

    def derivatives(theta: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        ll, P = parts(theta)
        R = Y - P
        S = np.hstack([R[:, [j]] * X for j in non_base])
        H = np.zeros((m * k, m * k))
        for a, ja in enumerate(non_base):
            for b in range(a, m):
                jb = non_base[b]
                w = P[:, ja] * ((1.0 if a == b else 0.0) - P[:, jb])
                blk = -((X * w[:, None]).T @ X)
                H[a * k : (a + 1) * k, b * k : (b + 1) * k] = blk
                if b != a:
                    H[b * k : (b + 1) * k, a * k : (a + 1) * k] = blk.T
        return ll, S, H

    theta = np.zeros(m * k)
    ll, S, H = derivatives(theta)
    converged, polish, it = False, 0, 0
    for it in range(1, maxiter + 1):
        g = S.sum(axis=0)
        try:
            L = np.linalg.cholesky(-H)
        except np.linalg.LinAlgError:
            break
        step = np.linalg.solve(L.T, np.linalg.solve(L, g))
        crit = float(g @ step)
        if crit < tol:
            converged = True
            if crit < 1e-24 or polish >= 2:
                break
            polish += 1
        t = 1.0
        for _ in range(60):
            cand = theta + t * step
            ll_c = parts(cand)[0]
            if np.isfinite(ll_c) and ll_c >= ll - 1e-12 * abs(ll):
                break
            t *= 0.5
        else:
            break
        theta = cand
        ll, S, H = derivatives(theta)
    return theta, ll, S, H, it, converged
