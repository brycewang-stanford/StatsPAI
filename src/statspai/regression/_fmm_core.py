"""Likelihood, EM start values and Newton-Raphson for ``sp.fmm``.

The model is a finite mixture of ``K`` regressions that share one design
``X``,

    f(y_i | x_i, z_i) = sum_k pi_k(z_i) f_k(y_i | x_i' b_k),

with class probabilities from a multinomial logit in ``z_i`` (class 1 is
the base, its coefficients are zero) and a Gaussian, logit or Poisson
class density. This is Stata's ``fmm k, lcprob(z): regress | logit |
poisson y x``.

The parameter vector stacks the class-membership coefficients of classes
``2..K`` (``q`` each), the outcome coefficients of every class (``p``
each), and for the Gaussian family the log standard deviation of every
class. The per-observation scores are analytic; every operation is
complex-safe, so the Hessian is the complex-step derivative of the total
score, exact to rounding. EM supplies the start; Newton-Raphson on the full
likelihood takes it to the maximum.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from scipy import special

from ..exceptions import ConvergenceFailure, NumericalInstability

FAMILIES = ("gaussian", "logit", "poisson")
_LOG_2PI = float(np.log(2.0 * np.pi))


def _lse(a: np.ndarray) -> np.ndarray:
    """Row-wise log-sum-exp, complex-safe (the shift uses the real part)."""
    m = np.max(np.real(a), axis=1, keepdims=True)
    return cast(
        np.ndarray, (m + np.log(np.sum(np.exp(a - m), axis=1, keepdims=True)))[:, 0]
    )


def _softplus(eta: np.ndarray) -> np.ndarray:
    """``log(1 + exp(eta))``, stable and complex-safe."""
    if np.isrealobj(eta):
        return cast(np.ndarray, np.logaddexp(0.0, eta))
    pos = np.real(eta) > 0
    with np.errstate(over="ignore", invalid="ignore"):
        a = eta + np.log1p(np.exp(-eta))
        b = np.log1p(np.exp(eta))
    return np.where(pos, a, b)


class MixtureModel:
    """Data and parameter layout of one finite mixture regression."""

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        K: int,
        family: str,
        w: np.ndarray,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.K = int(K)
        self.family = family
        self.n, self.p = self.X.shape
        self.q = self.Z.shape[1]
        self.n_gamma = (self.K - 1) * self.q
        self.n_beta = self.K * self.p
        self.n_scale = self.K if family == "gaussian" else 0
        self.size = self.n_gamma + self.n_beta + self.n_scale
        if family == "poisson":
            self._lgy = special.gammaln(self.y + 1.0)
        self.y_scale = float(np.var(self.y)) if family == "gaussian" else 1.0

    # -- layout ---------------------------------------------------------
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(G, B, lns)``: ``G`` is ``q x K`` with a zero base column."""
        G = np.zeros((self.q, self.K), dtype=theta.dtype)
        if self.K > 1:
            G[:, 1:] = theta[: self.n_gamma].reshape(self.K - 1, self.q).T
        B = theta[self.n_gamma : self.n_gamma + self.n_beta].reshape(self.K, self.p).T
        lns = theta[self.n_gamma + self.n_beta :]
        return G, B, lns

    def pack(
        self, G: np.ndarray, B: np.ndarray, lns: Optional[np.ndarray]
    ) -> np.ndarray:
        G = np.asarray(G, dtype=float)
        G = G - G[:, [0]]
        parts = [G[:, 1:].T.ravel(), np.asarray(B, dtype=float).T.ravel()]
        if self.n_scale:
            parts.append(np.asarray(lns, dtype=float).ravel())
        return np.concatenate(parts)

    def permute(self, theta: np.ndarray, order: np.ndarray) -> np.ndarray:
        """The same point with classes relabelled: new class ``j`` is old
        class ``order[j]``; the new class 1 becomes the base."""
        G, B, lns = self.unpack(theta)
        return self.pack(G[:, order], B[:, order], lns[order] if self.n_scale else None)

    # -- likelihood pieces ---------------------------------------------
    def class_log_prior(
        self, theta: np.ndarray, Z: Optional[np.ndarray] = None
    ) -> np.ndarray:
        G, _, _ = self.unpack(theta)
        eta = (self.Z if Z is None else Z) @ G
        return cast(np.ndarray, eta - _lse(eta)[:, None])

    def class_mean(
        self, theta: np.ndarray, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """``n x K`` class-specific mean of the outcome."""
        _, B, _ = self.unpack(theta)
        eta = (self.X if X is None else X) @ B
        if self.family == "gaussian":
            return cast(np.ndarray, eta)
        if self.family == "logit":
            return cast(np.ndarray, np.exp(eta - _softplus(eta)))
        return cast(np.ndarray, np.exp(eta))

    def _log_density(
        self, theta: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """``log f_ik``, the derivative of it in ``x_i' b_k`` and, for the
        Gaussian family, the standardised squared residual."""
        _, B, lns = self.unpack(theta)
        eta = X @ B
        yc = y[:, None]
        if self.family == "gaussian":
            s2 = np.exp(2.0 * lns)[None, :]
            r = yc - eta
            q2 = r * r / s2
            logf = -0.5 * _LOG_2PI - lns[None, :] - 0.5 * q2
            return logf, r / s2, q2
        if self.family == "logit":
            sp_ = _softplus(eta)
            return yc * eta - sp_, yc - np.exp(eta - sp_), None
        mu = np.exp(eta)
        lgy = self._lgy if y is self.y else special.gammaln(y + 1.0)
        return yc * eta - mu - lgy[:, None], yc - mu, None

    def joint(
        self,
        theta: np.ndarray,
        X: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """``(l_i, log prior, posterior, d log f / d eta, q2)``."""
        X = self.X if X is None else X
        y = self.y if y is None else y
        log_pi = self.class_log_prior(theta, Z)
        logf, u, q2 = self._log_density(theta, X, y)
        a = log_pi + logf
        ll = _lse(a)
        post = np.exp(a - ll[:, None])
        return ll, log_pi, post, u, q2

    def obs_loglik(self, theta: np.ndarray) -> np.ndarray:
        return self.joint(theta)[0]

    def loglik(self, theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            v = float(np.real(np.sum(self.w * self.obs_loglik(theta))))
        return v if np.isfinite(v) else -np.inf

    def scores(self, theta: np.ndarray) -> np.ndarray:
        """``n x size`` per-observation (unweighted) scores."""
        _, log_pi, post, u, q2 = self.joint(theta)
        cols = []
        if self.K > 1:
            d = post - np.exp(log_pi)
            for k in range(1, self.K):
                cols.append(d[:, [k]] * self.Z)
        for k in range(self.K):
            cols.append((post[:, k] * u[:, k])[:, None] * self.X)
        if self.n_scale:
            cols.append(post * (q2 - 1.0))
        return np.hstack(cols)

    def total_score(self, theta: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, self.w @ self.scores(theta))

    def hessian(self, theta: np.ndarray, h: float = 1e-30) -> np.ndarray:
        """Hessian of the weighted log-likelihood by the complex step of the
        analytic total score (no subtractive cancellation)."""
        theta = np.asarray(theta, dtype=float)
        H = np.empty((self.size, self.size))
        for j in range(self.size):
            tc = theta.astype(complex)
            tc[j] += 1j * h
            H[:, j] = np.imag(self.total_score(tc)) / h
        return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# EM for start values
# ---------------------------------------------------------------------------


def _weighted_mlogit_step(
    Z: np.ndarray, R: np.ndarray, w: np.ndarray, G: np.ndarray, steps: int = 3
) -> np.ndarray:
    """A few Newton steps of the multinomial logit with fractional outcomes
    ``R`` (rows sum to one) and weights ``w``; class 1 is the base."""
    n, q = Z.shape
    K = R.shape[1]
    G = G.copy()
    for _ in range(steps):
        eta = Z @ G
        P = np.exp(eta - _lse(eta)[:, None])
        g = ((w[:, None] * (R - P))[:, 1:].T @ Z).ravel()
        m = (K - 1) * q
        H = np.zeros((m, m))
        for a in range(1, K):
            for b in range(a, K):
                c = w * P[:, a] * ((1.0 if a == b else 0.0) - P[:, b])
                blk = -((Z * c[:, None]).T @ Z)
                H[(a - 1) * q : a * q, (b - 1) * q : b * q] = blk
                H[(b - 1) * q : b * q, (a - 1) * q : a * q] = blk.T
        try:
            step = np.linalg.solve(-H + 1e-8 * np.eye(m), g)
        except np.linalg.LinAlgError:
            break
        big = float(np.max(np.abs(step)))
        if big > 5.0:
            step *= 5.0 / big
        G[:, 1:] += step.reshape(K - 1, q).T
        if big < 1e-10:
            break
    return G


def _m_step_outcome(
    model: MixtureModel, Wk: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, Optional[float]]:
    """Weighted M-step of one class: WLS (Gaussian) or IRLS steps."""
    X, y = model.X, model.y
    if model.family == "gaussian":
        XtW = X.T * Wk
        b_new = np.linalg.solve(XtW @ X + 1e-12 * np.eye(model.p), XtW @ y)
        r = y - X @ b_new
        s2 = float(np.sum(Wk * r * r) / np.sum(Wk))
        return b_new, s2
    b = b.copy()
    for _ in range(3):
        eta = X @ b
        if model.family == "logit":
            mu = special.expit(eta)
            v = mu * (1.0 - mu)
        else:
            mu = np.exp(np.minimum(eta, 50.0))
            v = mu
        g = X.T @ (Wk * (y - mu))
        info = (X.T * (Wk * v)) @ X + 1e-8 * np.eye(model.p)
        step = np.linalg.solve(info, g)
        big = float(np.max(np.abs(step)))
        if big > 3.0:
            step *= 3.0 / big
        b = b + step
        if big < 1e-10:
            break
    return b, None


def em(
    model: MixtureModel, R: np.ndarray, max_iter: int, tol: float
) -> Tuple[np.ndarray, int]:
    """EM (one or a few Newton steps in the non-Gaussian M-steps, i.e. a
    generalised EM) from the posterior matrix ``R``. Returns the point and
    the number of iterations."""
    K, w = model.K, model.w
    G = np.zeros((model.q, K))
    B = np.zeros((model.p, K))
    lns = np.zeros(K)
    floor = 1e-10 * model.y_scale
    ll_old = -np.inf
    theta = None
    it = 0
    for it in range(1, max_iter + 1):
        Wr = w[:, None] * R
        nk = Wr.sum(axis=0)
        if np.any(nk <= 1e-8 * w.sum()):
            raise NumericalInstability(
                "fmm: a class lost all of its observations during EM.",
                recovery_hint="Use fewer classes or more starts.",
                diagnostics={"class_counts": nk.tolist()},
            )
        if model.q == 1:
            share = nk / nk.sum()
            G[0, :] = np.log(share) - np.log(share[0])
        else:
            G = _weighted_mlogit_step(model.Z, R, w, G)
        for k in range(K):
            B[:, k], s2 = _m_step_outcome(model, Wr[:, k], B[:, k])
            if s2 is not None:
                if not np.isfinite(s2) or s2 <= floor:
                    raise NumericalInstability(
                        f"fmm: the variance of class {k + 1} collapsed to zero "
                        "during EM.",
                        recovery_hint="Use fewer classes or more starts.",
                        diagnostics={"class": k + 1, "variance": s2},
                    )
                lns[k] = 0.5 * np.log(s2)
        theta = model.pack(G, B, lns if model.n_scale else None)
        with np.errstate(all="ignore"):
            ll_i, _, R, _, _ = model.joint(theta)
        ll = float(np.sum(w * ll_i))
        if not np.isfinite(ll):
            raise NumericalInstability(
                "fmm: the log-likelihood is not finite during EM.",
                recovery_hint="Check the outcome for the family, or use fewer classes.",
            )
        if abs(ll - ll_old) <= tol * (1.0 + abs(ll)):
            break
        ll_old = ll
    assert theta is not None
    return theta, it


# ---------------------------------------------------------------------------
# Newton-Raphson on the full likelihood
# ---------------------------------------------------------------------------


def newton(
    model: MixtureModel, theta: np.ndarray, max_iter: int, tol: float
) -> Tuple[np.ndarray, float, np.ndarray, int, Dict[str, Any]]:
    """Maximise the weighted log-likelihood from ``theta``.

    Each step solves ``(-H) d = g``; where ``-H`` is not positive definite
    its eigenvalues are floored (a Levenberg-Marquardt-type modification),
    and the step is halved until the log-likelihood does not fall. The
    iteration stops after a step below ``tol * (1 + max|theta|)`` taken
    with a positive definite ``-H``. Returns ``(theta, ll, H, iterations,
    info)`` with ``info['nr']`` the scaled gradient ``g'(-H)^{-1}g`` (Stata's
    ``nrtolerance`` statistic) at the returned point.
    """
    theta = np.asarray(theta, dtype=float).copy()
    ll = model.loglik(theta)
    if not np.isfinite(ll):
        raise NumericalInstability("fmm: the start value has a non-finite likelihood.")
    floor = 1e-10 * model.y_scale
    it = 0
    for it in range(1, max_iter + 1):
        g = model.total_score(theta)
        H = model.hessian(theta)
        A = -H
        evals, evecs = np.linalg.eigh(A)
        pd_ = bool(evals.min() > 1e-12 * max(1.0, float(np.abs(evals).max())))
        if pd_:
            step = np.linalg.solve(A, g)
        else:
            lam = np.maximum(np.abs(evals), 1e-6 * max(1.0, float(np.abs(evals).max())))
            step = evecs @ ((evecs.T @ g) / lam)
        big = float(np.max(np.abs(step)))
        if big > 10.0:
            step *= 10.0 / big
        t = 1.0
        for _ in range(50):
            cand = theta + t * step
            ll_c = model.loglik(cand)
            if np.isfinite(ll_c) and ll_c >= ll - 1e-13 * abs(ll):
                break
            t *= 0.5
        else:
            break
        theta, ll = cand, ll_c
        if model.n_scale:
            s2 = np.exp(2.0 * model.unpack(theta)[2])
            if np.any(s2 <= floor):
                k = int(np.argmin(s2))
                raise NumericalInstability(
                    f"fmm: the variance of class {k + 1} collapses to zero; "
                    "the likelihood is unbounded there.",
                    recovery_hint="Use fewer classes or more starts.",
                    diagnostics={"class": k + 1, "variance": float(s2[k])},
                )
        if pd_ and t * big < tol * (1.0 + float(np.max(np.abs(theta)))):
            break
    g = model.total_score(theta)
    H = model.hessian(theta)
    try:
        nr = float(g @ np.linalg.solve(-H, g))
    except np.linalg.LinAlgError:
        nr = float("inf")
    return theta, ll, H, it, {"nr": nr}


def initial_posterior(
    model: MixtureModel, rng: Optional[np.random.Generator]
) -> np.ndarray:
    """Start posterior from a quantile split into ``K`` groups, softened so
    that no class is empty.

    The split variable is the pooled residual (Gaussian), ``log(1 + y)``
    (Poisson) or ``y`` (logit). With ``rng`` it is a random mixture of
    that variable and a random direction in the standardised covariates,
    so different starts split the sample along different directions.
    """
    n, K = model.n, model.K
    if model.family == "gaussian":
        Xw = model.X.T * model.w
        b = np.linalg.lstsq(Xw @ model.X, Xw @ model.y, rcond=None)[0]
        base = model.y - model.X @ b
    elif model.family == "poisson":
        base = np.log1p(model.y)
    else:
        base = model.y.copy()
    sd = float(np.std(base))
    base = (base - base.mean()) / (sd if sd > 0 else 1.0)
    Xs = model.X.std(axis=0)
    live = Xs > 0
    Xstd = (model.X[:, live] - model.X[:, live].mean(axis=0)) / Xs[live]
    if rng is None:
        score = base + 1e-3 * Xstd.sum(axis=1) if Xstd.size else base
    else:
        a = rng.normal(size=Xstd.shape[1]) if Xstd.size else np.zeros(0)
        c = rng.uniform(0.0, 2.0)
        score = c * base + (Xstd @ a if Xstd.size else 0.0) + 0.1 * rng.normal(size=n)
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    lab = np.minimum((ranks * K) // n, K - 1)
    R = np.full((n, K), 0.25 / K)
    R[np.arange(n), lab] += 0.75
    return R


def bfgs(
    model: MixtureModel, theta: np.ndarray, max_iter: int
) -> Tuple[np.ndarray, float]:
    """Quasi-Newton ascent with the analytic score (cheap per start)."""
    from scipy.optimize import minimize

    scale = max(1.0, float(model.w.sum()))

    def f(t: np.ndarray) -> float:
        v = model.loglik(t)
        return -v / scale if np.isfinite(v) else np.inf

    def g(t: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            out = -model.total_score(t) / scale
        return np.where(np.isfinite(out), out, 0.0)

    with np.errstate(all="ignore"):
        res = minimize(
            f, theta, jac=g, method="BFGS", options={"maxiter": max_iter, "gtol": 1e-9}
        )
    return np.asarray(res.x, dtype=float), model.loglik(res.x)


def require_converged(nr: float, function: str = "fmm") -> None:
    if not np.isfinite(nr) or nr > 1e-5:
        raise ConvergenceFailure(
            f"{function}: Newton-Raphson stopped with g'(-H)^-1 g = {nr:.3g} "
            "(Stata's nrtolerance criterion is 1e-5).",
            recovery_hint="Raise max_iter, use more starts, or fewer classes.",
            diagnostics={"nr": nr},
        )
