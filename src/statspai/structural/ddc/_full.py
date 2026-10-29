"""Rust's third stage: the full likelihood over utilities and increments.

The full likelihood of a panel is the choice likelihood plus the
likelihood of the observed state increments,

    l_f(theta, p) = sum_t log P(a_t | s_t; theta, p) + sum_t log p(d_t),

where ``p`` is the increment distribution (Rust's ``theta_3``) and the
choice probabilities depend on ``p`` through the value function. The free
increment parameters are the probabilities of every observed increment
but the largest, whose probability is one minus their sum; increments
never observed have probability zero and stay there.

Scores in ``p`` follow from the implicit function theorem, as for the
utility parameters: with ``F = sum_d p_d G_d``,

    dV/dq_k = (I - beta F^P)^{-1} beta sum_a P_a (dF_a/dq_k) V,
    dv(s, a)/dq_k = beta ((dF_a/dq_k) V)(s) + beta (F_a dV/dq_k)(s).

At ``beta = 0`` the choice probabilities do not depend on ``p`` and the
information matrix is block diagonal, so the full and two-step estimates
and standard errors coincide.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import DataInsufficient, MethodIncompatibility
from ._core import (
    centred_scores,
    controlled_transition,
    loglik_from_P,
    solve_value,
    value_derivative,
)
from ._data import increments_matrix

__all__ = ["IncrementModel", "row_increments", "full_mle", "full_vcov"]


def row_increments(enc: Dict[str, Any], reset_idx: List[int]) -> np.ndarray:
    """Observed increment attached to each row (NaN where unobserved).

    From the ``increment`` column when given (Rust's convention: the step
    into the row); otherwise the step out of the row to the unit's next
    period, measured from 0 after a reset action.
    """
    if enc.get("increment") is not None:
        inc = np.asarray(enc["increment"], dtype=float)
    else:
        nxt = enc["next_state"]
        if nxt is None:
            raise MethodIncompatibility(
                "stage='full' needs the observed state increments",
                recovery_hint=(
                    "pass id= (and time=) so increments come from the panel, "
                    "or increment=<column of observed increments>"
                ),
            )
        reset = np.isin(enc["a"], np.asarray(reset_idx, dtype=np.int64))
        base = np.where(reset, 0, enc["s"])
        inc = np.where(nxt >= 0, nxt - base, np.nan).astype(float)
    obs = inc[np.isfinite(inc)]
    if obs.size == 0:
        raise DataInsufficient(
            "no observed state increments",
            recovery_hint="each unit needs at least two consecutive periods",
        )
    if np.any(obs < 0) or np.any(np.mod(obs, 1) != 0):
        raise MethodIncompatibility(
            "state increments must be non-negative integers",
            recovery_hint=(
                "the increment model assumes the state only moves up "
                "(restarting at 0 after a reset action)"
            ),
        )
    return inc


class IncrementModel:
    """Increment distribution with free, dependent and zero entries."""

    def __init__(self, inc: np.ndarray, S: int, J: int, reset_idx: List[int]):
        obs = inc[np.isfinite(inc)].astype(np.int64)
        self.n = np.bincount(obs).astype(float)
        D = self.n.size
        support = np.flatnonzero(self.n > 0)
        if support.size < 2:
            raise MethodIncompatibility(
                "only one increment value is observed; its probability is one "
                "and there is nothing to estimate",
                recovery_hint="use stage='partial'",
            )
        self.free = support[:-1]
        self.dep = int(support[-1])
        self.D = D
        self.G = np.stack(
            [increments_matrix(np.eye(D)[d], S, J, reset_idx) for d in range(D)]
        )
        self.dF = self.G[self.free] - self.G[self.dep][None]

    @property
    def names(self) -> List[str]:
        return [f"p_inc_{d}" for d in self.free]

    def probs(self, q: np.ndarray) -> np.ndarray:
        p = np.zeros(self.D)
        p[self.free] = q
        p[self.dep] = 1.0 - float(np.sum(q))
        return p

    def frequencies(self) -> np.ndarray:
        return np.asarray(self.n[self.free] / self.n.sum())

    def transitions(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(np.einsum("d,djst->jst", self.probs(q), self.G))

    def loglik(self, q: np.ndarray) -> float:
        p = self.probs(q)
        m = self.n > 0
        return float(np.sum(self.n[m] * np.log(p[m])))

    def grad(self, q: np.ndarray) -> np.ndarray:
        p = self.probs(q)
        return np.asarray(
            self.n[self.free] / p[self.free] - self.n[self.dep] / p[self.dep]
        )

    def row_scores(self, inc: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Per-row score of ``log p(d_t)`` in the free parameters."""
        p = self.probs(q)
        out = np.zeros((inc.size, self.free.size))
        ok = np.isfinite(inc)
        d = inc[ok].astype(np.int64)
        sc = np.zeros((d.size, self.free.size))
        for k, f in enumerate(self.free):
            sc[:, k] = (d == f) / p[f] - (d == self.dep) / p[self.dep]
        out[ok] = sc
        return out

    def feasible(self, q: np.ndarray) -> bool:
        p = self.probs(q)
        return bool(np.all(p[self.n > 0] > 0))


def _choice_part(
    phi: np.ndarray,
    model: IncrementModel,
    beta: float,
    theta: np.ndarray,
    q: np.ndarray,
    counts: np.ndarray,
    V0: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Choice log-likelihood, per-(s, a) scores in (theta, q), P and V."""
    F = model.transitions(q)
    V, v, P, _ = solve_value(phi @ theta, F, beta, V0=V0)
    dv_theta = value_derivative(phi, F, beta, P)
    S = P.shape[0]
    M = np.eye(S) - beta * controlled_transition(P, F)
    w = np.einsum("mjst,t->sjm", model.dF, V)
    dV = np.linalg.solve(M, beta * np.einsum("sj,sjm->sm", P, w))
    dv_q = beta * w + beta * np.einsum("jst,tm->sjm", F, dV)
    g = centred_scores(np.concatenate([dv_theta, dv_q], axis=2), P)
    return loglik_from_P(counts, P), g, P, V


def full_loglik(
    phi: np.ndarray,
    model: IncrementModel,
    beta: float,
    z: np.ndarray,
    counts: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Full log-likelihood and its gradient in ``z = (theta, q)``."""
    K = phi.shape[2]
    theta, q = z[:K], z[K:]
    llc, g, _, _ = _choice_part(phi, model, beta, theta, q, counts)
    grad = np.einsum("sj,sjk->k", counts, g)
    grad[K:] += model.grad(q)
    return llc + model.loglik(q), np.asarray(grad)


def _hessian(grad_fn: Any, z: np.ndarray) -> np.ndarray:
    n = z.size
    H = np.empty((n, n))
    for k in range(n):
        h = 1e-5 * max(1.0, abs(z[k]))
        e = np.zeros(n)
        e[k] = h
        H[:, k] = (grad_fn(z + e) - grad_fn(z - e)) / (2 * h)
    return 0.5 * (H + H.T)


def full_mle(
    phi: np.ndarray,
    model: IncrementModel,
    beta: float,
    counts: np.ndarray,
    theta0: np.ndarray,
    tol: float,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """Newton ascent of the full likelihood from the two-step estimate.

    Returns ``(theta, q, converged, iterations)``.
    """
    K = phi.shape[2]
    z = np.r_[theta0, model.frequencies()]

    def f(zz: np.ndarray) -> float:
        if not model.feasible(zz[K:]):
            return -np.inf
        return full_loglik(phi, model, beta, zz, counts)[0]

    def grad_fn(zz: np.ndarray) -> np.ndarray:
        return full_loglik(phi, model, beta, zz, counts)[1]

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g = grad_fn(z)
        H = _hessian(grad_fn, z)
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        dec = float(g @ step)
        f0 = f(z)
        if abs(dec) <= 1e-10 * max(1.0, abs(f0)):
            # predicted gain at rounding level: this is the maximum. Take the
            # last Newton step unless it lowers f beyond rounding; its gain is
            # below the rounding of f (~1e-12 |f|), so an exact ">= f0" test
            # kept or dropped it by platform (gradient 3e-5 left on Linux).
            if dec > 0 and f(z + step) >= f0 - 1e-12 * max(1.0, abs(f0)):
                z = z + step
            converged = True
            break
        if dec < 0:  # not an ascent direction: Hessian indefinite here
            break
        t = 1.0
        while t > 1e-10 and f(z + t * step) < f0 - 1e-12 * max(1.0, abs(f0)):
            t *= 0.5
        z = z + t * step
        if np.max(np.abs(t * step)) <= tol * max(1.0, np.max(np.abs(z))):
            converged = True
            break
    return z[:K], z[K:], converged, it


def full_vcov(
    vce: str,
    phi: np.ndarray,
    model: IncrementModel,
    beta: float,
    theta: np.ndarray,
    q: np.ndarray,
    counts: np.ndarray,
    enc: Dict[str, Any],
    inc: np.ndarray,
) -> np.ndarray:
    """Variance of ``(theta, q)`` from the full likelihood."""
    K = theta.size
    z = np.r_[theta, q]

    def grad_fn(zz: np.ndarray) -> np.ndarray:
        return full_loglik(phi, model, beta, zz, counts)[1]

    if vce == "hessian":
        return np.asarray(np.linalg.inv(-_hessian(grad_fn, z)))
    _, g, _, _ = _choice_part(phi, model, beta, theta, q, counts)
    obs = g[enc["s"], enc["a"]].copy()
    obs[:, K:] += model.row_scores(inc, q)
    if vce == "opg":
        return np.asarray(np.linalg.inv(obs.T @ obs))
    if enc["ids"] is not None:
        codes = pd.factorize(enc["ids"])[0]
        sc = np.zeros((codes.max() + 1, obs.shape[1]))
        np.add.at(sc, codes, obs)
    else:
        sc = obs
    Ainv = np.linalg.inv(-_hessian(grad_fn, z))
    return np.asarray(Ainv @ (sc.T @ sc) @ Ainv)
