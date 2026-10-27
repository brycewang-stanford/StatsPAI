"""Numerical kernels shared by the NFXP, CCP and NPL estimators.

Notation, used throughout the package:

* ``S`` states, ``J`` actions, ``K`` utility parameters.
* ``phi``: array ``(S, J, K)``; flow utility ``u = phi @ theta``.
* ``F``: array ``(J, S, S)``; ``F[a, s, s']`` is the probability of moving
  from ``s`` to ``s'`` after action ``a``.
* ``V``: integrated (ex-ante) value function, ``V(s) = log sum_a exp v(s, a)``.
  Euler's constant is left out of ``V`` everywhere (it shifts every choice
  value by the same amount and never changes a choice probability).
* ``v``: choice-specific values ``v(s, a) = u(s, a) + beta * F[a] @ V``.
* ``counts``: array ``(S, J)`` of observed state-action frequencies; every
  likelihood here is ``sum_{s, a} counts[s, a] * log P(a | s)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import special

from ...exceptions import ConvergenceFailure, NumericalInstability


def choice_values(
    u: np.ndarray, F: np.ndarray, beta: float, V: np.ndarray
) -> np.ndarray:
    """``v(s, a) = u(s, a) + beta * (F[a] @ V)(s)``."""
    return np.asarray(u + beta * (F @ V).T)


def softmax_rows(v: np.ndarray) -> np.ndarray:
    return np.asarray(special.softmax(v, axis=1))


def controlled_transition(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Transition matrix of the state under choice probabilities ``P``."""
    return np.asarray(np.einsum("sj,jst->st", P, F))


def solve_value(
    u: np.ndarray,
    F: np.ndarray,
    beta: float,
    V0: np.ndarray | None = None,
    tol: float = 1e-11,
    n_contraction: int = 25,
    max_newton: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Fixed point of the smoothed Bellman operator.

    A few successive approximations, then Newton-Kantorovich steps
    ``V <- V - (I - beta F^P)^{-1} (V - T V)``, which converge
    quadratically for any ``beta < 1`` (the contraction alone needs about
    ``log(tol) / log(beta)`` steps, i.e. hundreds of thousands at
    ``beta = 0.9999``).

    Returns ``(V, v, P, n_newton_steps)``.
    """
    S = u.shape[0]
    V = np.zeros(S) if V0 is None else np.asarray(V0, dtype=float).copy()
    for _ in range(n_contraction):
        V_new = special.logsumexp(choice_values(u, F, beta, V), axis=1)
        step = np.max(np.abs(V_new - V))
        V = V_new
        if step <= tol * max(1.0, np.max(np.abs(V))):
            break
    eye = np.eye(S)
    for it in range(max_newton + 1):
        v = choice_values(u, F, beta, V)
        TV = special.logsumexp(v, axis=1)
        resid = V - TV
        if np.max(np.abs(resid)) <= tol * max(1.0, np.max(np.abs(V))):
            return TV, choice_values(u, F, beta, TV), softmax_rows(v), it
        if it == max_newton:
            break
        P = softmax_rows(v)
        V = V - np.linalg.solve(eye - beta * controlled_transition(P, F), resid)
        if not np.all(np.isfinite(V)):
            raise NumericalInstability(
                "the value function diverged in the Newton-Kantorovich step",
                recovery_hint=(
                    "rescale the utility features (e.g. cost_scale) so flow "
                    "utilities are of order one, or lower beta"
                ),
            )
    raise ConvergenceFailure(
        "the Bellman fixed point did not converge",
        recovery_hint="rescale the utility features or lower beta",
        diagnostics={"max_residual": float(np.max(np.abs(resid)))},
    )


def value_derivative(
    phi: np.ndarray, F: np.ndarray, beta: float, P: np.ndarray
) -> np.ndarray:
    """``d v(s, a) / d theta`` at the fixed point, shape ``(S, J, K)``.

    By the implicit function theorem
    ``dV/dtheta = (I - beta F^P)^{-1} sum_a P_a * phi_a``.
    """
    S = P.shape[0]
    M = np.eye(S) - beta * controlled_transition(P, F)
    dV = np.linalg.solve(M, np.einsum("sj,sjk->sk", P, phi))
    return np.asarray(phi + beta * np.einsum("jst,tk->sjk", F, dV))


def centred_scores(dv: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Score of ``log P(a | s)``: ``dv(s, a) - sum_b P(b | s) dv(s, b)``."""
    return np.asarray(dv - np.einsum("sj,sjk->sk", P, dv)[:, None, :])


def loglik_from_P(counts: np.ndarray, P: np.ndarray) -> float:
    mask = counts > 0
    return float(np.sum(counts[mask] * np.log(P[mask])))


def hm_valuation(
    phi: np.ndarray, F: np.ndarray, beta: float, P: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Hotz-Miller representation of choice values given CCPs ``P``.

    With ``V^P = (I - beta F^P)^{-1} sum_a P_a (u_a - log P_a)`` (linear in
    theta), the policy-iteration choice values are
    ``v(s, a) = Z[s, a] @ theta + e[s, a]``.

    Returns ``(Z, e)`` with shapes ``(S, J, K)`` and ``(S, J)``.
    """
    S = P.shape[0]
    M = np.eye(S) - beta * controlled_transition(P, F)
    rhs_A = np.einsum("sj,sjk->sk", P, phi)
    rhs_b = -np.sum(P * np.log(P), axis=1)
    sol = np.linalg.solve(M, np.column_stack([rhs_A, rhs_b]))
    A, b = sol[:, :-1], sol[:, -1]
    Z = phi + beta * np.einsum("jst,tk->sjk", F, A)
    e = beta * (F @ b).T
    return Z, e


def fit_linear_logit(
    Z: np.ndarray,
    e: np.ndarray,
    counts: np.ndarray,
    theta0: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[np.ndarray, bool, int]:
    """Maximise ``sum counts * log softmax(Z theta + e)`` by damped Newton.

    The objective is concave in theta (a conditional logit with offset).
    """
    theta = np.asarray(theta0, dtype=float).copy()
    n_s = counts.sum(axis=1)

    def objective(th: np.ndarray) -> float:
        v = Z @ th + e
        return float(np.sum(counts * (v - special.logsumexp(v, axis=1)[:, None])))

    f = objective(theta)
    for it in range(1, max_iter + 1):
        v = Z @ theta + e
        P = softmax_rows(v)
        Zbar = np.einsum("sj,sjk->sk", P, Z)
        D = Z - Zbar[:, None, :]
        grad = np.einsum("sj,sjk->k", counts, D)
        H = -np.einsum("s,sj,sjk,sjl->kl", n_s, P, D, D)
        try:
            step = -np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(H, grad, rcond=None)[0]
        dec = float(grad @ step)
        t = 1.0
        while True:
            f_new = objective(theta + t * step)
            if f_new >= f + 1e-4 * t * dec or t < 1e-10:
                break
            t *= 0.5
        theta = theta + t * step
        f = f_new
        if dec <= tol * max(1.0, abs(f)):
            return theta, True, it
    return theta, False, max_iter
