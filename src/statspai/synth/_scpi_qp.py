"""Exact bound-and-sum constrained quadratic programme (private to scpi).

Clean-room implementation (see ``docs/dev/cleanroom/scpi.md``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..exceptions import ConvergenceFailure, MethodIncompatibility


def _sum_nullspace(m: int) -> np.ndarray:
    """Orthonormal basis (m x m-1) of {p : sum(p) = 0}."""
    if m <= 1:
        return np.zeros((m, 0))
    v = np.ones(m) / np.sqrt(m)
    v[0] += 1.0
    v /= np.linalg.norm(v)
    return (np.eye(m) - 2.0 * np.outer(v, v))[:, 1:]


def _equality_step(
    H: np.ndarray, g: np.ndarray, free: np.ndarray
) -> Tuple[np.ndarray, float, bool]:
    """Minimise 1/2 p'Hp + g'p over p supported on ``free`` with sum(p) = 0.

    Returns ``(p, nu, is_ray)``. When the reduced Hessian is singular and the
    reduced gradient has a component in its null space, ``p`` is a
    zero-curvature descent direction (``is_ray = True``); otherwise the
    (minimum-norm) Newton step. ``nu`` is the multiplier of the sum
    constraint at the Newton point: ``g_F + H_FF p + nu = 0``.
    """
    n = len(g)
    idx = np.flatnonzero(free)
    m = len(idx)
    p = np.zeros(n)
    if m == 0:
        return p, 0.0, False
    gF = g[idx]
    if m == 1:
        return p, -float(gF[0]), False
    Z = _sum_nullspace(m)
    HF = H[np.ix_(idx, idx)]
    R = Z.T @ HF @ Z
    R = 0.5 * (R + R.T)
    r = Z.T @ gF
    evals, evecs = np.linalg.eigh(R)
    scale = max(float(np.abs(evals).max()), 1e-300)
    pos = evals > 1e-11 * scale
    if not pos.all():
        null = evecs[:, ~pos]
        comp = null.T @ r
        if np.linalg.norm(comp) > 1e-11 * max(1.0, float(np.linalg.norm(r))):
            p[idx] = -(Z @ (null @ comp))
            return p, 0.0, True
    coef = (evecs[:, pos].T @ r) / evals[pos]
    pF = -(Z @ (evecs[:, pos] @ coef))
    p[idx] = pF
    nu = -float(np.mean(gF + HF @ pF))
    return p, nu, False


def qp_bounded_sum(
    H: np.ndarray,
    f: np.ndarray,
    lb: np.ndarray,
    total: float,
    max_iter: int = 500,
    free0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact primal active-set solve of min 1/2 w'Hw - f'w, w >= lb, 1'w = total.

    ``H`` is symmetric positive semi-definite (singular allowed); entries of
    ``lb`` may be ``-inf``. ``free0`` optionally names a starting working set
    (coordinates off their bounds); it is used when the corresponding point
    is feasible.
    """
    H = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T)
    f = np.asarray(f, float)
    lb = np.asarray(lb, float)
    finite = np.isfinite(lb)
    slack_room = float(total) - float(lb[finite].sum())
    if finite.all() and slack_room < -1e-12 * max(1.0, abs(total)):
        raise MethodIncompatibility(
            "Infeasible simplex problem: the lower bounds sum to more than the total.",
            recovery_hint="Lower the bounds or raise the total of the weights.",
            diagnostics={"sum_lb": float(lb.sum()), "total": float(total)},
        )
    w = np.where(finite, lb, 0.0)
    at_bound = finite.copy()
    started = False
    if free0 is not None and np.any(free0):
        fr = np.asarray(free0, bool) | ~finite
        w_try = w.copy()
        w_try[fr] += (total - w_try.sum()) / fr.sum()
        p, _, ray = _equality_step(H, H @ w_try - f, fr)
        if not ray:
            cand = w_try + p
            if np.all(
                cand[fr & finite]
                >= lb[fr & finite] - 1e-14 * (1 + np.abs(lb[fr & finite]))
            ):
                cand[fr & finite] = np.maximum(cand[fr & finite], lb[fr & finite])
                w, at_bound, started = cand, ~fr, True
    if not started:
        g = H @ w - f
        if (~finite).any():
            k = int(np.flatnonzero(~finite)[0])
        else:
            k = int(np.argmin(g))
        w[k] += float(total) - float(w.sum())
        at_bound[k] = False
    gscale = max(
        1.0,
        float(np.abs(f).max()),
        float(np.abs(H).max()) * max(1.0, float(np.abs(w).max())),
    )
    stationary = False
    for _ in range(max_iter):
        free = ~at_bound
        g = H @ w - f
        if not stationary:
            p, nu, ray = _equality_step(H, g, free)
            small = np.abs(p).max() <= 1e-15 * max(1.0, float(np.abs(w).max()))
            if ray or not small:
                alpha = np.inf if ray else 1.0
                dec = free & finite & (p < 0)
                block = -1
                if dec.any():
                    ratios = (lb[dec] - w[dec]) / p[dec]
                    i = int(np.argmin(ratios))
                    if ratios[i] < alpha:
                        alpha = max(float(ratios[i]), 0.0)
                        block = int(np.flatnonzero(dec)[i])
                if not np.isfinite(alpha):
                    raise ConvergenceFailure(
                        "The quadratic programme is unbounded below.",
                        recovery_hint="Check the constraint set; a bounded "
                        "feasible set is required.",
                    )
                w = w + alpha * p
                if block >= 0:
                    w[block] = lb[block]
                    at_bound[block] = True
                else:
                    stationary = True  # full Newton step: minimiser on this face
                continue
        # stationary on the current face: multipliers of the bounds
        idx_f = np.flatnonzero(free)
        nu = -float(np.mean(g[idx_f])) if len(idx_f) else 0.0
        lam = g + nu
        cand = at_bound & finite
        if not cand.any() or lam[cand].min() >= -1e-12 * gscale:
            return w
        j = int(np.flatnonzero(cand)[np.argmin(lam[cand])])
        at_bound[j] = False
        stationary = False
    raise ConvergenceFailure(
        f"Active-set QP did not converge in {max_iter} iterations.",
        recovery_hint="This is unexpected for a convex QP; please report the data.",
    )
