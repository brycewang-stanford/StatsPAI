"""Simulation constraint sets and exact fallbacks for ``sp.scpi`` (private).

Clean-room implementation (see ``docs/dev/cleanroom/scpi.md``); no
reference source was read. :func:`insample_solve` maps a constraint
geometry to the active-set method of :mod:`._scpi_insample`; when that does
not settle it identifies the face from an SLSQP guess and solves it in
closed form, then falls back to the exact Lagrangian-dual method.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import optimize

from ..exceptions import ConvergenceFailure, MethodIncompatibility
from ._scpi_insample import (
    _EPS,
    _is_point,
    _kkt_multipliers,
    _subproblem,
    insample_active_set,
)
from ._scpi_qp import qp_bounded_sum

# ---------------------------------------------------------------------- #
#  Exact dual method (used when the active-set method does not settle)
# ---------------------------------------------------------------------- #


def _dual_polyhedral(
    c: np.ndarray,
    M: np.ndarray,
    G: np.ndarray,
    k0: float,
    ell: np.ndarray,
    total: float,
) -> Union[None, str, np.ndarray]:
    """min c'y s.t. y'My - 2G'y + k0 <= 0, y >= ell, sum(y) = total.

    Lagrangian duality in the quadratic constraint: for ``t >= 0`` let
    ``y(t)`` minimise ``y'My - 2(G - t c)'y`` over the polyhedron (an exact
    QP). ``phi(t) = q(y(t))`` is non-decreasing in ``t`` and the optimum is
    ``y(t*)`` with ``phi(t*) = 0``, or the LP optimum when the quadratic
    constraint does not bind. ``t*`` is bracketed, found by a root search
    and polished on the final active set, where ``y(t)`` is affine in ``t``.
    """
    n = len(c)
    finite = np.isfinite(ell)
    scale = 1.0 + abs(float(k0)) + float(np.abs(G).sum())

    def y_of(t: float) -> np.ndarray:
        return qp_bounded_sum(2.0 * M, 2.0 * (G - t * c), ell, float(total))

    def phi(y: np.ndarray) -> float:
        return float(y @ M @ y - 2.0 * G @ y + k0)

    try:
        y0 = y_of(0.0)
    except ConvergenceFailure:
        return None
    if phi(y0) > 1e-10 * scale:
        return None
    if phi(y0) >= -1e-12 * scale:
        # the quadratic constraint holds only where q is minimal over the
        # polyhedron: the feasible set is that minimiser (degenerate case,
        # no Slater point)
        return y0
    lp = optimize.linprog(
        c,
        A_eq=np.ones((1, n)),
        b_eq=[float(total)],
        bounds=[(ell[i] if finite[i] else None, None) for i in range(n)],
        method="highs",
    )
    if lp.status == 0 and phi(lp.x) <= 0:
        return np.asarray(lp.x, dtype=float)
    t_hi = max(1.0, float(np.linalg.norm(G))) / max(float(np.linalg.norm(c)), 1e-300)
    for _ in range(200):
        try:
            if phi(y_of(t_hi)) >= 0:
                break
        except ConvergenceFailure:
            return "unbounded"
        t_hi *= 4.0
    else:
        return "unbounded"
    if phi(y_of(t_hi)) <= 0:  # pragma: no cover - guarded by the loop above
        return y_of(t_hi)
    t = optimize.brentq(
        lambda t_: phi(y_of(t_)), 0.0, t_hi, xtol=1e-300, rtol=4 * _EPS, maxiter=500
    )
    y = y_of(t)
    # polish on the active set at t
    free = ~finite | (
        y
        > np.where(finite, ell, 0.0)
        + 1e-12 * (1.0 + np.abs(np.where(finite, ell, 0.0)))
    )
    idx = np.flatnonzero(free)
    fix = np.flatnonzero(~free)
    m = len(idx)
    K = np.zeros((m + 1, m + 1))
    K[:m, :m] = 2.0 * M[np.ix_(idx, idx)]
    K[:m, m] = 1.0
    K[m, :m] = 1.0
    base = np.zeros(n)
    base[fix] = ell[fix]
    r0 = np.concatenate(
        [
            2.0 * G[idx] - 2.0 * M[np.ix_(idx, fix)] @ ell[fix],
            [float(total) - ell[fix].sum()],
        ]
    )
    r1 = np.concatenate([-2.0 * c[idx], [0.0]])
    try:
        sa = np.linalg.solve(K, r0)
        sb = np.linalg.solve(K, r1)
    except np.linalg.LinAlgError:
        return y
    a_full, b_full = base.copy(), np.zeros(n)
    a_full[idx], b_full[idx] = sa[:m], sb[:m]
    qa = float(b_full @ M @ b_full)
    qb = float(2.0 * a_full @ M @ b_full - 2.0 * G @ b_full)
    qc = phi(a_full)
    roots = (
        np.roots([qa, qb, qc])
        if abs(qa) > 0
        else np.array([-qc / qb]) if qb != 0 else np.array([])
    )
    roots = np.real(roots[np.abs(np.imag(roots)) <= 1e-12 * (1.0 + np.abs(roots))])
    if roots.size:
        tp = float(roots[np.argmin(np.abs(roots - t))])
        yp = a_full + tp * b_full
        ok = abs(tp - t) <= 1e-6 * max(1.0, t) and np.all(
            yp[finite] >= ell[finite] - 1e-12
        )
        if ok:
            return yp
    return y


def _face_polish(
    c: np.ndarray, M: np.ndarray, G: np.ndarray, geom: Dict[str, Any]
) -> Any:
    """Identify the active bounds from an approximate (SLSQP) solution, solve
    that face in closed form and accept it only if it satisfies the KKT
    conditions; ``None`` otherwise."""
    w = np.asarray(geom["w"], float)
    J = len(c)
    guess = insample_slsqp(c, M, G, geom)
    if guess is None:
        return None
    if geom["kind"] == "simplex":
        cc, MM, GG, k0 = c, M, G, 0.0
        ell = np.asarray(geom["lb"], float) - w
        Q2 = geom.get("Q2")
        ball = None if Q2 is None else (w, float(Q2))
        total, x_guess, back = 0.0, guess, None
    elif geom["kind"] == "l1":
        Q = float(geom["Q"])
        E = np.hstack([np.eye(J), -np.eye(J), np.zeros((J, 1))])
        x = w + guess
        x_guess = np.concatenate(
            [np.maximum(x, 0), np.maximum(-x, 0), [max(Q - np.abs(x).sum(), 0.0)]]
        )
        cc, MM, GG = E.T @ c, E.T @ M @ E, E.T @ (G + M @ w)
        k0 = float(w @ M @ w + 2 * G @ w)
        ell, ball, total, back = np.zeros(2 * J + 1), None, Q, E
    else:
        return None
    scale = 1.0 + float(np.abs(x_guess).max())
    fixed = x_guess <= ell + 1e-7 * scale
    if fixed.all():
        return None
    sol = _subproblem(cc, MM, GG, k0, ell, fixed, True, total, ball)
    if not _is_point(sol):
        return None
    y, mus = sol
    finite = np.isfinite(ell)
    if np.any(y[finite] < ell[finite] - 1e-12 * scale):
        return None
    y = np.where(finite & (y < ell), ell, y)
    grad_q = 2.0 * (MM @ y - GG)
    grads, active = [grad_q], [True]
    if ball is not None:
        grads.append(2.0 * (ball[0] + y))
        active.append(True)
    _, lmin = _kkt_multipliers(cc, grads, active, None, ~fixed, fixed & finite, True)
    if lmin < -1e-8 * (1.0 + float(np.abs(cc).max())):
        return None
    return y if back is None else back @ y - w


def _dual_solve(
    c: np.ndarray, M: np.ndarray, G: np.ndarray, geom: Dict[str, Any]
) -> Any:
    """Exact dual solution of :func:`insample_solve` problems with bounds."""
    w = np.asarray(geom["w"], float)
    J = len(c)
    if geom["kind"] == "simplex":
        ell = np.asarray(geom["lb"], float) - w
        Q2 = geom.get("Q2")
        if Q2 is None:
            return _dual_polyhedral(c, M, G, 0.0, ell, 0.0)
        R2 = float(Q2)

        def mixed(r: float) -> Any:
            if np.isinf(r):
                return _dual_polyhedral(
                    c, np.eye(J), -w, float(w @ w) - R2**2, ell, 0.0
                )
            return _dual_polyhedral(
                c, M + r * np.eye(J), G - r * w, r * (float(w @ w) - R2**2), ell, 0.0
            )

        return _ratio_search(
            mixed,
            lambda y: float(y @ M @ y - 2 * G @ y),
            lambda y: float(np.sum((w + y) ** 2) - R2**2),
        )
    if geom["kind"] == "l1":
        Q = float(geom["Q"])
        E = np.hstack([np.eye(J), -np.eye(J), np.zeros((J, 1))])
        res = _dual_polyhedral(
            E.T @ c,
            E.T @ M @ E,
            E.T @ (G + M @ w),
            float(w @ M @ w + 2 * G @ w),
            np.zeros(2 * J + 1),
            Q,
        )
        return res if res is None or isinstance(res, str) else E @ res - w
    return None


def _ratio_search(solve_mixed: Any, q_of: Any, ball_gap: Any) -> Any:
    """Two convex constraints ``q <= 0`` and ``b <= 0``: the solution with
    ``q + r b <= 0`` at the multiplier ratio ``r`` where ``b`` binds."""

    def ok(x: Any) -> bool:
        return x is not None and not isinstance(x, str)

    x_q = solve_mixed(0.0)
    if ok(x_q) and ball_gap(x_q) <= 1e-12:
        return x_q
    x_b = solve_mixed(np.inf)
    if x_b is None:
        return None
    if ok(x_b) and q_of(x_b) <= 1e-12:
        return x_b
    hi = 1.0
    for _ in range(2100):
        xh = solve_mixed(hi)
        if ok(xh) and ball_gap(xh) <= 0:
            break
        hi *= 2.0
    else:  # pragma: no cover
        return None

    def f(r: float) -> float:
        xr = solve_mixed(r)
        return float(ball_gap(xr)) if ok(xr) else 1.0

    lo = 0.0
    if not ok(x_q) or f(lo) <= 0:
        lo = hi / 2.0
        while lo > 1e-300 and f(lo) <= 0:
            lo /= 2.0
    r = optimize.brentq(f, lo, hi, xtol=1e-300, rtol=4 * _EPS, maxiter=500)
    return solve_mixed(r)


# ---------------------------------------------------------------------- #
#  Simulation constraint sets
# ---------------------------------------------------------------------- #


def insample_solve(
    c: np.ndarray, M: np.ndarray, G: np.ndarray, geom: Dict[str, Any]
) -> Any:
    """min c'y s.t. y'My - 2G'y <= 0 and y in the simulation constraint set.

    ``geom['kind']``: ``"none"`` (unrestricted), ``"simplex"``
    (``y >= lb - w``, ``sum(y) = 0``, and ``||w + y|| <= Q2`` when ``Q2`` is
    present), ``"l1"`` (``||w + y||_1 <= Q``) or ``"l2"``
    (``||w + y||_2 <= Q``); ``w`` are the estimated weights. Returns ``y``,
    ``"unbounded"``, or ``None`` when the problem could not be solved.
    """
    c = np.asarray(c, float)
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    G = np.asarray(G, float)
    w = np.asarray(geom["w"], float)
    J = len(c)
    kind = geom["kind"]
    free = np.full(J, -np.inf)
    if kind == "none":
        return insample_active_set(c, M, G, 0.0, free)
    if kind == "l2":
        return insample_active_set(c, M, G, 0.0, free, ball=(w, float(geom["Q"])))
    if kind == "simplex":
        ell = np.asarray(geom["lb"], float) - w
        Q2 = geom.get("Q2")
        ball = None if Q2 is None else (w, float(Q2))
        res = insample_active_set(c, M, G, 0.0, ell, "==", 0.0, ball)
        if res is None:
            res = _face_polish(c, M, G, geom)
        return res if res is not None else _dual_solve(c, M, G, geom)
    if kind == "l1":
        # w + y = pos - neg with pos, neg >= 0 and sum(pos + neg) <= Q
        Q = float(geom["Q"])
        E = np.hstack([np.eye(J), -np.eye(J)])
        x0 = np.concatenate([np.maximum(w, 0.0), np.maximum(-w, 0.0)])
        res = insample_active_set(
            E.T @ c,
            E.T @ M @ E,
            E.T @ (G + M @ w),
            float(w @ M @ w + 2 * G @ w),
            np.zeros(2 * J),
            "<=",
            Q,
            None,
            x0,
        )
        if res is None:
            res = _face_polish(c, M, G, geom)
            return res if res is not None else _dual_solve(c, M, G, geom)
        if isinstance(res, str):
            return res
        return E @ res - w
    raise MethodIncompatibility(f"Unknown constraint geometry {kind!r}.")


def geometry_from_constraint(
    w: np.ndarray,
    constr: Dict[str, Any],
    lb: Optional[np.ndarray] = None,
    Q1: Optional[float] = None,
    Q2: Optional[float] = None,
) -> Dict[str, Any]:
    """Simulation constraint set from a ``(p, dir)`` description.

    ``p = 'L1'`` with ``dir = '=='``: weights at least ``lb`` summing to
    their estimated total; ``dir = '<='``: the L1 ball of radius ``Q1``;
    ``p = 'L2'``: the L2 ball of radius ``Q1``; ``'L1-L2'``: the first set
    intersected with the L2 ball of radius ``Q2``; ``'no norm'``: none.
    """
    w = np.asarray(w, float)
    J = len(w)
    lb = np.full(J, -np.inf) if lb is None else np.asarray(lb, float)
    p, d = constr.get("p"), constr.get("dir")
    if p == "L1" and d == "==":
        return {"kind": "simplex", "w": w, "lb": lb}
    if p == "L1":
        return {"kind": "l1", "w": w, "Q": float(Q1 or 0.0)}
    if p == "L2":
        return {"kind": "l2", "w": w, "Q": float(Q1 or 0.0)}
    if p == "L1-L2":
        return {"kind": "simplex", "w": w, "lb": lb, "Q2": float(Q2 or 0.0)}
    return {"kind": "none", "w": w}


def insample_slsqp(
    c: np.ndarray,
    M: np.ndarray,
    G: np.ndarray,
    geom: Any,
    constr: Optional[Dict[str, Any]] = None,
    lb: Optional[np.ndarray] = None,
    Q1: Optional[float] = None,
    Q2: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Generic SLSQP solve of the in-sample problem (an independent check).

    ``geom`` is a geometry dict as for :func:`insample_solve`, or the
    estimated weights when ``constr`` (with ``lb``, ``Q1``, ``Q2``)
    describes the set as in :func:`geometry_from_constraint`.
    """
    if constr is not None:
        geom = geometry_from_constraint(geom, constr, lb, Q1, Q2)
    c = np.asarray(c, float)
    w = np.asarray(geom["w"], float)
    J = len(c)
    cons: List[Dict[str, Any]] = [
        {
            "type": "ineq",
            "fun": lambda y: -(y @ M @ y - 2 * G @ y),
            "jac": lambda y: -(2 * M @ y - 2 * G),
        }
    ]
    bounds = None
    kind = geom["kind"]
    if kind == "simplex":
        low = np.asarray(geom["lb"], float) - w
        bounds = [(low[j] if np.isfinite(low[j]) else None, None) for j in range(J)]
        cons.append(
            {"type": "eq", "fun": lambda y: y.sum(), "jac": lambda y: np.ones(J)}
        )
        if geom.get("Q2") is not None:
            R2 = float(geom["Q2"])
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda y: R2**2 - np.sum((w + y) ** 2),
                    "jac": lambda y: -2 * (w + y),
                }
            )
    elif kind == "l2":
        R = float(geom["Q"])
        cons.append(
            {
                "type": "ineq",
                "fun": lambda y: R**2 - np.sum((w + y) ** 2),
                "jac": lambda y: -2 * (w + y),
            }
        )
    elif kind == "l1":
        R1 = float(geom["Q"])
        cons.append({"type": "ineq", "fun": lambda y: R1 - np.abs(w + y).sum()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = optimize.minimize(
            lambda y: c @ y,
            np.zeros(J),
            jac=lambda y: c,
            constraints=cons,
            bounds=bounds,
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 2000},
        )
    if not np.all(np.isfinite(res.x)):
        return None
    return np.asarray(res.x, dtype=float)
