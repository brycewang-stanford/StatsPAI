"""In-sample simulation problems of ``sp.scpi`` (private).

Clean-room implementation (see ``docs/dev/cleanroom/scpi.md``); no
reference source was read.

For one Gaussian draw ``G`` of the score, a horizon with post-treatment donor
outcomes ``p`` and a side, the in-sample bound solves [@cattaneo2021prediction]

    min c'y  s.t.  y'M y - 2 G'y <= 0,  y in Delta*

with ``c = +/- p``, ``M = B'B / T0`` and ``y`` the deviation of the weights
from their estimate. ``Delta*`` is a polyhedron (lower bounds and a sum
constraint), possibly intersected with a Euclidean ball, or an L1 ball.

The problems are solved exactly:

* one convex quadratic constraint, linear objective: closed form in the
  eigenbasis (:func:`_min_linear_one_quad`);
* two (a ball added): the optimum of ``q1 + r q2 <= 0`` for the KKT ratio
  ``r`` of the two multipliers, found by a monotone 1-D root search
  (:func:`_min_linear_two_quads`);
* lower bounds and a sum constraint: a primal active-set method whose
  working-set subproblem is one of the two closed forms above restricted to
  an affine subspace (:func:`insample_active_set`).

The constraint sets of the simulation and the exact fallbacks are in
:mod:`._scpi_simulate`.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize

from ._scpi_qp import _sum_nullspace

_EPS = np.finfo(float).eps
Quad = Tuple[np.ndarray, np.ndarray, float]  # (M, b, k) for z'Mz + 2b'z + k


def _quad_val(z: np.ndarray, M: np.ndarray, b: np.ndarray, k: float) -> float:
    return float(z @ M @ z + 2 * b @ z + k)


# ---------------------------------------------------------------------- #
#  Closed forms
# ---------------------------------------------------------------------- #


def _min_linear_one_quad(g: Any, M: Any, b: Any, k: Any) -> Any:
    """min g'z s.t. z'Mz + 2b'z + k <= 0 for positive semi-definite M.

    Returns ``(z, mu)`` with ``g + 2 mu (M z + b) = 0`` at the optimum;
    ``("ray", d)`` when the objective decreases without bound along the
    feasible direction ``d`` (on which the constraint does not increase);
    ``("unbounded", None)`` when it is unbounded but not along a ray; or
    ``None`` when the set is empty.
    """
    g = np.asarray(g, float)
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    b = np.asarray(b, float)
    k = float(k)
    n = len(g)
    if n == 0:
        return (np.zeros(0), 0.0) if k <= 1e-12 * max(1.0, abs(k)) else None
    evals, V = np.linalg.eigh(M)
    scale = max(float(np.abs(evals).max()), 1e-300)
    pos = evals > 1e-11 * scale
    gt, bt = V.T @ g, V.T @ b
    tol_g = 1e-11 * max(float(np.linalg.norm(g)), 1e-300)
    tol_b = 1e-11 * max(float(np.linalg.norm(b)), float(np.sqrt(scale)), 1e-300)
    lam = evals[pos]
    gp, bp = gt[pos], bt[pos]
    zt = np.zeros(n)
    centre = -bp / lam
    p_min = k - float(np.sum(bp**2 / lam))  # smallest value on the positive block
    if (~pos).any():
        VN = V[:, ~pos]
        gN, bN = gt[~pos], bt[~pos]
        nb = float(np.linalg.norm(bN))
        if nb <= tol_b:
            if np.linalg.norm(gN) > tol_g:  # flat for the constraint, not the objective
                d = -(VN @ gN)
                return ("ray", d / np.linalg.norm(d))
        else:
            t = float(gN @ bN) / nb**2
            perp = gN - t * bN
            if np.linalg.norm(perp) > tol_g:
                d = -(VN @ perp)
                return ("ray", d / np.linalg.norm(d))
            if t > tol_g / nb:  # objective and constraint both fall along -b_N
                d = -(VN @ bN)
                return ("ray", d / np.linalg.norm(d))
            if t >= -tol_g / nb:  # g_N = 0: the set is unbounded in the positive block
                if np.linalg.norm(gp) > tol_g:
                    return ("unbounded", None)
                zt[pos] = centre
                zt[~pos] = (-p_min / 2.0) * bN / nb**2
                return V @ zt, 0.0
            # g_N = t b_N with t < 0: mu = -t/2; the positive block solves
            # g_P + 2 mu (Lam u + b_P) = 0 and u_N makes the constraint bind
            mu = -t / 2.0
            u = -(bp + gp / (2.0 * mu)) / lam
            pval = float(np.sum(lam * u**2 + 2 * bp * u)) + k
            zt[pos] = u
            zt[~pos] = (-pval / 2.0) * bN / nb**2
            return V @ zt, mu
    r2 = -p_min
    if r2 < -1e-12 * max(1.0, abs(k), float(np.sum(bp**2 / lam))):
        return None
    r2 = max(r2, 0.0)
    a = float(np.sqrt(np.sum(gp**2 / lam)))
    if a == 0.0:
        zt[pos] = centre
        return V @ zt, 0.0
    radius = float(np.sqrt(r2))
    zt[pos] = centre - radius * (gp / lam) / a
    mu = a / (2 * radius) if radius > 0 else np.inf
    return V @ zt, mu


def _is_point(sol: Any) -> bool:
    return sol is not None and not isinstance(sol[0], str)


def _min_linear_two_quads(g: Any, q1: Any, q2: Any) -> Any:
    """min g'z s.t. q1(z) <= 0 and q2(z) <= 0 (both convex quadratics).

    ``q = (M, b, k)`` stands for ``z'Mz + 2b'z + k``. If the solution with
    either constraint alone satisfies the other, it is returned; otherwise
    both bind and the optimum is the solution with the single constraint
    ``q1 + r q2 <= 0`` at the ratio ``r = mu2 / mu1`` of the KKT
    multipliers, the root of ``q2(z(r)) = 0``. Returns
    ``(z, (mu1, mu2))``, a ray / unbounded marker as
    :func:`_min_linear_one_quad`, or ``None`` (empty set or no root).
    """
    M1, b1, k1 = (np.asarray(q1[0], float), np.asarray(q1[1], float), float(q1[2]))
    M2, b2, k2 = (np.asarray(q2[0], float), np.asarray(q2[1], float), float(q2[2]))

    g = np.asarray(g, float)
    n = len(g)
    s2 = float(M2[0, 0]) if n else 0.0
    iso = n > 0 and s2 > 0 and np.allclose(M2, s2 * np.eye(n), rtol=0, atol=1e-14 * s2)
    if iso:  # M1 + r s2 I shares the eigenvectors of M1
        lam1, V1 = np.linalg.eigh(0.5 * (M1 + M1.T))
        gt1, bt1, bt2 = V1.T @ g, V1.T @ b1, V1.T @ b2

    def mixed(r: float) -> Any:
        if np.isinf(r):
            return _min_linear_one_quad(g, M2, b2, k2)
        if iso and r > 0:
            lam = lam1 + r * s2
            if lam.min() > 1e-11 * lam.max():
                bt = bt1 + r * bt2
                kk = k1 + r * k2
                r2 = float(np.sum(bt**2 / lam)) - kk
                if r2 < -1e-12 * max(1.0, abs(kk)):
                    return None
                rad = np.sqrt(max(r2, 0.0))
                a = float(np.sqrt(np.sum(gt1**2 / lam)))
                zt = -bt / lam
                if a == 0.0:
                    return V1 @ zt, 0.0
                zt = zt - rad * (gt1 / lam) / a
                return V1 @ zt, (a / (2 * rad) if rad > 0 else np.inf)
        return _min_linear_one_quad(g, M1 + r * M2, b1 + r * b2, k1 + r * k2)

    def q2_of(z: np.ndarray) -> float:
        return _quad_val(z, M2, b2, k2)

    def q1_of(z: np.ndarray) -> float:
        return _quad_val(z, M1, b1, k1)

    tol1 = 1e-12 * (1.0 + abs(k1))
    tol2 = 1e-12 * (1.0 + abs(k2))
    s0 = mixed(0.0)
    if s0 is None:
        return None
    if _is_point(s0) and q2_of(s0[0]) <= tol2:
        return s0[0], (float(s0[1]), 0.0)
    sinf = mixed(np.inf)
    if sinf is None:
        return None
    if _is_point(sinf) and q1_of(sinf[0]) <= tol1:
        return sinf[0], (0.0, float(sinf[1]))
    if not _is_point(s0) and not _is_point(sinf):
        return s0
    hi = 1.0
    for _ in range(2100):
        sh = mixed(hi)
        if sh is None:
            return None
        if _is_point(sh) and q2_of(sh[0]) <= 0:
            break
        hi *= 2.0
    else:  # pragma: no cover - the ratio is finite when both bind
        return None
    lo = 0.0
    if not _is_point(s0):
        lo = hi / 2.0
        while lo > 1e-300:
            sl = mixed(lo)
            if _is_point(sl) and q2_of(sl[0]) > 0:
                break
            lo /= 2.0

    def gap(r: float) -> float:
        s = mixed(r)
        return q2_of(s[0]) if _is_point(s) else 1.0

    if gap(lo) <= 0:
        r = lo
    else:
        r = optimize.brentq(gap, lo, hi, xtol=1e-300, rtol=4 * _EPS, maxiter=500)
    s = mixed(r)
    if not _is_point(s):  # pragma: no cover
        return None
    return s[0], (float(s[1]), float(r * s[1]))


# ---------------------------------------------------------------------- #
#  Working-set subproblem and active-set method
# ---------------------------------------------------------------------- #


def _subproblem(
    c: Any,
    Qm: Any,
    G: Any,
    k0: Any,
    ell: Any,
    fixed: Any,
    sum_on: Any,
    sum_val: Any,
    ball: Any,
    base: Optional[np.ndarray] = None,
) -> Any:
    """Minimise c'y over {q(y) <= 0, y[fixed] = ell[fixed], [1'y = sum_val], [ball]}.

    ``q(y) = y'Qm y - 2G'y + k0``; ``ball = (w, R)`` adds ``||w + y|| <= R``.
    The free coordinates are parametrised around ``base`` (a point of the
    affine set; by default the fixed coordinates at ``ell`` and the rest
    spread to meet the sum). Returns ``(y, (mu_q, mu_ball))``, a ray
    ``("ray", d)`` with ``d`` lifted to the full space, ``("unbounded",
    None)``, or ``None`` (no free coordinate, or an empty set).
    """
    c = np.asarray(c, float)
    Qm = np.asarray(Qm, float)
    G = np.asarray(G, float)
    ell = np.asarray(ell, float)
    fixed = np.asarray(fixed, bool)
    n = len(c)
    idx = np.flatnonzero(~fixed)
    m = len(idx)
    if m == 0:
        return None
    if base is None:
        base = np.zeros(n)
        base[fixed] = ell[fixed]
        if sum_on:
            base[idx] = (float(sum_val) - base[fixed].sum()) / m
    else:
        base = np.array(base, float)
        base[fixed] = ell[fixed]
    Zb = _sum_nullspace(m) if sum_on else np.eye(m)
    MF = Qm[np.ix_(idx, idx)]
    R1 = Zb.T @ MF @ Zb
    h1 = Zb.T @ (Qm @ base - G)[idx]
    k1 = _quad_val(base, Qm, -G, float(k0))
    e = Zb.T @ c[idx]
    if ball is None:
        sol = _min_linear_one_quad(e, R1, h1, k1)
        mus = None if not _is_point(sol) else (float(sol[1]), 0.0)
    else:
        wb, Rad = np.asarray(ball[0], float), float(ball[1])
        cen = wb + base
        R2 = Zb.T @ Zb
        h2 = Zb.T @ cen[idx]
        k2 = float(cen @ cen) - Rad**2
        sol = _min_linear_two_quads(e, (R1, h1, k1), (R2, h2, k2))
        mus = None if not _is_point(sol) else sol[1]
    if sol is None:
        return None
    if isinstance(sol[0], str):
        if sol[0] == "unbounded":
            return sol
        d = np.zeros(n)
        d[idx] = Zb @ sol[1]
        return ("ray", d / np.linalg.norm(d))
    y = base.copy()
    y[idx] = base[idx] + Zb @ sol[0]
    return y, mus


def _kkt_multipliers(
    c: np.ndarray,
    grads: Sequence[np.ndarray],
    active: Sequence[bool],
    mus: Optional[Sequence[float]],
    free: np.ndarray,
    cand: np.ndarray,
    has_sum: bool,
) -> Tuple[np.ndarray, float]:
    """Bound multipliers ``lam = c + sum_i mu_i grad_i + nu 1`` on ``cand``.

    With the subproblem's multipliers ``mus`` the sum multiplier ``nu`` is
    read off the free coordinates. When those leave the choice open (no free
    direction, or an active quadratic constraint with a zero multiplier, so
    that the multipliers are not unique), they are chosen by a small LP that
    maximises the smallest bound multiplier subject to stationarity on the
    free coordinates.
    """
    idx_f = np.flatnonzero(free)
    G = np.column_stack(grads) if grads else np.zeros((len(c), 0))
    room = len(idx_f) > (1 if has_sum else 0)
    degenerate = mus is None or not all(np.isfinite(mus)) or not room
    if not degenerate:
        mu = np.array([m if a else 0.0 for m, a in zip(mus or [], active)])
        base = c + G @ mu
        nu = -float(np.mean(base[idx_f])) if has_sum else 0.0
        lam = base + nu
        lmin = float(lam[cand].min()) if cand.any() else np.inf
        # any valid multiplier vector with lam >= 0 certifies optimality; the
        # LP is only needed when these are not unique and show a negative one
        if lmin >= 0 or not any(
            a and abs(m) <= 1e-12 for m, a in zip(mus or [], active)
        ):
            return lam, lmin
    k = G.shape[1]
    scale = 1.0 + float(np.abs(c).max())
    tol = 1e-9 * scale
    fast = _multipliers_one_dim(c, G, active, idx_f, cand, has_sum, tol)
    if fast is not None:
        return fast
    # variables: mu_1..mu_k (>= 0; 0 if inactive), nu (free if sum), t;
    # stationarity on the free coordinates within +/- tol
    rows, rhs = [], []
    for i in idx_f:
        r = np.concatenate([G[i], [1.0 if has_sum else 0.0, 0.0]])
        rows.append(r)
        rhs.append(-c[i] + tol)
        rows.append(-r)
        rhs.append(c[i] + tol)
    for i in np.flatnonzero(cand):
        rows.append(np.concatenate([-G[i], [-(1.0 if has_sum else 0.0), 1.0]]))
        rhs.append(c[i])
    bounds = [(0.0, None if a else 0.0) for a in active] + [
        (None, None) if has_sum else (0.0, 0.0),
        (None, scale),
    ]
    cost = np.zeros(k + 2)
    cost[-1] = -1.0
    res = optimize.linprog(
        cost, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs"
    )
    if res.status != 0:
        return c.copy(), -np.inf
    mu, nu = res.x[:k], res.x[k]
    lam = c + G @ mu + nu
    return lam, (float(lam[cand].min()) if cand.any() else np.inf)


def _multipliers_one_dim(
    c: np.ndarray,
    G: np.ndarray,
    active: Sequence[bool],
    idx_f: np.ndarray,
    cand: np.ndarray,
    has_sum: bool,
    tol: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """The multiplier LP of :func:`_kkt_multipliers` when stationarity on the
    free coordinates leaves at most a one-dimensional family (exact,
    without an LP solver); ``None`` when the family is larger."""
    cols = [G[:, i] for i, a in enumerate(active) if a]
    pos = [i for i, a in enumerate(active) if a]
    if has_sum:
        cols.append(np.ones(len(c)))
    if not cols:
        return None
    X = np.column_stack(cols)
    XF = X[idx_f]
    if len(idx_f):
        sol, *_ = np.linalg.lstsq(XF, -c[idx_f], rcond=None)
        if len(idx_f) and np.abs(XF @ sol + c[idx_f]).max() > tol:
            return None
        _, sv, vt = np.linalg.svd(XF)
        rank = int(np.sum(sv > 1e-10 * max(sv.max(), 1e-300))) if sv.size else 0
    else:
        sol, rank, vt = np.zeros(X.shape[1]), 0, np.eye(X.shape[1])
    null = vt[rank:].T
    if null.shape[1] > 1:
        return None
    nmu = len(pos)
    jc = np.flatnonzero(cand)
    # x(s) = sol + s d; maximise min_j (c_j + X_j x(s)) with mu(s) >= 0
    d = null[:, 0] if null.shape[1] == 1 else np.zeros(X.shape[1])
    a0 = c[jc] + X[jc] @ sol
    a1 = X[jc] @ d
    lo, hi = -np.inf, np.inf
    for i in range(nmu):
        if abs(d[i]) <= 1e-15:
            if sol[i] < -tol:
                return None
            continue
        bound = -sol[i] / d[i]
        if d[i] > 0:
            lo = max(lo, bound)
        else:
            hi = min(hi, bound)
    if lo > hi:
        return None
    cands = [v for v in (lo, hi) if np.isfinite(v)]
    for i in range(len(jc)):
        for k2 in range(i + 1, len(jc)):
            if a1[i] != a1[k2]:
                cands.append((a0[k2] - a0[i]) / (a1[i] - a1[k2]))
    cands = [v for v in cands if lo - 1e-15 <= v <= hi + 1e-15] or [
        0.0 if lo <= 0 <= hi else (lo if np.isfinite(lo) else hi)
    ]
    best = max(cands, key=lambda v: float(np.min(a0 + a1 * v)) if len(jc) else 0.0)
    x = sol + best * d
    mu = np.zeros(G.shape[1])
    mu[pos] = np.maximum(x[:nmu], 0.0)
    nu = x[nmu] if has_sum else 0.0
    lam = c + G @ mu + nu
    return lam, (float(lam[cand].min()) if cand.any() else np.inf)


def insample_active_set(
    c: np.ndarray,
    Qm: np.ndarray,
    G: np.ndarray,
    k0: float,
    ell: np.ndarray,
    sum_mode: Optional[str] = None,
    sum_val: float = 0.0,
    ball: Optional[Tuple[np.ndarray, float]] = None,
    y0: Optional[np.ndarray] = None,
    max_iter: int = 300,
) -> Union[None, str, np.ndarray]:
    """min c'y s.t. y'Qy - 2G'y + k0 <= 0, y >= ell, [1'y (=|<=) sum_val], [ball].

    Primal active-set method from the feasible start ``y0`` (default 0):
    on a working set of bounds held at ``ell`` the subproblem is solved in
    closed form (:func:`_subproblem`); a step that would leave the box is
    cut at the first bound, which joins the working set; at a stationary
    point the bound with the most negative KKT multiplier is released.
    ``sum_mode`` is ``None``, ``"=="`` or ``"<="``; ``ball = (w, R)`` adds
    ``||w + y|| <= R``. Returns the minimiser, ``"unbounded"``, or ``None``
    when the start is infeasible or the method does not settle.
    """
    c = np.asarray(c, float)
    Qm = 0.5 * (np.asarray(Qm, float) + np.asarray(Qm, float).T)
    G = np.asarray(G, float)
    ell = np.asarray(ell, float)
    n = len(c)
    y = np.zeros(n) if y0 is None else np.array(y0, float)
    if sum_mode == "<=":  # slack coordinate turns the inequality into an equality
        slack = float(sum_val) - float(y.sum())
        if slack < -1e-12 * (1.0 + abs(float(sum_val))):
            return None
        wb = None if ball is None else (np.append(ball[0], 0.0), ball[1])
        Qx = np.zeros((n + 1, n + 1))
        Qx[:n, :n] = Qm
        Ex = np.eye(n + 1)
        Ex[n, n] = 0.0
        res = _active_set(
            np.append(c, 0.0),
            Qx,
            np.append(G, 0.0),
            k0,
            np.append(ell, 0.0),
            True,
            sum_val,
            wb,
            Ex,
            np.append(y, max(slack, 0.0)),
            max_iter,
        )
        return res if res is None or isinstance(res, str) else res[:n]
    return _active_set(
        c, Qm, G, k0, ell, sum_mode == "==", sum_val, ball, None, y, max_iter
    )


def _active_set(
    c: np.ndarray,
    Qm: np.ndarray,
    G: np.ndarray,
    k0: float,
    ell: np.ndarray,
    has_sum: bool,
    sum_val: float,
    ball: Optional[Tuple[np.ndarray, float]],
    ball_mask: Optional[np.ndarray],
    y: np.ndarray,
    max_iter: int,
) -> Union[None, str, np.ndarray]:
    finite = np.isfinite(ell)
    tol_b = 1e-12 * (1.0 + np.abs(np.where(finite, ell, 0.0)))
    if np.any(y[finite] < ell[finite] - tol_b[finite]):
        return None
    qscale = 1.0 + abs(float(k0)) + float(np.abs(G @ y))
    if _quad_val(y, Qm, -G, float(k0)) > 1e-10 * qscale:
        return None
    if has_sum and abs(float(y.sum()) - float(sum_val)) > 1e-10 * (
        1.0 + abs(float(sum_val))
    ):
        return None
    if ball is not None:
        bvec = ball[0] + (y if ball_mask is None else ball_mask @ y)
        if float(bvec @ bvec) > ball[1] ** 2 * (1 + 1e-10):
            return None
    y = np.where(finite & (y < ell), ell, y)
    at = finite & (np.abs(y - np.where(finite, ell, 0.0)) <= tol_b)
    released: Dict[int, int] = {}
    for _ in range(max_iter):
        free = ~at
        if free.sum() > (1 if has_sum else 0):
            sol = _subproblem_masked(
                c, Qm, G, k0, ell, at, has_sum, sum_val, ball, ball_mask, y
            )
        else:
            sol = (y.copy(), None)
        if sol is None:
            return None
        if isinstance(sol[0], str):
            if sol[0] == "unbounded":
                # unbounded on the face, not along a ray: bounds may still
                # stop it, so let the caller's exact fallbacks decide
                return None if (free & finite).any() else "unbounded"
            d = sol[1]
            dec = free & finite & (d < -1e-14 * np.linalg.norm(d))
            if not dec.any():
                return "unbounded"
            ratios = (ell[dec] - y[dec]) / d[dec]
            i = int(np.argmin(ratios))
            y = y + ratios[i] * d
            j = int(np.flatnonzero(dec)[i])
            y[j] = ell[j]
            at[j] = True
            continue
        target, mus = sol
        step = target - y
        dec = free & finite & (step < 0)
        alpha, block = 1.0, -1
        if dec.any():
            ratios = (ell[dec] - y[dec]) / step[dec]
            i = int(np.argmin(ratios))
            if ratios[i] < 1.0:
                alpha, block = max(float(ratios[i]), 0.0), int(np.flatnonzero(dec)[i])
        y = y + alpha * step
        if block >= 0:
            y[block] = ell[block]
            at[block] = True
            continue
        cand = at & finite
        if not cand.any():
            return y
        grad_q = 2.0 * (Qm @ y - G)
        q_now = _quad_val(y, Qm, -G, float(k0))
        grads, active = [grad_q], [q_now >= -1e-9 * qscale]
        if ball is not None:
            bvec = ball[0] + (y if ball_mask is None else ball_mask @ y)
            gb = 2.0 * (bvec if ball_mask is None else ball_mask.T @ bvec)
            grads.append(gb)
            active.append(
                float(bvec @ bvec) - ball[1] ** 2 >= -1e-9 * (1.0 + ball[1] ** 2)
            )
        mus_full = (
            None if mus is None else (list(mus) + [0.0] * (len(grads) - len(mus)))
        )
        lam, lmin = _kkt_multipliers(c, grads, active, mus_full, ~at, cand, has_sum)
        if lmin >= -1e-10 * max(1.0, float(np.abs(c).max())):
            return y
        j = int(np.flatnonzero(cand)[np.argmin(lam[cand])])
        released[j] = released.get(j, 0) + 1
        if released[j] > 3:
            return None  # cycling; the caller falls back
        at[j] = False
    return None


def _subproblem_masked(
    c: np.ndarray,
    Qm: np.ndarray,
    G: np.ndarray,
    k0: float,
    ell: np.ndarray,
    fixed: np.ndarray,
    has_sum: bool,
    sum_val: float,
    ball: Optional[Tuple[np.ndarray, float]],
    ball_mask: Optional[np.ndarray],
    y: np.ndarray,
) -> Any:
    """:func:`_subproblem` around the current point; supports a ball that
    involves only the coordinates selected by the diagonal ``ball_mask``."""
    if ball is None or ball_mask is None:
        return _subproblem(c, Qm, G, k0, ell, fixed, has_sum, sum_val, ball, base=y)
    n = len(c)
    idx = np.flatnonzero(~fixed)
    base = np.array(y, float)
    base[fixed] = ell[fixed]
    Zb = _sum_nullspace(len(idx)) if has_sum else np.eye(len(idx))
    MF = Qm[np.ix_(idx, idx)]
    R1 = Zb.T @ MF @ Zb
    h1 = Zb.T @ (Qm @ base - G)[idx]
    k1 = _quad_val(base, Qm, -G, float(k0))
    e = Zb.T @ c[idx]
    cen = ball[0] + ball_mask @ base
    Mf = ball_mask[np.ix_(idx, idx)]
    R2 = Zb.T @ Mf @ Zb
    h2 = Zb.T @ (ball_mask.T @ cen)[idx]
    k2 = float(cen @ cen) - ball[1] ** 2
    sol = _min_linear_two_quads(e, (R1, h1, k1), (R2, h2, k2))
    if sol is None:
        return None
    if isinstance(sol[0], str):
        if sol[0] == "unbounded":
            return sol
        d = np.zeros(n)
        d[idx] = Zb @ sol[1]
        return ("ray", d / np.linalg.norm(d))
    out = base.copy()
    out[idx] = base[idx] + Zb @ sol[0]
    return out, sol[1]


def warn_failed(n_failed: int, n_total: int) -> None:
    """Warn when more than 20% of the simulated problems failed."""
    if n_total and n_failed > 0.2 * n_total:
        warnings.warn(
            f"{n_failed}/{n_total} in-sample simulation problems could not be "
            "solved; the in-sample bounds use the remaining draws.",
            RuntimeWarning,
            stacklevel=3,
        )


__all__ = ["insample_active_set", "warn_failed"]
