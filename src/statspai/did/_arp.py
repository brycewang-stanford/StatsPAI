"""Conditional moment-inequality confidence sets for Honest DiD (native).

Rambachan and Roth [@rambachan2023more] build confidence sets for a
post-treatment target ``theta = l' tau_post`` under a polyhedral restriction
``Delta = {delta : A delta <= d}`` on the parallel-trends violation by
inverting, over a grid of ``theta``, the conditional (and conditional
least-favourable hybrid) moment-inequality tests of Andrews, Roth and Pakes
[@andrews2023inference]. This module implements that inversion in Python for
the relative-magnitudes restriction ``Delta^RM(Mbar)`` -- the union over the
location ``s`` and sign of the largest pre-period first difference of
polyhedral pieces -- so that ``sp.honest_did(method="relative_magnitude")``
no longer needs R.

The construction follows the reference implementation, R ``HonestDiD``
0.2.8 (Rambachan and Roth; MIT licence, copyright 2023 HonestDiD authors;
the notice is reproduced in THIRD_PARTY_NOTICES.md), step for step: the
same moment matrices, the same default ``theta`` grid (``+/- 20`` standard deviations of the target, 1,000
points), the same primal linear program for the test statistic, the same
degeneracy rule for switching between the primal and the dual computation
of the truncation points, and the same bisection tolerances. Matching those
choices is what makes a finite-grid confidence set comparable point by
point: both implementations report the smallest and largest accepted grid
value, so agreement is exact when every grid decision agrees.

Two things cannot be made identical and are documented rather than hidden:

* The conditional least-favourable hybrid (``"C-LF"``, HonestDiD's
  default) calibrates its first stage with a simulated least-favourable
  critical value (1,000 draws). The draws here come from NumPy, not from
  R's generator, so that critical value -- and hence the hybrid set -- can
  differ by Monte Carlo error. The purely conditional test
  (``"Conditional"``) involves no simulation.
* Degenerate linear programs can admit several optimal dual vertices; the
  test then depends on which vertex the solver returns. HiGHS and
  ``lpSolve`` usually agree; where they do not, the difference is confined
  to grid points adjacent to the set boundary.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats

__all__ = [
    "create_a_rm",
    "rm_confidence_set",
    "conditional_test",
]

_TOL_LAMBDA = 1e-6
_TOL_EQUALITY = 1e-6
_TOL_C = 1e-6


# ---------------------------------------------------------------------------
#  Restriction matrices
# ---------------------------------------------------------------------------


def create_a_rm(
    n_pre: int, n_post: int, m_bar: float, s: int, max_positive: bool
) -> np.ndarray:
    """Moment matrix ``A`` of ``Delta^RM_{s,sign}(Mbar)`` (``d = 0``).

    Columns index ``delta`` without the normalised reference period. Rows
    impose ``|delta_{r+1} - delta_r| <= +/-(delta_{s+1} - delta_s)`` on every
    pre-period first difference and ``<= Mbar`` times that on every
    post-period one, where ``s`` (in ``-(n_pre - 1), ..., 0``) locates the
    largest pre-period difference and ``max_positive`` fixes its sign.
    """
    k = n_pre + n_post
    a_tilde = np.zeros((k, k + 1))
    for r in range(k):
        a_tilde[r, r : r + 2] = (-1.0, 1.0)
    v = np.zeros(k + 1)
    # R: v_max_dif[(numPre + s):(numPre + 1 + s)] <- c(-1, 1) (1-based).
    v[n_pre + s - 1 : n_pre + s + 1] = (-1.0, 1.0)
    if not max_positive:
        v = -v
    a_ub = np.vstack([np.tile(v, (n_pre, 1)), np.tile(m_bar * v, (n_post, 1))])
    a = np.vstack([a_tilde - a_ub, -a_tilde - a_ub])
    a = a[np.einsum("ij,ij->i", a, a) > 1e-10]
    return np.delete(a, n_pre, axis=1)


def _construct_gamma(l_vec: np.ndarray) -> np.ndarray:
    """Invertible change of basis whose first row is ``l`` (HonestDiD's RREF rule)."""
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    b = np.column_stack([l_vec, np.eye(l_vec.size)])
    pivots: List[int] = []
    rows = b.copy()
    r = 0
    for c in range(rows.shape[1]):
        if r >= rows.shape[0]:
            break
        p = r + int(np.argmax(np.abs(rows[r:, c])))
        if abs(rows[p, c]) < 1e-12:
            continue
        rows[[r, p]] = rows[[p, r]]
        rows[r] /= rows[r, c]
        for i in range(rows.shape[0]):
            if i != r:
                rows[i] -= rows[i, c] * rows[r]
        pivots.append(c)
        r += 1
    return b[:, pivots].T


# ---------------------------------------------------------------------------
#  Linear programs
# ---------------------------------------------------------------------------


def _eta_lp(
    y: np.ndarray, x: np.ndarray, sigma: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """``min eta`` s.t. ``y - X delta <= eta * sd`` -> (eta, delta, lambda, ok).

    ``lambda`` are the (non-negative) multipliers of the inequality rows:
    the dual solution ``gamma`` of Andrews, Roth and Pakes.
    """
    sd = np.sqrt(np.diag(sigma))
    k = x.shape[1]
    c = np.r_[1.0, np.zeros(k)]
    a_ub = -np.column_stack([sd, x])
    res = optimize.linprog(
        c,
        A_ub=a_ub,
        b_ub=-y,
        bounds=[(None, None)] * (k + 1),
        method="highs-ds",
    )
    if res.status != 0:
        return float("nan"), np.full(k, np.nan), np.zeros(y.size), False
    lam = -np.asarray(res.ineqlin.marginals, dtype=float)
    return float(res.fun), np.asarray(res.x[1:], dtype=float), lam, True


def _max_program(s_t, gamma, sigma, w_t, c) -> Tuple[float, np.ndarray]:
    """``max f'x`` s.t. ``W' x = e_1``, ``x >= 0`` (value, solution)."""
    b = (sigma @ gamma) / float(gamma @ sigma @ gamma)
    f = s_t + b * c
    beq = np.zeros(w_t.shape[1])
    beq[0] = 1.0
    res = optimize.linprog(
        -f, A_eq=w_t.T, b_eq=beq, bounds=[(0, None)] * f.size, method="highs-ds"
    )
    if res.status != 0:
        return float("nan"), np.full(f.size, np.nan)
    return float(-res.fun), np.asarray(res.x, dtype=float)


#: Vertex enumeration is used for the dual value function when the number of
#: candidate bases is at most this; larger problems fall back to one LP per
#: evaluation.
_MAX_BASES = 20000


def _polytope_vertices(w_t: np.ndarray) -> Optional[np.ndarray]:
    """All vertices of ``{x >= 0 : W' x = e_1}`` (rows), or ``None`` if too many.

    The first column of ``W`` is the vector of moment standard deviations,
    which is strictly positive, so the polytope is bounded and the maximum
    of any linear objective over it is attained at one of these vertices.
    The dual bisection evaluates that maximum dozens of times with only the
    objective changing; enumerating the vertices once replaces each of those
    linear programs with a matrix-vector product.
    """
    from itertools import combinations
    from math import comb

    m = w_t.shape[0]
    q = int(np.linalg.matrix_rank(w_t))
    if q == 0 or comb(m, q) > _MAX_BASES:
        return None
    e1 = np.zeros(w_t.shape[1])
    e1[0] = 1.0
    out = []
    for basis in combinations(range(m), q):
        wb = w_t[list(basis)].T  # (k+1) x q
        if np.linalg.matrix_rank(wb) < q:
            continue
        xb, *_ = np.linalg.lstsq(wb, e1, rcond=None)
        if np.max(np.abs(wb @ xb - e1)) > 1e-10 or np.min(xb) < -1e-12:
            continue
        x = np.zeros(m)
        x[list(basis)] = np.maximum(xb, 0.0)
        out.append(x)
    if not out:
        return None
    return np.unique(np.round(np.asarray(out), 14), axis=0)


def _roundeps(x: float, eps: float = np.finfo(float).eps ** 0.75) -> float:
    return 0.0 if abs(x) < eps else x


def _vlo_vup_dual(eta, s_t, gamma, sigma, w_t, vertices=None) -> Tuple[float, float]:
    """Truncation points of the conditional test via the dual LP value function."""

    if vertices is None:
        vertices = _polytope_vertices(w_t)
    b_vec = (sigma @ gamma) / float(gamma @ sigma @ gamma)
    if vertices is not None:
        v_s, v_b = vertices @ s_t, vertices @ b_vec

    def check(c):
        if vertices is None:
            value, x = _max_program(s_t, gamma, sigma, w_t, c)
        else:
            obj = v_s + v_b * c
            j = int(np.argmax(obj))
            value, x = float(obj[j]), vertices[j]
        return (not np.isnan(value)) and abs(c - value) <= _TOL_EQUALITY, x

    sigma_b = float(np.sqrt(gamma @ sigma @ gamma))
    low_initial = min(-100.0, eta - 20.0 * sigma_b)
    high_initial = max(100.0, eta + 20.0 * sigma_b)
    maxiters, switchiters = 10000, 10
    ok, _ = check(eta)
    if not ok:
        return eta, float("inf")
    b = (sigma @ gamma) / float(gamma @ sigma @ gamma)

    def newton_then_bisect(start, inner, upward):
        ok, x = check(start)
        if ok:
            return float("inf") if upward else float("-inf")
        dif, iters = 0.0, 1
        mid = _roundeps(float(x @ s_t)) / (1.0 - float(x @ b))
        while True:
            ok, x = check(mid)
            if ok or iters >= maxiters:
                break
            iters += 1
            if iters >= switchiters:
                dif = _TOL_C + 1.0
                break
            mid = _roundeps(float(x @ s_t)) / (1.0 - float(x @ b))
        lo, hi = (inner, mid) if upward else (mid, inner)
        while dif > _TOL_C and iters < maxiters:
            iters += 1
            mid = 0.5 * (hi + lo)
            if check(mid)[0]:
                if upward:
                    lo = mid
                else:
                    hi = mid
            elif upward:
                hi = mid
            else:
                lo = mid
            dif = hi - lo
        return mid

    vup = newton_then_bisect(high_initial, eta, upward=True)
    vlo = newton_then_bisect(low_initial, eta, upward=False)
    return vlo, vup


def _truncnorm_quantile(p: float, lo: float, hi: float) -> float:
    """Quantile of a standard normal truncated to ``[lo, hi]``."""
    return float(stats.truncnorm.ppf(p, lo, hi))


# ---------------------------------------------------------------------------
#  Conditional (ARP) test with nuisance parameters
# ---------------------------------------------------------------------------


def conditional_test(
    y: np.ndarray,
    x: np.ndarray,
    sigma: np.ndarray,
    alpha: float,
    hybrid: str = "ARP",
    lf_cv: Optional[float] = None,
    kappa: Optional[float] = None,
    vertices: Optional[np.ndarray] = None,
) -> bool:
    """Reject ``H0: exists delta, E[y] - X delta <= 0`` at level ``alpha``?

    ``hybrid="ARP"`` is the conditional test; ``"LF"`` the conditional
    least-favourable hybrid with first-stage critical value ``lf_cv`` and
    first-stage size ``kappa``. ``vertices`` (from
    :func:`_polytope_vertices` of ``[sd, X]``) may be passed when the same
    ``X`` and ``sigma`` are tested at many ``y``: the test statistic is then
    the maximum of ``gamma' y`` over those vertices -- the dual of the primal
    program -- and the maximising vertex is the dual solution.
    """
    m, k = x.shape
    if vertices is None:
        eta, _, lam, ok = _eta_lp(y, x, sigma)
    else:
        vals = vertices @ y
        j = int(np.argmax(vals))
        eta, lam, ok = float(vals[j]), vertices[j], True
    if not ok:
        return False  # HonestDiD: LP did not converge -> do not reject
    if hybrid == "LF":
        mod_size = (alpha - kappa) / (1.0 - kappa)
        if eta > lf_cv:
            return True
    else:
        mod_size = alpha

    b_idx = lam > _TOL_LAMBDA
    degenerate = int(b_idx.sum()) != k + 1
    x_b = x[b_idx]
    full_rank = min(x_b.shape) > 0 and np.linalg.matrix_rank(x_b) == min(x_b.shape)

    if (not full_rank) or degenerate:
        sd = np.sqrt(np.diag(sigma))
        w_t = np.column_stack([sd, x])
        gamma = lam
        s2 = float(gamma @ sigma @ gamma)
        if abs(s2) < np.finfo(float).eps:
            return eta > 0
        s_t = y - (sigma @ gamma) * float(gamma @ y) / s2
        vlo, vup = _vlo_vup_dual(eta, s_t, gamma, sigma, w_t, vertices)
        sigma_b = np.sqrt(s2)
        maxstat = eta / sigma_b
        zlo = vlo / sigma_b
        zup = (min(vup, lf_cv) if hybrid == "LF" else vup) / sigma_b
    else:
        sd = np.sqrt(np.diag(sigma))
        bc_idx = ~b_idx
        eye = np.eye(m)
        s_b, s_bc = eye[b_idx], eye[bc_idx]
        w_b_inv = np.linalg.inv(np.column_stack([sd[b_idx], x_b]))
        gamma_b = np.column_stack([sd[bc_idx], x[bc_idx]]) @ w_b_inv @ s_b - s_bc
        e1 = np.zeros(int(b_idx.sum()))
        e1[0] = 1.0
        v_b = (e1 @ w_b_inv @ s_b).ravel()
        sigma2_b = float(v_b @ sigma @ v_b)
        sigma_b = np.sqrt(sigma2_b)
        rho = (gamma_b @ sigma @ v_b) / sigma2_b
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = (-gamma_b @ y) / rho + float(v_b @ y)
        vlo = float(np.max(bound[rho > 0])) if np.any(rho > 0) else float("-inf")
        vup = float(np.min(bound[rho < 0])) if np.any(rho < 0) else float("inf")
        maxstat = eta / sigma_b
        zlo = vlo / sigma_b
        zup = (min(vup, lf_cv) if hybrid == "LF" else vup) / sigma_b

    if not (zlo <= maxstat <= zup):
        return False
    cval = max(0.0, _truncnorm_quantile(1.0 - mod_size, zlo, zup))
    return bool(maxstat > cval)


def _least_favorable_cv(
    x: Optional[np.ndarray],
    sigma: np.ndarray,
    kappa: float,
    sims: int = 1000,
    seed: int = 0,
) -> float:
    """First-stage critical value of the conditional-LF hybrid (simulated)."""
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=sims)
    if x is None:
        eta = np.max(draws / np.sqrt(np.diag(sigma)), axis=1)
        return float(np.quantile(eta, 1.0 - kappa))
    # HonestDiD solves the eta program at y = xi for each draw. Its value is
    # the maximum of gamma' xi over the dual polytope's vertices.
    verts = _polytope_vertices(np.column_stack([np.sqrt(np.diag(sigma)), x]))
    if verts is not None:
        etas = np.max(draws @ verts.T, axis=1)
    else:
        etas = []
        for xi in draws:
            e, _, _, ok = _eta_lp(xi, x, sigma)
            if ok:
                etas.append(e)
        etas = np.asarray(etas)
    return float(np.quantile(etas, 1.0 - kappa))


# ---------------------------------------------------------------------------
#  No-nuisance case (a single post-treatment period)
# ---------------------------------------------------------------------------


def _test_no_nuisance(y, sigma, a, d, alpha, hybrid, lf_cv, kappa) -> bool:
    sig_tilde = np.sqrt(np.diag(a @ sigma @ a.T))
    a_t = a / sig_tilde[:, None]
    d_t = d / sig_tilde
    moments = a_t @ y - d_t
    j = int(np.argmax(moments))
    max_moment = float(moments[j])
    if hybrid == "LF" and max_moment > lf_cv:
        return True
    gamma = a_t[j]
    a_bar = a_t - a_t[j][None, :]
    d_bar = d_t - d_t[j]
    s2 = float(gamma @ sigma @ gamma)
    sigmabar = np.sqrt(s2)
    c = (sigma @ gamma) / s2
    z = y - c * float(gamma @ y)
    ac = a_bar @ c
    with np.errstate(divide="ignore", invalid="ignore"):
        obj = (d_bar - a_bar @ z) / ac
    vlo = float(np.max(obj[ac < 0])) if np.any(ac < 0) else float("-inf")
    vup = float(np.min(obj[ac > 0])) if np.any(ac > 0) else float("inf")
    size = (alpha - kappa) / (1.0 - kappa) if hybrid == "LF" else alpha
    mu = float(d_t[j])
    q = _truncnorm_quantile(1.0 - size, (vlo - mu) / sigmabar, (vup - mu) / sigmabar)
    crit = max(0.0, mu + q * sigmabar)
    return bool(max_moment + mu > crit)


# ---------------------------------------------------------------------------
#  Delta^RM confidence set
# ---------------------------------------------------------------------------


def _ci_fixed_s(
    betahat,
    sigma,
    n_pre,
    n_post,
    l_vec,
    m_bar,
    s,
    max_positive,
    alpha,
    hybrid,
    kappa,
    grid,
    seed,
    skip=None,
) -> np.ndarray:
    """Acceptance vector of one polyhedral piece over ``grid``.

    Points flagged in ``skip`` are already accepted by another piece of the
    union and are not re-tested (their entry here is left at 0): the union's
    acceptance vector is unchanged, and the test count falls sharply.
    """
    if skip is None:
        skip = np.zeros(grid.size, dtype=bool)
    a = create_a_rm(n_pre, n_post, m_bar, s, max_positive)
    d = np.zeros(a.shape[0])
    post_cols = slice(n_pre, n_pre + n_post)
    if n_post > 1:
        rows = np.flatnonzero(np.any(a[:, post_cols] != 0, axis=1))
    else:
        keep = a[:, -1] != 0
        a, d = a[keep], d[keep]
        rows = np.arange(a.shape[0])

    accept = np.zeros(grid.size)
    if n_post == 1:
        lf_cv = None
        if hybrid == "LF":
            lf_cv = _least_favorable_cv(None, a @ sigma @ a.T, kappa, seed=seed)
        e_post = np.zeros(betahat.size)
        e_post[n_pre] = 1.0
        for i, theta in enumerate(grid):
            if skip[i]:
                continue
            reject = _test_no_nuisance(
                betahat - e_post * theta, sigma, a, d, alpha, hybrid, lf_cv, kappa
            )
            accept[i] = 0.0 if reject else 1.0
        return accept

    gamma_mat = _construct_gamma(l_vec)
    a_gi = a[:, post_cols] @ np.linalg.inv(gamma_mat)
    a_one, a_rest = a_gi[:, 0], a_gi[:, 1:]
    y_all = a @ betahat - d
    sigma_y = a @ sigma @ a.T
    x_arp = a_rest[rows]
    sigma_arp = sigma_y[np.ix_(rows, rows)]
    lf_cv = None
    if hybrid == "LF":
        lf_cv = _least_favorable_cv(x_arp, sigma_arp, kappa, seed=seed)
    # X and sigma do not move with theta, so neither does the dual polytope.
    verts = _polytope_vertices(np.column_stack([np.sqrt(np.diag(sigma_arp)), x_arp]))
    for i, theta in enumerate(grid):
        if skip[i]:
            continue
        y_t = (y_all - a_one * theta)[rows]
        reject = conditional_test(
            y_t, x_arp, sigma_arp, alpha, hybrid, lf_cv, kappa, vertices=verts
        )
        accept[i] = 0.0 if reject else 1.0
    return accept


def rm_confidence_set(
    betahat: np.ndarray,
    sigma: np.ndarray,
    n_pre: int,
    n_post: int,
    m_bar: float,
    l_vec: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    method: str = "C-LF",
    grid_points: int = 1000,
    grid_lb: Optional[float] = None,
    grid_ub: Optional[float] = None,
    seed: int = 0,
    progress: Optional[Callable[[int], None]] = None,
    max_expand: int = 0,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Confidence set for ``l' tau_post`` under ``Delta^RM(Mbar)``.

    Returns ``(lower, upper, grid, accept)``: the smallest and largest
    accepted grid values (``nan`` if none is accepted) and the full
    acceptance vector, so a caller can detect a non-convex or
    boundary-open set.

    The default grid is HonestDiD's: ``grid_points`` values spanning
    ``+/-20 sd(l' tau_post)``. With ``max_expand > 0``, a side whose end
    point is accepted -- and whose bound the caller did not fix through
    ``grid_lb`` / ``grid_ub`` -- is extended by another ``grid_points``
    points at the same step (40 sd on the default grid), up to
    ``max_expand`` times, so the reported bound is the set's end point
    rather than the grid's. Only the new points are tested.

    If the default grid accepts nothing because the estimate itself lies
    outside it, the grid is moved to ``+/-20 sd`` around the estimate.
    """
    betahat = np.asarray(betahat, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    if l_vec is None:
        l_vec = np.eye(n_post)[0]
    l_vec = np.asarray(l_vec, dtype=float).ravel()
    method_u = method.upper().replace("_", "-")
    if method_u in {"C-LF", "LF"}:
        hybrid = "LF"
    elif method_u in {"CONDITIONAL", "ARP"}:
        hybrid = "ARP"
    else:
        raise ValueError("method must be 'C-LF' or 'Conditional'")
    kappa = alpha / 10.0
    post = sigma[n_pre:, n_pre:]
    sd_theta = float(np.sqrt(l_vec @ post @ l_vec))
    lb = -20.0 * sd_theta if grid_lb is None else float(grid_lb)
    ub = 20.0 * sd_theta if grid_ub is None else float(grid_ub)
    grid = np.linspace(lb, ub, int(grid_points))

    def union_accept(pts: np.ndarray) -> np.ndarray:
        accept = np.zeros(pts.size)
        for s in range(-(n_pre - 1), 1):
            for positive in (True, False):
                acc = _ci_fixed_s(
                    betahat,
                    sigma,
                    n_pre,
                    n_post,
                    l_vec,
                    m_bar,
                    s,
                    positive,
                    alpha,
                    hybrid,
                    kappa,
                    pts,
                    seed,
                    skip=accept == 1,
                )
                accept = np.maximum(accept, acc)
                if progress is not None:
                    progress(1)
        return accept

    accept = union_accept(grid)
    if grid.size > 1:
        step = float(grid[1] - grid[0])
        for _ in range(int(max_expand)):
            grow_lo = grid_lb is None and accept[0] == 1
            grow_hi = grid_ub is None and accept[-1] == 1
            if not (grow_lo or grow_hi):
                break
            n_new = int(grid_points)
            if grow_lo:
                new = grid[0] - step * np.arange(n_new, 0, -1)
                grid = np.concatenate([new, grid])
                accept = np.concatenate([union_accept(new), accept])
            if grow_hi:
                new = grid[-1] + step * np.arange(1, n_new + 1)
                grid = np.concatenate([grid, new])
                accept = np.concatenate([accept, union_accept(new)])
    hit = grid[accept == 1]
    if hit.size == 0 and grid_lb is None and grid_ub is None:
        # HonestDiD centres its default grid at zero. An estimate more than
        # 20 standard deviations from zero lies outside it, so nothing is
        # accepted and the set would be reported as empty. Test the same
        # number of points around the estimate instead.
        centre = float(l_vec @ betahat[n_pre:])
        if not lb <= centre <= ub:
            grid = np.linspace(
                centre - 20.0 * sd_theta, centre + 20.0 * sd_theta, int(grid_points)
            )
            accept = union_accept(grid)
            hit = grid[accept == 1]
    if hit.size == 0:
        return float("nan"), float("nan"), grid, accept
    return float(hit.min()), float(hit.max()), grid, accept
