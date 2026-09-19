"""Numerical kernels for ``sp.scest`` / ``sp.scpi`` (private to ``scpi.py``).

Everything here reproduces an optimisation problem that R ``scpi`` hands to a
conic solver (CVXR/CLARABEL for the weights, ECOSolveR for the in-sample
simulation, quantreg's Frisch-Newton LP for ``Qtools::rrq``).  StatsPAI solves
the same problems exactly (active-set / closed form / HiGHS LP) instead, so the
cross-language gap is bounded by the *reference* solver's tolerance, not ours.

Weight problems (``b.est`` in R, V = identity):

* ``simplex``  min ||A - Z w||^2  s.t. w >= 0, sum(w) = Q
* ``lasso``    min ||A - Z w||^2  s.t. ||w||_1 <= Q
* ``ridge``    min ||A - Z w||^2  s.t. ||w||_2 <= Q
* ``L1-L2``    min ||A - Z w||^2  s.t. w >= 0, sum(w) = 1, ||w||_2 <= Q2
* ``ols``      min ||A - Z w||^2

In-sample simulation problem (``insampleUncertaintyGetDiag`` in R): for a draw
``G`` and a direction ``c``

    min_x c'x  s.t.  (x - b)'Q(x - b) - 2 G'(x - b) <= 0,  x in the constraint set

with the locally-relaxed constraint set of ``local.geom`` / ``local.geom.2step``.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize

_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------- #
#  Weight estimation
# ---------------------------------------------------------------------- #


def qp_bounded_sum(
    H: np.ndarray,
    f: np.ndarray,
    lb: np.ndarray,
    total: float,
    max_iter: int = 500,
) -> np.ndarray:
    """Exact primal active-set solve of min 1/2 w'Hw - f'w, w >= lb, 1'w = total.

    ``H`` must be positive semi-definite; the minimiser is unique when ``H`` is
    positive definite (full-column-rank donor matrix).  The returned point
    satisfies the KKT conditions to machine precision (the final step is a
    direct solve of the equality-constrained system on the free set).
    """
    J = f.shape[0]
    lb = np.asarray(lb, dtype=float)
    slack = total - lb.sum()
    if slack < -1e-12 * max(1.0, abs(total)):
        raise ValueError("Infeasible simplex constraint: sum(lb) exceeds Q.")
    w = lb + max(slack, 0.0) / J
    active = np.zeros(J, dtype=bool)
    scale = max(1.0, float(np.abs(H).max()), float(np.abs(f).max()))
    for _ in range(max_iter):
        free = ~active
        nf = int(free.sum())
        g = H @ w - f
        K = np.zeros((nf + 1, nf + 1))
        K[:nf, :nf] = H[np.ix_(free, free)]
        K[:nf, nf] = 1.0
        K[nf, :nf] = 1.0
        rhs = np.concatenate([-g[free], [0.0]])
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        p = np.zeros(J)
        p[free] = sol[:nf]
        nu = sol[nf]
        if np.max(np.abs(p)) <= 1e-13 * (1.0 + np.max(np.abs(w))):
            lam = g + nu
            lam[free] = 0.0
            if not active.any() or lam[active].min() >= -1e-11 * scale:
                break
            idx = np.where(active)[0]
            active[idx[np.argmin(lam[idx])]] = False
            continue
        alpha, block = 1.0, -1
        for j in np.where(free & (p < 0))[0]:
            a = (lb[j] - w[j]) / p[j]
            if a < alpha:
                alpha, block = a, j
        w = w + alpha * p
        if block >= 0:
            active[block] = True
            w[block] = lb[block]
    else:  # pragma: no cover - active-set on a convex QP terminates
        raise RuntimeError("simplex active-set did not converge")
    # polish: exact equality-constrained solve on the free set
    free = ~active
    nf = int(free.sum())
    K = np.zeros((nf + 1, nf + 1))
    K[:nf, :nf] = H[np.ix_(free, free)]
    K[:nf, nf] = 1.0
    K[nf, :nf] = 1.0
    rhs = np.concatenate(
        [f[free] - H[np.ix_(free, active)] @ lb[active], [total - lb[active].sum()]]
    )
    try:
        sol = np.linalg.solve(K, rhs)
        cand = w.copy()
        cand[free] = sol[:nf]
        cand[active] = lb[active]
        if np.all(cand[free] >= lb[free] - 1e-12):
            w = cand
    except np.linalg.LinAlgError:  # singular H on the free set: keep iterate
        pass
    return w


def lasso_ball(Z: np.ndarray, A: np.ndarray, Q: float) -> np.ndarray:
    """min ||A - Z w||^2 s.t. ||w||_1 <= Q (no sign restriction), exactly.

    The constrained solution lies on the (piecewise-linear) lasso path; the
    path knot bracketing ``||w||_1 = Q`` is found with ``lars_path`` and the
    final point is re-solved from its KKT system on the active set.
    """
    from sklearn.linear_model import lars_path

    H = Z.T @ Z
    f = Z.T @ A
    w_ols = np.linalg.lstsq(Z, A, rcond=None)[0]
    if np.linalg.matrix_rank(Z) == Z.shape[1] and np.abs(w_ols).sum() <= Q:
        return w_ols
    _, _, coefs = lars_path(Z, A, method="lasso", alpha_min=0.0, max_iter=10000)
    norms = np.abs(coefs).sum(axis=0)
    if Q >= norms[-1]:  # path ends (interpolation / OLS) inside the ball
        return coefs[:, -1]
    k = int(np.searchsorted(norms, Q, side="right")) - 1
    k = min(max(k, 0), coefs.shape[1] - 2)
    t = (Q - norms[k]) / (norms[k + 1] - norms[k])
    w = coefs[:, k] + t * (coefs[:, k + 1] - coefs[:, k])
    # KKT polish on the support with the path's sign pattern
    tol = 1e-10 * max(1.0, np.abs(w).max())
    S = np.abs(w) > tol
    s = np.sign(w[S])
    nS = int(S.sum())
    K = np.zeros((nS + 1, nS + 1))
    K[:nS, :nS] = H[np.ix_(S, S)]
    K[:nS, nS] = s
    K[nS, :nS] = s
    try:
        sol = np.linalg.solve(K, np.concatenate([f[S], [Q]]))
    except np.linalg.LinAlgError:  # pragma: no cover
        return w
    cand = np.zeros_like(w)
    cand[S] = sol[:nS]
    nu = sol[nS]
    grad = f - H @ cand
    ok = (
        nu >= -1e-9 * max(1.0, abs(nu))
        and np.all(np.sign(cand[S]) == s)
        and np.all(np.abs(grad[~S]) <= nu * (1 + 1e-8) + 1e-10)
    )
    return cand if ok else w


def ridge_ball(Z: np.ndarray, A: np.ndarray, Q: float) -> np.ndarray:
    """min ||A - Z w||^2 s.t. ||w||_2 <= Q via the secular equation (exact).

    If the (minimum-norm) least-squares solution lies inside the ball it is
    returned (unique when Z has full column rank); otherwise the bound binds
    and w = (Z'Z + lam I)^{-1} Z'A with ||w|| = Q, lam > 0 by Brent's method.
    """
    H = Z.T @ Z
    f = Z.T @ A
    d, V = np.linalg.eigh(H)
    tol = 1e-12 * max(d.max(), 1e-300)
    pos = d > tol
    c = V.T @ f
    c[~pos] = 0.0  # f lies in range(Z'): numerical noise only
    d = np.where(pos, d, 0.0)
    w_mn = V[:, pos] @ (c[pos] / d[pos])
    if np.linalg.norm(w_mn) <= Q:
        return w_mn

    def phi(lam: float) -> float:
        return float(np.linalg.norm(c[pos] / (d[pos] + lam)) - Q)

    hi = max(np.linalg.norm(c) / Q, 1e-300)
    lo = hi
    while phi(lo) < 0:
        lo /= 16.0
    lam = optimize.brentq(phi, lo, hi, xtol=1e-300, rtol=4 * _EPS, maxiter=500)
    return V @ (c / (d + lam))


def simplex_l2(Z: np.ndarray, A: np.ndarray, Q2: float) -> np.ndarray:
    """min ||A - Z w||^2 s.t. w >= 0, sum(w) = 1, ||w||_2 <= Q2 (exact).

    If the simplex solution violates the L2 bound, the bound binds and the
    solution is the simplex solution of the ridge-penalised problem whose
    penalty makes ||w||_2 = Q2 (monotone in the penalty; Brent root).
    """
    J = Z.shape[1]
    if Q2 < 1.0 / np.sqrt(J) - 1e-12:
        raise ValueError(
            f"L1-L2 constraint infeasible: Q2={Q2:.6g} < "
            f"1/sqrt(J)={1 / np.sqrt(J):.6g}."
        )
    H = Z.T @ Z
    f = Z.T @ A
    lb = np.zeros(J)
    w = qp_bounded_sum(H, f, lb, 1.0)
    if np.linalg.norm(w) <= Q2:
        return w

    def phi(mu: float) -> float:
        return float(
            np.linalg.norm(qp_bounded_sum(H + mu * np.eye(J), f, lb, 1.0)) - Q2
        )

    hi = max(1.0, float(np.trace(H)))
    while phi(hi) > 0:
        hi *= 10.0
    mu = optimize.brentq(phi, 0.0, hi, xtol=1e-300, rtol=4 * _EPS, maxiter=500)
    return qp_bounded_sum(H + mu * np.eye(J), f, lb, 1.0)


def lm_coef(X: np.ndarray, y: np.ndarray, tol: float = 1e-7) -> Tuple[np.ndarray, int]:
    """OLS coefficients with R ``lm.fit`` aliasing (NaN for dropped columns).

    Columns are kept greedily in their given order unless (numerically) in the
    span of the kept ones -- the column-pivoting rule of LINPACK ``dqrdc2``
    that ``lm.fit`` uses with ``tol = 1e-7``.
    """
    n, p = X.shape
    keep = []
    for j in range(p):
        col = X[:, j]
        nrm = np.linalg.norm(col)
        if nrm == 0:
            continue
        if keep:
            Xk = X[:, keep]
            resid = col - Xk @ np.linalg.lstsq(Xk, col, rcond=None)[0]
            if np.linalg.norm(resid) <= tol * nrm:
                continue
        keep.append(j)
    coef = np.full(p, np.nan)
    if keep:
        coef[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    return coef, len(keep)


def shrinkage_ridge(
    A: np.ndarray, Z: np.ndarray, J: int, KM: int = 0
) -> Dict[str, float]:
    """R ``shrinkage.EST('ridge', ...)`` (V = identity): the ridge radius Q.

    lambda = sigma^2 (J + KM) / ||b_ols||^2 and Q = ||b_ols|| / (1 + lambda),
    with ``b_ols`` from ``lm.wfit``.  When the regression is (nearly)
    saturated (``nrow(Z) <= ncol(Z) + 10``) the donors are first screened with
    the ||w||_1 <= 1 problem (|w_j| > 1e-8), keeping at most max(T0 - 10, 2).
    """

    def _q(Zs: np.ndarray, kcols: int) -> Tuple[float, float]:
        coef, rank = lm_coef(Zs, A)
        df_res = Zs.shape[0] - rank
        if df_res > 0:
            fitted = Zs[:, ~np.isnan(coef)] @ coef[~np.isnan(coef)]
            sig = float(np.sqrt(np.sum((A - fitted) ** 2) / df_res))
        else:
            sig = np.nan
        ss = float(np.nansum(coef**2))
        lam = sig**2 * (kcols + KM) / ss
        return float(np.sqrt(ss) / (1.0 + lam)), float(lam)

    Q, lam = _q(Z, J + KM)
    if np.isnan(Q) or Z.shape[0] <= Z.shape[1] + 10:
        w_l1 = lasso_ball(Z, A, 1.0)
        active = np.abs(w_l1) > 1e-8
        cap = max(A.shape[0] - 10, 2)
        if active.sum() >= cap:
            # R: rank(-abs(w)) <= cap (average ranks on ties)
            from scipy.stats import rankdata

            active = rankdata(-np.abs(w_l1)) <= cap
        Zs = Z[:, active]
        Q, lam = _q(Zs, Zs.shape[1])
    return {"Q": Q, "lambda": lam}


# ---------------------------------------------------------------------- #
#  In-sample simulation problems
# ---------------------------------------------------------------------- #


def _min_linear_one_quad(g, M, b, k):
    """min g'z s.t. z'Mz + 2b'z + k <= 0 for positive semi-definite M.

    Returns ``(z, mu)``; ``("ray", d)`` when M is singular and the objective
    decreases along a direction ``d`` of its null space (the constraint set is
    a cylinder there: the caller must stop at the first blocking bound); or
    ``None`` if the set is empty.
    """
    try:
        np.linalg.cholesky(M)  # positive-definiteness check
        sol = np.linalg.solve(M, np.column_stack([b, g]))
        Mb, Mg = sol[:, 0], sol[:, 1]
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(M)
        tol = 1e-10 * max(vals.max(), 1e-300)
        null = vals <= tol
        Vn = vecs[:, null]
        gn = Vn.T @ g
        if np.linalg.norm(gn) > 1e-12 * max(np.linalg.norm(g), 1e-300):
            return "ray", -(Vn @ gn)
        Vr, inv = vecs[:, ~null], 1.0 / vals[~null]
        Mb = Vr @ (inv * (Vr.T @ b))
        Mg = Vr @ (inv * (Vr.T @ g))
    r2 = b @ Mb - k
    if r2 < 0:
        if r2 < -1e-12 * (1.0 + abs(k) + abs(b @ Mb)):
            return None
        r2 = 0.0
    t2 = g @ Mg
    if t2 <= 0 or r2 == 0:
        return -Mb, 0.0
    t, r = np.sqrt(t2), np.sqrt(r2)
    return -Mb - r * Mg / t, t / (2 * r)


def _quad_val(z, M, b, k):
    return z @ M @ z + 2 * b @ z + k


def _min_linear_two_quads(g, q1, q2):
    """min g'z s.t. q1(z) <= 0, q2(z) <= 0 (both convex, M1 PD, M2 PD).

    Single-constraint closed forms first.  If both bind, the KKT point is
    the closed-form solution of the aggregated constraint q1 + t q2 <= 0 at
    the ratio t = mu2 / mu1 > 0 where q2 = 0; psi(t) = q2(z(t)) is
    continuous and decreasing (positive as t -> 0, negative as t -> inf),
    so t is found by Brent's method on log t to machine precision.
    """
    (M1, b1, k1), (M2, b2, k2) = q1, q2
    s1 = 1.0 + abs(k1) + np.abs(b1).sum()
    s2 = 1.0 + abs(k2) + np.abs(b2).sum()
    o1 = _min_linear_one_quad(g, M1, b1, k1)
    if o1 is not None and not isinstance(o1[0], str):
        if _quad_val(o1[0], M2, b2, k2) <= 1e-13 * s2:
            return o1[0], (o1[1], 0.0)
    o2 = _min_linear_one_quad(g, M2, b2, k2)
    if o2 is not None and not isinstance(o2[0], str):
        if _quad_val(o2[0], M1, b1, k1) <= 1e-13 * s1:
            return o2[0], (0.0, o2[1])

    def solve_t(t):
        return _min_linear_one_quad(g, M1 + t * M2, b1 + t * b2, k1 + t * k2)

    def psi(logt):
        out = solve_t(np.exp(logt))
        if out is None or isinstance(out[0], str):  # pragma: no cover
            raise FloatingPointError
        return _quad_val(out[0], M2, b2, k2)

    try:
        lo, hi = -5.0, 5.0
        f_lo, f_hi = psi(lo), psi(hi)
        while f_lo <= 0 and lo > -700:
            lo -= 10.0
            f_lo = psi(lo)
        while f_hi >= 0 and hi < 700:
            hi += 10.0
            f_hi = psi(hi)
        if f_lo <= 0 or f_hi >= 0:
            return None
        logt = optimize.brentq(psi, lo, hi, xtol=1e-15, rtol=4 * _EPS, maxiter=500)
    except FloatingPointError:  # pragma: no cover
        return None
    t = np.exp(logt)
    z, mu1 = solve_t(t)
    return z, (mu1, t * mu1)


def _subproblem(c, Qm, G, k0, ell, fixed, sum_on, sum_val, ball):
    """Minimise c'y over {q(y) <= 0, y[fixed] = ell[fixed], [1'y = sum_val],
    [ball]} in closed form / by the two-multiplier dual Newton."""
    n = c.shape[0]
    free = ~fixed
    nf = int(free.sum())
    if nf == 0:
        return None
    u0 = np.zeros(n)
    u0[fixed] = ell[fixed]
    idx = np.where(free)[0]
    if sum_on:
        u0[free] = (sum_val - ell[fixed].sum()) / nf
        m = nf - 1
    else:
        m = nf
    if m == 0:
        return u0, (0.0, 0.0)
    # D maps z (free coordinates; with the sum on, the last free coordinate
    # is minus the sum of the others) to y - u0.  Products with D are formed
    # by indexing instead of dense matrix multiplication.
    Qu = Qm @ u0
    v = Qu - G
    k1 = u0 @ Qu - 2 * G @ u0 + k0
    if sum_on:
        a_, l_ = idx[:m], idx[-1]
        Qaa = Qm[np.ix_(a_, a_)]
        Qal = Qm[a_, l_]
        M1 = Qaa - Qal[:, None] - Qal[None, :] + Qm[l_, l_]
        b1 = v[a_] - v[l_]
        g = c[a_] - c[l_]

        def lift(z):
            out = np.zeros(n)
            out[a_] = z
            out[l_] = -z.sum()
            return out

    else:
        M1 = Qm[np.ix_(idx, idx)]
        b1 = v[idx]
        g = c[idx]

        def lift(z):
            out = np.zeros(n)
            out[idx] = z
            return out

    if ball is None:
        out = _min_linear_one_quad(g, M1, b1, k1)
        if out is None:
            return None
        if isinstance(out[0], str):
            return "ray", lift(out[1])
        return u0 + lift(out[0]), (out[1], 0.0)
    a, R = ball
    au = a + u0
    if sum_on:
        M2 = np.eye(m) + 1.0
        b2 = au[a_] - au[l_]
    else:
        M2 = np.eye(m)
        b2 = au[idx]
    k2 = au @ au - R**2
    out = _min_linear_two_quads(g, (M1, b1, k1), (M2, b2, k2))
    if out is None:
        return None
    return u0 + lift(out[0]), out[1]


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
) -> Optional[np.ndarray]:
    """min c'y s.t. y'Qy - 2G'y + k0 <= 0, y >= ell, [1'y (=|<=) sum_val],
    [||a + y||^2 <= R^2]: primal active-set method over the bound / sum
    constraints.

    Starting from the feasible point ``y0`` (default 0, i.e. x = beta), each
    iteration solves the problem with the working set held at equality
    (closed form for one quadratic, dual Newton with the ball), then either
    steps towards that solution until the first blocking bound (which joins
    the working set; convexity keeps every iterate feasible) or, at the
    subproblem optimum, releases the constraint with the most negative
    multiplier.  The returned point satisfies the full KKT system (convex
    problem => globally optimal); ``None`` sends the caller to SLSQP.
    """
    n = c.shape[0]
    finite = np.isfinite(ell)
    y = np.zeros(n) if y0 is None else np.asarray(y0, float).copy()
    if sum_mode == "eq":
        gap = sum_val - y.sum()
        if gap != 0.0:
            k = (
                int(np.argmin(np.where(finite, ell - y, -np.inf)))
                if finite.any()
                else 0
            )
            y[k] += gap
    if np.any(finite & (y < ell - 1e-12)):
        return None
    fixed = finite & (np.abs(y - ell) <= 1e-15 * (1 + np.abs(ell)))
    sum_on = sum_mode == "eq" or (
        sum_mode == "le" and abs(y.sum() - sum_val) <= 1e-15 * (1 + abs(sum_val))
    )
    scale_c = max(np.abs(c).max(), 1e-300)
    for _ in range(8 * n + 20):
        sub = _subproblem(c, Qm, G, k0, ell, fixed, sum_on, sum_val, ball)
        if sub is None:
            return None
        free = ~fixed
        if isinstance(sub[0], str):  # descent along the null space of Q
            d = sub[1]
            alpha, block, block_sum = np.inf, -1, False
        else:
            ystar, mus = sub
            d = ystar - y
            alpha, block, block_sum = 1.0, -1, False
        cand = np.where(free & finite & (d < 0))[0]
        if cand.size:
            ratios = (ell[cand] - y[cand]) / d[cand]
            j = int(np.argmin(ratios))
            if ratios[j] < alpha:
                alpha, block = max(ratios[j], 0.0), int(cand[j])
        if sum_mode == "le" and not sum_on and d.sum() > 0:
            rs = (sum_val - y.sum()) / d.sum()
            if rs < alpha:
                alpha, block, block_sum = max(rs, 0.0), -1, True
        if not np.isfinite(alpha):
            return "unbounded"
        if alpha < 1.0 or isinstance(sub[0], str):
            y = y + alpha * d
            if block_sum:
                sum_on = True
            else:
                fixed[block] = True
                y[block] = ell[block]
            continue
        y = ystar
        grad = c + mus[0] * (2 * Qm @ y - 2 * G)
        if ball is not None:
            grad = grad + mus[1] * 2 * (ball[0] + y)
        nu = -np.mean(grad[free]) if sum_on else 0.0
        lam = grad + nu
        worst = None
        if fixed.any():
            fidx = np.where(fixed)[0]
            worst = int(fidx[np.argmin(lam[fidx])])
        release_sum = sum_mode == "le" and sum_on and nu < -1e-9 * scale_c
        if (
            worst is not None
            and lam[worst] < -1e-9 * scale_c
            and (not release_sum or lam[worst] <= nu)
        ):
            fixed[worst] = False
            continue
        if release_sum:
            sum_on = False
            continue
        return y
    return None


def insample_slsqp(
    c: np.ndarray,
    Qm: np.ndarray,
    G: np.ndarray,
    beta: np.ndarray,
    constr: Dict,
    lb: np.ndarray,
    Q1: Optional[float],
    Q2: Optional[float],
) -> Optional[np.ndarray]:
    """Generic SLSQP solve of the in-sample problem (lasso / ridge / fallback).

    Variables are ``y = x - beta``; for the L1 ball ``x`` is split into its
    positive and negative parts.  Returns ``y`` or ``None`` on failure.
    """
    J = beta.shape[0]
    p, direction = constr["p"], constr["dir"]
    qs = max(float(np.abs(Qm).max()), 1e-300)
    cs = max(float(np.abs(c).max()), 1e-300)
    l1 = p == "L1" and direction == "<="
    if l1:
        xpos = np.clip(beta, 0, None)
        xneg = np.clip(-beta, 0, None)
        z0 = np.concatenate([xpos, xneg])

        def to_y(z):
            return z[:J] - z[J:] - beta

        def dy(g):  # chain rule dy/dz
            return np.concatenate([g, -g])

        bounds = [(0.0, None)] * (2 * J)
    else:
        z0 = np.zeros(J)

        def to_y(z):
            return z

        def dy(g):
            return g

        bounds = [
            (lb[j] - beta[j] if np.isfinite(lb[j]) else None, None) for j in range(J)
        ]
    cons = [
        {
            "type": "ineq",
            "fun": lambda z: -(to_y(z) @ Qm @ to_y(z) - 2 * G @ to_y(z)) / qs,
            "jac": lambda z: dy(-(2 * Qm @ to_y(z) - 2 * G) / qs),
        }
    ]
    if l1:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda z: Q1 - z.sum(),
                "jac": lambda z: -np.ones(2 * J),
            }
        )
    if (p == "L1" and direction == "==") or p == "L1-L2":
        tot = (Q1 if Q1 is not None else 1.0) - beta.sum()
        cons.append(
            {
                "type": "eq",
                "fun": lambda z: to_y(z).sum() - tot,
                "jac": lambda z: dy(np.ones(J)),
            }
        )
    if p in ("L2", "L1-L2"):
        rad = Q2 if p == "L1-L2" else Q1
        cons.append(
            {
                "type": "ineq",
                "fun": lambda z: rad**2 - np.sum((to_y(z) + beta) ** 2),
                "jac": lambda z: dy(-2 * (to_y(z) + beta)),
            }
        )
    res = optimize.minimize(
        lambda z: c @ to_y(z) / cs,
        z0,
        jac=lambda z: dy(c / cs),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-15},
    )
    if not res.success:
        return None
    return to_y(res.x)


# ---------------------------------------------------------------------- #
#  Quantile regression (quantreg::rq.fit, Qtools::rrq)
# ---------------------------------------------------------------------- #


def rq_fit(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Exact linear-programming quantile regression (vertex solution).

    Solved with HiGHS and polished by re-solving the basic observations
    (zero residuals) exactly.  ``quantreg``'s default Frisch-Newton interior
    point returns the same vertex up to its duality-gap tolerance (1e-6)
    whenever the LP optimum is unique.
    """
    n, p = X.shape
    cost = np.concatenate([np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = optimize.linprog(cost, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:  # pragma: no cover
        raise RuntimeError(f"quantile regression LP failed: {res.message}")
    beta = res.x[:p]
    r = y - X @ beta
    order = np.argsort(np.abs(r))
    basis = order[:p]
    Xb = X[basis]
    if np.linalg.matrix_rank(Xb) == p:
        cand = np.linalg.solve(Xb, y[basis])

        def obj(b):
            u = y - X @ b
            return np.sum(u * (tau - (u < 0)))

        if obj(cand) <= obj(beta) + 1e-12 * max(1.0, abs(obj(beta))):
            beta = cand
    return beta


def rrq(X: np.ndarray, y: np.ndarray, taus) -> np.ndarray:
    """``Qtools::rrq`` restricted regression quantiles (He, 1997 location-scale).

    beta = LAD(y ~ X); gamma = LAD(|r| ~ X); zeta_k = rq(r ~ s - 1, tau_k)
    with s = X gamma; coefficient column k = beta + zeta_k * gamma.
    """
    beta = rq_fit(X, y, 0.5)
    r = y - X @ beta
    gamma = rq_fit(X, np.abs(r), 0.5)
    s = X @ gamma
    out = np.empty((X.shape[1], len(taus)))
    for k, tau in enumerate(taus):
        zeta = rq_fit(s[:, None], r, float(tau))[0]
        out[:, k] = beta + zeta * gamma
    return out


def warn_failed(n_failed: int, n_total: int) -> None:
    if n_total and n_failed / n_total > 0.2:
        warnings.warn(
            f"{n_failed}/{n_total} in-sample simulation problems could not be "
            "solved; the in-sample bounds use the remaining draws.",
            RuntimeWarning,
            stacklevel=3,
        )
