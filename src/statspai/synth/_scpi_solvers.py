"""Exact solvers for the scpi family (private to ``scpi.py``).

Clean-room implementation written from the method papers
[@cattaneo2021prediction; @cattaneo2025scpi] and the R ``scpi`` help pages,
with R ``scpi`` (GPL-3) used only as a black box; see
``docs/dev/cleanroom/scpi.md``. No reference source was read.

Weight problems (``V`` = identity, outcome feature only)::

    simplex   min ||A - Z w||^2  s.t.  w >= 0, sum(w) = 1
    lasso     min ||A - Z w||^2  s.t.  ||w||_1 <= Q
    ridge     min ||A - Z w||^2  s.t.  ||w||_2 <= Q
    L1-L2     min ||A - Z w||^2  s.t.  w >= 0, sum(w) = 1, ||w||_2 <= Q2
    ols       min ||A - Z w||^2

In-sample simulation problem (one per draw, horizon and side)::

    min c'y  s.t.  y'M y - 2 g'y + k <= 0,  y in a polyhedron or a ball

where ``y`` is the deviation of the weights from their estimate. Everything
is solved to machine precision: closed forms on a fixed active set, an exact
primal active-set method for the bound constraints, 1-D monotone root
searches for the norm balls, and an exact LP for quantile regression.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import optimize

from ..exceptions import ConvergenceFailure, MethodIncompatibility
from ._scpi_insample import (  # noqa: F401  (re-exported for scpi.py and the tests)
    _min_linear_one_quad,
    _min_linear_two_quads,
    _quad_val,
    _subproblem,
    insample_active_set,
    warn_failed,
)
from ._scpi_qp import _equality_step, _sum_nullspace, qp_bounded_sum  # noqa: F401
from ._scpi_simulate import (  # noqa: F401
    geometry_from_constraint,
    insample_slsqp,
    insample_solve,
)

_EPS = np.finfo(float).eps

# ---------------------------------------------------------------------- #
#  Weight estimation
# ---------------------------------------------------------------------- #


def _least_squares_min_norm(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.lstsq(Z, A, rcond=None)[0], dtype=float)


def _min_l1_least_squares(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Among the least-squares solutions, one with the smallest L1 norm (a vertex)."""
    J = Z.shape[1]
    w_mn = _least_squares_min_norm(Z, A)
    if np.linalg.matrix_rank(Z) == J:
        return w_mn
    fitted = Z @ w_mn
    c = np.ones(2 * J)
    res = optimize.linprog(
        c,
        A_eq=np.hstack([Z, -Z]),
        b_eq=fitted,
        bounds=[(0, None)] * (2 * J),
        method="highs-ds",
    )
    if res.status != 0:  # pragma: no cover - LP with a known feasible point
        return w_mn
    w = res.x[:J] - res.x[J:]
    support = np.abs(w) > 0
    if support.any():  # polish the vertex on its support
        sol = np.linalg.lstsq(Z[:, support], fitted, rcond=None)[0]
        w = np.zeros(J)
        w[support] = sol
    return np.asarray(w, dtype=float)


def lasso_ball(Z: np.ndarray, A: np.ndarray, Q: float) -> np.ndarray:
    """min ||A - Z w||^2 s.t. ||w||_1 <= Q (no sign restriction), exactly.

    When a least-squares solution lies inside the ball it is returned (the
    smallest-L1 one if least squares is not unique, which has at most
    rank(Z) non-zeros). Otherwise the ball binds and the problem is the
    sum-constrained QP in ``w = p - n`` with ``p, n >= 0``, ``sum(p+n) = Q``.
    """
    Z = np.asarray(Z, float)
    A = np.asarray(A, float)
    J = Z.shape[1]
    w0 = _min_l1_least_squares(Z, A)
    if np.abs(w0).sum() <= Q * (1 + 1e-12):
        return w0
    H = Z.T @ Z
    f = Z.T @ A
    H2 = np.block([[H, -H], [-H, H]])
    f2 = np.concatenate([f, -f])
    x = qp_bounded_sum(2 * H2, 2 * f2, np.zeros(2 * J), float(Q))
    return np.asarray(x[:J] - x[J:], dtype=float)


def ridge_ball(Z: np.ndarray, A: np.ndarray, Q: float) -> np.ndarray:
    """min ||A - Z w||^2 s.t. ||w||_2 <= Q via the secular equation (exact)."""
    Z = np.asarray(Z, float)
    A = np.asarray(A, float)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    keep = s > s.max() * max(Z.shape) * _EPS if s.size else s > 0
    U, s, Vt = U[:, keep], s[keep], Vt[keep]
    b = U.T @ A
    w_ls = Vt.T @ (b / s)
    if np.linalg.norm(w_ls) <= Q:
        return np.asarray(w_ls, dtype=float)

    def norm_gap(mu: float) -> float:
        return float(np.linalg.norm(s * b / (s**2 + mu)) - Q)

    hi = float(np.linalg.norm(s * b)) / Q
    mu = optimize.brentq(norm_gap, 0.0, hi, xtol=1e-300, rtol=4 * _EPS, maxiter=500)
    w = Vt.T @ (s * b / (s**2 + mu))
    return np.asarray(w * (Q / np.linalg.norm(w)), dtype=float)


def simplex_weights(Z: np.ndarray, A: np.ndarray, total: float = 1.0) -> np.ndarray:
    """min ||A - Z w||^2 s.t. w >= 0, sum(w) = total (exact)."""
    Z = np.asarray(Z, float)
    A = np.asarray(A, float)
    J = Z.shape[1]
    return qp_bounded_sum(2 * Z.T @ Z, 2 * Z.T @ A, np.zeros(J), float(total))


def simplex_l2(Z: np.ndarray, A: np.ndarray, Q2: float) -> np.ndarray:
    """min ||A - Z w||^2 s.t. w >= 0, sum(w) = 1, ||w||_2 <= Q2 (exact)."""
    Z = np.asarray(Z, float)
    A = np.asarray(A, float)
    J = Z.shape[1]
    floor = 1.0 / np.sqrt(J)
    if Q2 < floor * (1 - 1e-12):
        raise MethodIncompatibility(
            f"L1-L2 constraint infeasible: Q2 = {Q2:.6g} is below 1/sqrt(J) = "
            f"{floor:.6g}, the smallest L2 norm of weights that sum to one.",
            recovery_hint="Raise Q2 to at least 1/sqrt(number of donors).",
            diagnostics={"Q2": float(Q2), "min_norm": float(floor)},
        )
    H = Z.T @ Z
    f = Z.T @ A
    w = qp_bounded_sum(2 * H, 2 * f, np.zeros(J), 1.0)
    if np.linalg.norm(w) <= Q2:
        return w
    if Q2 <= floor:
        return np.full(J, 1.0 / J)
    cache: Dict[float, np.ndarray] = {}

    def solve(mu: float) -> np.ndarray:
        if mu not in cache:
            cache[mu] = qp_bounded_sum(
                2 * (H + mu * np.eye(J)), 2 * f, np.zeros(J), 1.0
            )
        return cache[mu]

    def gap(mu: float) -> float:
        return float(np.linalg.norm(solve(mu)) - Q2)

    hi = max(1.0, float(np.abs(H).max()))
    while gap(hi) > 0:
        hi *= 4.0
        if hi > 1e30:  # pragma: no cover - uniform weights reach any Q2 > floor
            break
    mu = optimize.brentq(gap, 0.0, hi, xtol=1e-300, rtol=4 * _EPS, maxiter=500)
    w = solve(mu)
    # exact polish on the active set: (H + mu I) w_F - f_F + nu 1 = 0,
    # sum(w_F) = 1, ||w_F|| = Q2, solved in mu by a 1-D root
    free = w > 0
    idx = np.flatnonzero(free)
    if len(idx) >= 2:
        HF, fF = H[np.ix_(idx, idx)], f[idx]
        m = len(idx)

        def wF(mu_: float) -> np.ndarray:
            K = np.zeros((m + 1, m + 1))
            K[:m, :m] = HF + mu_ * np.eye(m)
            K[:m, m] = 1.0
            K[m, :m] = 1.0
            rhs = np.concatenate([fF, [1.0]])
            return np.linalg.solve(K, rhs)[:m]

        try:
            lo_mu, hi_mu = mu * (1 - 1e-6), mu * (1 + 1e-6) + 1e-300
            g_lo = np.linalg.norm(wF(lo_mu)) - Q2
            g_hi = np.linalg.norm(wF(hi_mu)) - Q2
            if g_lo * g_hi < 0:
                mu2 = optimize.brentq(
                    lambda m_: np.linalg.norm(wF(m_)) - Q2,
                    lo_mu,
                    hi_mu,
                    xtol=1e-300,
                    rtol=4 * _EPS,
                )
                cand = np.zeros(J)
                cand[idx] = wF(mu2)
                if cand.min() >= 0:
                    w = cand
        except np.linalg.LinAlgError:  # pragma: no cover
            pass
    return w


def lm_coef(X: np.ndarray, y: np.ndarray, tol: float = 1e-7) -> Tuple[np.ndarray, int]:
    """Least-squares coefficients with linearly dependent columns aliased.

    Columns are taken in order; a column whose part orthogonal to the
    columns already kept has norm below ``tol`` times its own norm (or is
    zero) is aliased and gets ``NaN``, as in R's ``lm.fit`` documentation.
    Returns ``(coef, rank)``.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    kept = []
    basis = np.zeros((n, 0))
    for j in range(p):
        col = X[:, j]
        nrm = np.linalg.norm(col)
        if nrm == 0:
            continue
        resid = col - basis @ (basis.T @ col)
        resid = resid - basis @ (basis.T @ resid)
        if np.linalg.norm(resid) <= tol * nrm:
            continue
        kept.append(j)
        basis = np.column_stack([basis, resid / np.linalg.norm(resid)])
    coef = np.full(p, np.nan)
    if kept:
        coef[kept] = np.linalg.lstsq(X[:, kept], y, rcond=None)[0]
    return coef, len(kept)


def lm_fitted(
    X: np.ndarray, y: np.ndarray, X_new: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fitted values (and predictions at ``X_new``) of ``lm_coef``."""
    coef, _ = lm_coef(X, y)
    use = ~np.isnan(coef)
    fit = X[:, use] @ coef[use]
    pred = None if X_new is None else np.asarray(X_new, float)[:, use] @ coef[use]
    return fit, pred


def shrinkage_ridge(
    A: np.ndarray, Z: np.ndarray, J: int, KM: int = 0
) -> Dict[str, float]:
    """Data-driven ridge radius [@cattaneo2025scpi, Section 3.1].

    With least squares on the donors kept, ``sigma^2 = RSS / (T0 - k)``,
    ``lambda = k sigma^2 / ||w_ls||^2`` (the Hoerl-Kennard-Baldwin choice) and
    ``Q = ||w_ls|| / (1 + lambda)``. When ``T0 - J <= 10`` the design is first
    reduced: the donors with a non-zero weight in the L1-ball fit
    (``||w||_1 <= 1``) are kept, at most ``max(T0 - 10, 2)`` of them (the
    largest in absolute value). The floor at 0.5 is applied by the caller.
    """
    A = np.asarray(A, float)
    Z = np.asarray(Z, float)
    T0 = Z.shape[0]
    cols: np.ndarray = np.arange(Z.shape[1])
    if T0 - J <= 10:
        w1 = lasso_ball(Z, A, 1.0)
        nz = np.flatnonzero(np.abs(w1) > 1e-6)
        cap = max(T0 - 10, 2)
        if len(nz) > cap:
            order = nz[np.argsort(-np.abs(w1[nz]), kind="stable")]
            nz = np.sort(order[:cap])
        cols = nz if len(nz) else np.flatnonzero(np.abs(w1) == np.abs(w1).max())[:1]
    Zk = Z[:, cols]
    k = Zk.shape[1] + KM
    w_ls = np.linalg.lstsq(Zk, A, rcond=None)[0]
    rss = float(np.sum((A - Zk @ w_ls) ** 2))
    dof = T0 - k
    norm2 = float(w_ls @ w_ls)
    if dof <= 0 or norm2 == 0:
        lam = np.nan
        Q = float(np.sqrt(norm2))
    else:
        lam = k * (rss / dof) / norm2
        Q = float(np.sqrt(norm2) / (1.0 + lam))
    return {"Q": Q, "lambda": float(lam), "n_kept": int(Zk.shape[1])}


# ---------------------------------------------------------------------- #
#  Quantile regression: exact LP and restricted regression quantiles
# ---------------------------------------------------------------------- #


def rq_fit(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Linear quantile regression solved as an exact LP (a vertex solution)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    cost = np.concatenate([np.zeros(2 * p), np.full(n, tau), np.full(n, 1.0 - tau)])
    A_eq = np.hstack([X, -X, np.eye(n), -np.eye(n)])
    res = optimize.linprog(
        cost, A_eq=A_eq, b_eq=y, bounds=[(0, None)] * (2 * p + 2 * n), method="highs-ds"
    )
    if res.status != 0:  # pragma: no cover - the LP is always feasible and bounded
        raise ConvergenceFailure(
            f"Quantile regression LP failed: {res.message}",
            recovery_hint="Check the out-of-sample design for missing or "
            "infinite values.",
        )
    return np.asarray(res.x[:p] - res.x[p : 2 * p], dtype=float)


def rrq(X: np.ndarray, y: np.ndarray, taus: Any) -> np.ndarray:
    """Restricted regression quantiles (location-scale), one column per tau.

    The three steps of the restricted regression quantile: median regression
    ``b`` of ``y`` on ``X``; median regression ``g`` of ``|y - X b|`` on
    ``X``; for each ``tau`` the ``tau``-quantile regression ``c(tau)`` of the
    residuals on the fitted scale ``X g`` through the origin. The
    ``tau``-quantile is then ``X (b + c(tau) g)``, so quantiles never cross.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = rq_fit(X, y, 0.5)
    r = y - X @ b
    g = rq_fit(X, np.abs(r), 0.5)
    s = X @ g
    cols = [b + rq_fit(s[:, None], r, float(t))[0] * g for t in np.atleast_1d(taus)]
    return np.column_stack(cols)
