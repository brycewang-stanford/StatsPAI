"""Energy balancing weights for categorical treatments (``sp.weightit(method='energy')``).

The weights minimise the energy distance between each weighted treatment
group and the target distribution: the full sample (ATE) or the focal
group (ATT). With ``improved=True`` (ATE) the energy distances between
every pair of weighted groups are added, the improved energy balancing
weights of Huling and Mak (2024). Weights are non-negative (at least
``min_w``) and sum to each group's size; ``lam`` penalises their squares.

The objective was written from the method's definition and its scaling
fixed against R ``WeightIt`` 1.7.0 output (GPL: used as a black box,
documentation and outputs only). With ``v_t = w_t / n_t`` the weighted
distribution of group ``t`` and ``E(., .)`` the energy distance
``2 E|X - Y| - E|X - X'| - E|Y - Y'|`` under the distance matrix:

* ATE, improved: ``sum_t [E(v_t, F_n) + lam/2 |v_t|^2] + sum_{t<s} E(v_t, v_s)``;
* ATE, standard: ``sum_t (n_t / n)^2 [E(v_t, F_n) + lam/2 |v_t|^2]``;
* ATT: ``sum_{t != focal} [E(v_t, F_focal) + lam/2 |v_t|^2]``.

The quadratic programme is solved exactly: a primal-dual active-set
iteration on the bound constraints, each step the KKT system of the free
weights, with a primal active-set method as fallback; the result
satisfies the optimality conditions to rounding error.

References
----------
[@huling2024energy]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from ..exceptions import ConvergenceFailure, MethodIncompatibility

DISTANCES = ("scaled_euclidean", "mahalanobis", "euclidean")


def distance_matrix(X: np.ndarray, dist: str) -> np.ndarray:
    """Pairwise distances between the rows of ``X``.

    ``'scaled_euclidean'`` divides each column by its standard deviation
    (divisor ``n - 1``), ``'mahalanobis'`` whitens with the sample
    covariance, ``'euclidean'`` uses the columns as they are.
    """
    if dist == "euclidean":
        Z = X
    elif dist == "scaled_euclidean":
        sd = X.std(axis=0, ddof=1)
        Z = X / np.where(sd > 0, sd, 1.0)
    elif dist == "mahalanobis":
        S = np.atleast_2d(np.cov(X, rowvar=False))
        Z = X @ np.linalg.cholesky(np.linalg.pinv(S))
    else:
        raise MethodIncompatibility(
            f"weightit: dist must be one of {DISTANCES}; got {dist!r}.",
            recovery_hint="Use 'scaled_euclidean' (the default).",
        )
    return cdist(Z, Z)


def energy_weights(
    X: np.ndarray,
    G: np.ndarray,
    K: int,
    estimand: str,
    focal: Optional[int],
    *,
    improved: bool = True,
    lam: float = 1e-4,
    dist: str = "scaled_euclidean",
    min_w: float = 1e-8,
    M: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Energy balancing weights; ``M`` holds terms whose weighted means are
    set exactly to the target means (WeightIt ``moments`` / ``int``)."""
    n = len(G)
    D = distance_matrix(X, dist)
    sizes = np.bincount(G, minlength=K).astype(float)
    if estimand == "ATE":
        weighted = list(range(K))
        target = np.full(n, 1.0 / n)
    else:
        assert focal is not None
        weighted = [k for k in range(K) if k != focal]
        target = (G == focal) / sizes[focal]
    free = np.flatnonzero(G != focal) if estimand != "ATE" else np.arange(n)
    g = G[free]
    Dff = D[np.ix_(free, free)]
    Dtarget = D[free] @ target
    nf = len(free)
    # objective 1/2 w' P w + c' w over the weights of the free units
    Pm = np.zeros((nf, nf))
    c = np.zeros(nf)
    a = {k: (g == k) / sizes[k] for k in weighted}  # w -> v_k
    scale = {k: 1.0 for k in weighted}
    if estimand == "ATE" and not improved:
        scale = {k: (sizes[k] / n) ** 2 for k in weighted}
    for k in weighted:
        s = scale[k]
        Pm -= 2 * s * np.outer(a[k], a[k]) * Dff
        Pm += s * lam * np.diag(a[k] ** 2)
        c += 2 * s * a[k] * Dtarget
    if estimand == "ATE" and improved:
        for i, k in enumerate(weighted):
            for m in weighted[i + 1 :]:
                cross = np.outer(a[k], a[m]) * Dff
                Pm += 2 * (cross + cross.T)
                Pm -= 2 * np.outer(a[k], a[k]) * Dff
                Pm -= 2 * np.outer(a[m], a[m]) * Dff
    Pm = 0.5 * (Pm + Pm.T)
    rows = [(g == k).astype(float) for k in weighted]
    rhs = [sizes[k] for k in weighted]
    if M is not None and M.shape[1]:
        tmean = target @ M
        Mf = M[free]
        for k in weighted:
            for j in range(M.shape[1]):
                rows.append(a[k] * Mf[:, j])
                rhs.append(tmean[j])
    A = np.array(rows)
    b = np.array(rhs, dtype=float)
    lb = np.full(nf, float(min_w))
    w_free = solve_qp(Pm, c, A, b, lb)
    w = np.ones(n)
    w[free] = w_free
    return w


def _kkt_solve(
    P: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    at_bound: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimiser with the ``at_bound`` weights fixed at ``lb`` and the
    equality constraints; returns the weights and the multipliers ``nu``
    (stationarity: ``P w + c + A' nu = mu``, ``mu`` zero off the bound)."""
    F = ~at_bound
    w = lb.copy()
    m = A.shape[0]
    nF = int(F.sum())
    PF = P[np.ix_(F, F)]
    AF = A[:, F]
    r1 = -(c[F] + P[np.ix_(F, at_bound)] @ lb[at_bound])
    r2 = b - A[:, at_bound] @ lb[at_bound]
    KKT = np.block([[PF, AF.T], [AF, np.zeros((m, m))]])
    rhs = np.concatenate([r1, r2])
    try:
        sol = np.linalg.solve(KKT, rhs)
        if not np.all(np.isfinite(sol)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    w[F] = sol[:nF]
    return w, sol[nF:]


def _kkt_ok(
    P: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lb: np.ndarray,
    w: np.ndarray,
    nu: np.ndarray,
    at_bound: np.ndarray,
) -> bool:
    grad = P @ w + c + A.T @ nu
    tol = 1e-9 * max(1.0, float(np.abs(c).max()), float(np.abs(P).max()))
    return bool(
        np.all(w[~at_bound] >= lb[~at_bound])
        and np.all(grad[at_bound] >= -tol)
        and np.abs(A @ w - b).max() <= 1e-9 * max(1.0, float(np.abs(b).max()))
    )


def solve_qp(
    P: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, lb: np.ndarray
) -> np.ndarray:
    """``min 1/2 w'Pw + c'w`` subject to ``A w = b`` and ``w >= lb``.

    ``P`` need only be positive definite on the null space of ``A`` (the
    energy distance is a conditionally negative definite form). A
    primal-dual active-set iteration (bound set = rows whose weight would
    fall below ``lb`` or whose multiplier is positive) usually ends in a
    few steps; if it cycles, a primal active-set method from the feasible
    point finishes the job.
    """
    n = len(c)
    at_bound = np.zeros(n, dtype=bool)
    for _ in range(100):
        w, nu = _kkt_solve(P, c, A, b, lb, at_bound)
        mu = np.where(at_bound, P @ w + c + A.T @ nu, 0.0)
        new = (at_bound & (mu > 0)) | (w < lb)
        if np.array_equal(new, at_bound):
            if _kkt_ok(P, c, A, b, lb, w, nu, at_bound):
                return w
            break
        at_bound = new
    return _primal_active_set(P, c, A, b, lb)


def _primal_active_set(
    P: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray, lb: np.ndarray
) -> np.ndarray:
    """Primal active-set method (one bound enters or leaves per step)."""
    n = len(c)
    # feasible start: the equality-constrained point closest to uniform
    # weights that stays above the bounds
    w0 = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.any(w0 < lb):
        from scipy.optimize import linprog

        res = linprog(
            np.zeros(n),
            A_eq=A,
            b_eq=b,
            bounds=list(zip(lb, [None] * n)),
            method="highs",
        )
        if not res.success:
            raise MethodIncompatibility(
                "weightit: no non-negative weights satisfy the exact balance "
                "constraints.",
                recovery_hint="Lower moments= or drop interactions.",
            )
        w0 = np.maximum(res.x, lb)
    w = w0
    at_bound = w <= lb
    w[at_bound] = lb[at_bound]
    for _ in range(50 * n + 100):
        target, nu = _kkt_solve(P, c, A, b, lb, at_bound)
        step = target - w
        if np.abs(step).max() <= 1e-12 * max(1.0, float(np.abs(w).max())):
            grad = P @ target + c + A.T @ nu
            mu = np.where(at_bound, grad, np.inf)
            j = int(np.argmin(mu))
            tol = 1e-9 * max(1.0, float(np.abs(c).max()), float(np.abs(P).max()))
            if mu[j] >= -tol:
                return target
            at_bound[j] = False
            w = target
            continue
        neg = (~at_bound) & (step < 0)
        ratios = np.full(n, np.inf)
        ratios[neg] = (lb[neg] - w[neg]) / step[neg]
        j = int(np.argmin(ratios))
        alpha = min(1.0, float(ratios[j]))
        w = w + alpha * step
        if alpha < 1.0:
            at_bound[j] = True
            w[j] = lb[j]
    raise ConvergenceFailure(
        "weightit: the energy balancing programme did not converge.",
        recovery_hint="Raise lam= to make the problem better conditioned.",
    )
