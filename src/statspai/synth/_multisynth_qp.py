"""Block-structured active-set solver for the multisynth weight QP.

The partially pooled SCM objective of ``sp.staggered_synth`` is the convex
quadratic ``0.5 z'Hz + c'z`` over the stacked weights ``z = (g_1, ..., g_J)``
subject to ``z >= 0`` and one adding-up constraint per treated unit / cohort.
Its Hessian has the structure::

    H = blockdiag(q_j M_j'M_j + lam I) + c_pool M'M,     M = [M_1 ... M_J]

where ``M_j`` is the (padded) ``max_dim x n0_j`` donor matrix of block ``j``
and ``M`` has only ``max_dim`` rows. The solver below never forms the
``N x N`` matrix (``N = sum_j n0_j``, tens of thousands in a 300-unit panel):

* reduced costs ``Hz + c + A'mu`` are computed blockwise in ``O(max_dim N)``;
* the equality-constrained subproblem on the free set is solved block by
  block, and the pooled coupling ``c_pool M'M`` -- rank ``max_dim`` -- is
  folded back in through a ``max_dim x max_dim`` Schur complement;
* a block's factorisation is recomputed only when its free set changes.

The iteration itself (vertex start, ratio test, entering rule, tolerances)
is the Lawson-Hanson style primal active-set method the estimator has
always used, so the iterates and the returned weights are those of the
former dense implementation up to linear-algebra round-off.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..exceptions import ConvergenceFailure

__all__ = ["solve_multisynth_qp"]


def solve_multisynth_qp(
    xs: Sequence[np.ndarray],
    Ms: Sequence[np.ndarray],
    n1: Sequence[int],
    ndims: Sequence[int],
    nu: float,
    lam: float,
    n_lags: int,
    norm_pool: float,
    norm_sep: float,
) -> List[np.ndarray]:
    """Minimise the multisynth objective; returns one weight vector per block.

    Weights are on the ``n_j`` scale (block ``j`` sums to ``n1[j]``).
    """
    J = len(xs)
    c_sep = (1.0 - nu) / (norm_sep * J)
    c_pool = nu / (norm_pool * J**2 * n_lags)
    x_sum = np.sum(xs, axis=0)
    q = np.array([c_sep / ndims[j] for j in range(J)], dtype=float)
    cs = []
    for j in range(J):
        cj = -q[j] * (Ms[j].T @ xs[j])
        if c_pool > 0:
            cj = cj - c_pool * (Ms[j].T @ x_sum)
        cs.append(cj)
    z = _active_set(list(Ms), q, np.concatenate(cs), np.asarray(n1, float), c_pool, lam)
    cuts = np.cumsum([0] + [M.shape[1] for M in Ms])
    return [z[cuts[j] : cuts[j + 1]] for j in range(J)]


class _Block:
    """Cached KKT solution of one block's free-set subproblem.

    For the free set ``F`` of block ``j`` it holds the solution of::

        [ q M_F'M_F + lam I   1 ] [z ]   [ -c_F + M_F' lam_pool ]
        [ 1'                  0 ] [mu] = [  b_j                 ]

    as an affine function of the pooled multiplier ``lam_pool``:
    ``z = z0 + Z lam_pool``, ``mu = mu0 + muZ . lam_pool``.
    """

    __slots__ = ("F", "z0", "Z", "mu0", "muZ", "singular", "p", "P")

    def __init__(
        self,
        F: np.ndarray,
        Mcat: np.ndarray,
        q: float,
        c: np.ndarray,
        b: float,
        lam: float,
        coupled: bool,
    ) -> None:
        nF = F.size
        MF = Mcat[:, F]
        K = np.zeros((nF + 1, nF + 1))
        K[:nF, :nF] = q * (MF.T @ MF)
        if lam:
            K[np.arange(nF), np.arange(nF)] += lam
        K[:nF, nF] = 1.0
        K[nF, :nF] = 1.0
        m = MF.shape[0]
        rhs = np.zeros((nF + 1, 1 + (m if coupled else 0)))
        rhs[:nF, 0] = -c[F]
        rhs[nF, 0] = b
        if coupled:
            rhs[:nF, 1:] = MF.T
        sol, _, rank, _ = np.linalg.lstsq(K, rhs, rcond=None)
        self.F = F
        self.z0 = sol[:nF, 0]
        self.mu0 = float(sol[nF, 0])
        self.singular = bool(rank < nF + 1)
        if coupled:
            self.Z = sol[:nF, 1:]
            self.muZ = sol[nF, 1:]
            self.p = MF @ self.z0
            self.P = MF @ self.Z
        else:
            self.Z = self.muZ = self.p = self.P = None


def _active_set(
    Ms: List[np.ndarray],
    q: np.ndarray,
    c: np.ndarray,
    b: np.ndarray,
    c_pool: float,
    lam: float,
) -> np.ndarray:
    """Primal active-set method for ``min .5 z'Hz + c'z, Az = b, z >= 0``.

    Lawson-Hanson style: keep a feasible iterate, solve the equality
    constrained problem on the free set exactly (KKT system), step back to
    the boundary when that solution leaves the orthant, and free the
    coordinate with the most negative reduced cost once it does not.
    """
    J = len(Ms)
    sizes = np.array([M.shape[1] for M in Ms])
    cuts = np.concatenate([[0], np.cumsum(sizes)])
    N = int(cuts[-1])
    Mcat = np.hstack(Ms)
    m = Mcat.shape[0]
    blk = np.repeat(np.arange(J), sizes)
    coupled = c_pool > 0
    diag_h = (q[blk] + c_pool) * np.sum(Mcat**2, axis=0) + lam

    z = np.zeros(N)
    free = np.zeros(N, dtype=bool)
    for j in range(J):
        sl = slice(cuts[j], cuts[j + 1])
        # start each block at its best single donor (vertex of the simplex)
        vals = 0.5 * b[j] ** 2 * diag_h[sl] + b[j] * c[sl]
        k = cuts[j] + int(np.argmin(vals))
        z[k] = b[j]
        free[k] = True
    # H is positive semi-definite, so its largest entry is on the diagonal
    scale_c = max(1.0, float(np.abs(c).max()), float(diag_h.max()))
    tol_zero = 1e-14 * float(b.max())

    MT = np.ascontiguousarray(Mcat.T)  # (N, m): row k is donor column k
    # Per-block KKT solutions scattered into full-length arrays so that the
    # face solution is assembled with vectorised gathers.
    z0_all = np.zeros(N)
    mu0_all = np.zeros(J)
    singular = np.zeros(J, dtype=bool)
    if coupled:
        Z_all = np.zeros((N, m))
        muZ_all = np.zeros((J, m))
        P_all = np.zeros((J, m, m))
        p_all = np.zeros((J, m))
    dirty = set(range(J))

    for _ in range(20 * N + 100):
        for j in dirty:
            F_j = cuts[j] + np.flatnonzero(free[cuts[j] : cuts[j + 1]])
            blk_j = _Block(F_j, Mcat, q[j], c, b[j], lam, coupled)
            z0_all[F_j] = blk_j.z0
            mu0_all[j] = blk_j.mu0
            singular[j] = blk_j.singular
            if coupled:
                Z_all[F_j] = blk_j.Z
                muZ_all[j] = blk_j.muZ
                P_all[j] = blk_j.P
                p_all[j] = blk_j.p
        dirty = set()
        F = np.flatnonzero(free)
        starts = np.searchsorted(blk[F], np.arange(J))

        if coupled:
            zF = mu = None
            if not singular.any():
                S = P_all.sum(axis=0)
                u0 = p_all.sum(axis=0)
                lam_pool = np.linalg.solve(np.eye(m) + c_pool * S, -c_pool * u0)
                zF = z0_all[F] + Z_all[F] @ lam_pool
                mu = mu0_all + muZ_all @ lam_pool
                # The Schur route is exact in exact arithmetic but loses
                # accuracy when a block's own curvature is negligible next to
                # the pooled one (nu -> 1); certify the face solution.
                stat = _reduced_cost(
                    F, zF, starts, True, MT, blk, q, c, mu, c_pool, lam
                )
                if np.abs(stat).max() > 1e-10 * scale_c * max(1.0, np.abs(zF).max()):
                    zF = None
            if zF is None:
                zF, mu = _dense_kkt(F, Mcat, blk, q, c, b, c_pool, lam, J)
        else:
            zF = z0_all[F]
            mu = mu0_all.copy()

        if np.any(zF < 0.0):
            cur = z[F]
            neg = np.flatnonzero(zF < 0.0)
            ratios = cur[neg] / (cur[neg] - zF[neg])
            step = float(np.min(ratios))
            z[F] = cur + step * (zF - cur)
            blocking = F[neg[int(np.argmin(ratios))]]
            z[blocking] = 0.0
            hit = F[z[F] <= tol_zero]
            z[hit] = 0.0
            free[hit] = False
            free[blocking] = False
            dirty.update(blk[hit].tolist())
            dirty.add(int(blk[blocking]))
            # keep every block non-empty
            for j in sorted(dirty):
                blk_sl = slice(cuts[j], cuts[j + 1])
                if not free[blk_sl].any():
                    free[cuts[j] + int(np.argmax(z[blk_sl]))] = True
            continue
        z = np.zeros(N)
        z[F] = zF
        s_red = _reduced_cost(F, zF, starts, False, MT, blk, q, c, mu, c_pool, lam)
        s_red[F] = 0.0
        k = int(np.argmin(s_red))
        if s_red[k] >= -1e-13 * scale_c:
            return z
        free[k] = True
        dirty.add(int(blk[k]))
    raise ConvergenceFailure(
        "multisynth QP active-set solver did not converge",
        recovery_hint=(
            "Add a small ridge penalty (penalization > 0) or reduce n_lags; "
            "a flat objective makes the weight problem degenerate."
        ),
    )


def _reduced_cost(
    F: np.ndarray,
    zF: np.ndarray,
    starts: np.ndarray,
    only_free: bool,
    MT: np.ndarray,
    blk: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    c_pool: float,
    lam: float,
) -> np.ndarray:
    """``Hz + c + A'mu`` for ``z`` supported on ``F``, without forming ``H``.

    Evaluated on every coordinate, or on ``F`` only (``only_free``).
    ``starts`` are the positions in ``F`` where each block begins (every
    block has at least one free coordinate).
    """
    r = np.add.reduceat(MT[F] * zF[:, None], starts, axis=0)  # (J, m): M_j z_j
    if only_free:
        bi, Mi, ci = blk[F], MT[F], c[F]
    else:
        bi, Mi, ci = blk, MT, c
    out = q[bi] * np.einsum("km,km->k", Mi, r[bi]) + ci + mu[bi]
    if c_pool > 0:
        out += c_pool * (Mi @ r.sum(axis=0))
    if lam:
        if only_free:
            out += lam * zF
        else:
            out[F] += lam * zF
    return out


def _dense_kkt(
    F: np.ndarray,
    Mcat: np.ndarray,
    blk: np.ndarray,
    q: np.ndarray,
    c: np.ndarray,
    b: np.ndarray,
    c_pool: float,
    lam: float,
    J: int,
):
    """Free-set KKT solve with the coupled Hessian formed explicitly.

    Used only when some block's own KKT matrix is rank deficient (for
    example ``nu = 1`` without a ridge penalty, where the separate part of
    the Hessian vanishes); the minimum-norm least-squares solution then
    matches the former dense solver.
    """
    nF = F.size
    MF = Mcat[:, F]
    bF = blk[F]
    same = bF[:, None] == bF[None, :]
    H_FF = c_pool * (MF.T @ MF) + np.where(same, q[bF][:, None] * (MF.T @ MF), 0.0)
    if lam:
        H_FF[np.arange(nF), np.arange(nF)] += lam
    A_F = (bF[None, :] == np.arange(J)[:, None]).astype(float)
    K = np.zeros((nF + J, nF + J))
    K[:nF, :nF] = H_FF
    K[:nF, nF:] = A_F.T
    K[nF:, :nF] = A_F
    sol = np.linalg.lstsq(K, np.concatenate([-c[F], b]), rcond=None)[0]
    return sol[:nF], sol[nF:]
