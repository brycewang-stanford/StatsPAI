"""Exact fixed-effect residualisation by a sparse direct solve.

The fallback when alternating projections stop before converging. Their
rate is set by how well the fixed-effect graph is connected: on a chain of
firms linked by a few movers each, or on panels where most units sit in
groups of one, a thousand (or ten thousand) sweeps can leave the
residualised columns visibly off and the slope with them, while the same
sparse structure makes the normal equations cheap to factorise. So the
two methods cover each other: well-connected graphs converge in a few
sweeps and would fill in under a factorisation; poorly connected ones
factorise with little fill.

The projection is ``r = x - D a`` with ``a`` solving the weighted normal
equations ``D' W D a = D' W x``. ``D' W D`` is singular (one dimension per
connected component and absorbed dimension after the first), so it is
factorised once with a diagonal shift of ``1e-13`` after symmetric
scaling to a unit diagonal, and the solve is refined until the weighted
group means of the residual vanish. The shift touches only the null space
(``D' W x`` is orthogonal to it), and each refinement step multiplies the
remaining error in the other directions by at most
``shift / (shift + eigenvalue)``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

_SHIFT = 1e-13
_MAX_REFINE = 50
#: Schur complements up to this order are factorised densely.
_DENSE_MAX = 6000


def fe_design(codes: List[np.ndarray], n: int) -> Any:
    """Indicator design of dense codes (``n`` rows, one block per FE)."""
    from scipy import sparse

    k = len(codes)
    offsets = np.cumsum([0] + [int(c.max()) + 1 if c.size else 0 for c in codes])
    if k == 0:
        return sparse.csr_matrix((n, 0))
    indices = np.empty(n * k, dtype=np.int64)
    for j, c in enumerate(codes):
        indices[j::k] = np.asarray(c, dtype=np.int64) + offsets[j]
    indptr = np.arange(0, n * k + 1, k, dtype=np.int64)
    return sparse.csr_matrix(
        (np.ones(n * k), indices, indptr), shape=(n, int(offsets[-1]))
    )


class _Solver:
    """``a = M^{-1} g`` for ``M = D' W D`` (pseudo-inverse on its range).

    With ``n_diag > 0`` the first ``n_diag`` columns of ``D`` are one
    fixed-effect dimension, whose block of ``M`` is diagonal; it is
    eliminated and the Schur complement on the remaining levels is
    factorised (dense when it is small, sparse otherwise). ``M`` and the
    Schur complement are singular; both are scaled to a unit diagonal and
    shifted by ``_SHIFT`` before factorising, and the caller refines.
    """

    def __init__(self, M: Any, n_diag: int) -> None:
        from scipy import sparse

        M = sparse.csc_matrix(M)
        m = M.shape[0]
        self.n1 = n1 = n_diag if 0 < n_diag < m else 0
        if n1:
            A = M[:n1, :n1].diagonal()
            self.a_inv = 1.0 / np.where(A > 0, A, 1.0)
            B = M[:n1, n1:].tocsc()
            C = M[n1:, n1:]
            self.B = B
            Sc = (C - B.T @ sparse.diags(self.a_inv) @ B).tocsc()
        else:
            Sc = M
        d = Sc.diagonal()
        pos = d > 0
        self.s = np.where(pos, 1.0 / np.sqrt(np.where(pos, d, 1.0)), 1.0)
        S = sparse.diags(self.s)
        k = Sc.shape[0]
        Ss = (S @ Sc @ S).tocsc() + sparse.identity(k, format="csc") * _SHIFT
        if k <= _DENSE_MAX:
            from scipy.linalg import lu_factor, lu_solve

            fac = lu_factor(Ss.toarray(), check_finite=False)
            self._solve = lambda b: lu_solve(fac, b, check_finite=False)
        else:
            from scipy.sparse.linalg import splu

            lu = splu(Ss, permc_spec="MMD_AT_PLUS_A")
            self._solve = lu.solve

    def __call__(self, g: np.ndarray) -> np.ndarray:
        n1 = self.n1
        if not n1:
            return self.s * self._solve(self.s * g)
        g1, g2 = g[:n1], g[n1:]
        a2 = self.s * self._solve(self.s * (g2 - self.B.T @ (self.a_inv * g1)))
        a1 = self.a_inv * (g1 - self.B @ a2)
        return np.concatenate([a1, a2])


def residualise(
    X: np.ndarray,
    D: Any,
    weights: Optional[np.ndarray] = None,
    *,
    columns: Optional[List[int]] = None,
    n_diag: int = 0,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, bool]:
    """Exact weighted residuals of the columns of ``X`` on ``span(D)``.

    ``X`` is modified in place (only ``columns``, all by default) and
    returned with a flag: ``True`` when every column met the stopping rule
    (largest weighted group mean of the residual at most ``tol`` times the
    column's scale). Starting from partially residualised columns is fine:
    the residual of a vector already shifted by ``D a0`` is the same.
    ``n_diag``: the first ``n_diag`` columns of ``D`` are one plain
    fixed effect (eliminated first; pass the largest dimension there).
    """
    from scipy import sparse

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    Dc = sparse.csc_matrix(D)
    DtW = (Dc.multiply(w[:, None])).T.tocsr()
    M = DtW @ Dc
    diag = M.diagonal()
    level_w = np.where(diag > 0, diag, 1.0)
    solve = _Solver(M, n_diag)

    ok = True
    cols = range(X.shape[1]) if columns is None else columns
    for j in cols:
        r = np.array(X[:, j], dtype=np.float64)
        scale = max(1.0, float(np.max(np.abs(r))) if r.size else 1.0)
        done = False
        prev = np.inf
        for _ in range(_MAX_REFINE):
            g = DtW @ r
            worst = float(np.max(np.abs(g / level_w))) if g.size else 0.0
            if worst <= tol * scale:
                done = True
                break
            if worst >= prev:  # no progress: rounding floor reached
                done = worst <= 1e-9 * scale
                break
            prev = worst
            r -= Dc @ solve(g)
        X[:, j] = r
        ok = ok and done
    return X, ok


def residualise_codes(
    X: np.ndarray,
    codes: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
    *,
    columns: Optional[List[int]] = None,
) -> Tuple[np.ndarray, bool]:
    """:func:`residualise` on dense fixed-effect codes, eliminating the
    dimension with the most levels first."""
    order = sorted(
        range(len(codes)),
        key=lambda j: -(int(codes[j].max()) + 1 if codes[j].size else 0),
    )
    ordered = [codes[j] for j in order]
    n1 = int(ordered[0].max()) + 1 if ordered and ordered[0].size else 0
    return residualise(
        X,
        fe_design(ordered, X.shape[0]),
        weights,
        columns=columns,
        n_diag=n1 if len(codes) > 1 else 0,
    )


__all__ = ["fe_design", "residualise", "residualise_codes"]
