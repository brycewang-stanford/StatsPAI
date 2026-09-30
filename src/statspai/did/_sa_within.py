"""Two-way within fit of the Sun-Abraham interacted regression.

The saturated design has one ``1(G = g) 1(e = l)`` column per estimated
cohort x event-time cell: with a hundred cohorts that is a thousand columns,
each row nonzero in at most one of them. Materialising those columns and
demeaning them one at a time took ~35 s and several GB on a 180,000-row
panel (Minimum Wages, QJE 2019, ~100 cohorts).

Here the within transformation is never applied to the design. With ``D``
the unit and period dummies (one period dropped for rank), ``W`` the weights,
``R`` the sparse one-hot cell design and ``M = I - D (D'WD)^{-1} D'W`` the
weighted within projection (``W``-self-adjoint, idempotent):

* ``X~'W X~ = R'W R - F'(D'WD)^{-1}F`` with ``F = D'W R``;
* ``(D'WD)^{-1}`` is applied by eliminating the diagonal unit block, which
  leaves a ``(T-1) x (T-1)`` Schur complement ``S``;
* the cluster scores ``C diag(w u) M R`` reduce to the same sparse products.

Everything is exact -- no alternating-projection tolerance -- and costs a few
sparse products plus ``O(G T k)``. Outcome and covariates (dense, few columns)
are demeaned exactly by the same elimination.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse


class TwoWayFE:
    """Exact weighted projection onto unit and period effects."""

    def __init__(
        self,
        u_codes: np.ndarray,
        t_codes: np.ndarray,
        n_u: int,
        n_t: int,
        w: np.ndarray,
    ) -> None:
        n = len(u_codes)
        rows = np.arange(n)
        self.n, self.n_u, self.n_t = n, n_u, n_t
        self.u_codes, self.t_codes, self.w = u_codes, t_codes, w
        # Indicator matrices (n x U, n x T) and weighted transposes (U x n, T x n).
        self.Iu = sparse.csr_matrix((np.ones(n), (rows, u_codes)), shape=(n, n_u))
        self.It = sparse.csr_matrix((np.ones(n), (rows, t_codes)), shape=(n, n_t))
        self.DuW = sparse.csr_matrix((w, (u_codes, rows)), shape=(n_u, n))
        self.DtW = sparse.csr_matrix((w, (t_codes, rows)), shape=(n_t, n))
        self.A = np.asarray(self.DuW.sum(axis=1)).ravel()  # unit weight mass
        B = (self.DuW @ self.It).tocsc()  # U x T weighted cell mass
        self.B = B[:, 1:].tocsr()  # drop period 0 (rank)
        C = np.asarray(self.DtW.sum(axis=1)).ravel()[1:]
        BtAinvB = (self.B.T @ sparse.diags(1.0 / self.A) @ self.B).toarray()
        S = np.diag(C) - BtAinvB
        # Pseudo-inverse: exact on a connected design, and still the right
        # projection when the unit-period graph splits into components.
        self.S_pinv = np.linalg.pinv(S, hermitian=True) if S.size else S

    def coefficients(
        self, Fu: np.ndarray, Ft: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unit and period effects (period 0 = 0) for ``D'W v = [Fu; Ft]``."""
        Fu = np.asarray(Fu, dtype=float)
        Ft = np.asarray(Ft, dtype=float)
        AinvFu = Fu / self.A[:, None]
        H = Ft[1:] - self.B.T @ AinvFu
        b = self.S_pinv @ H
        a = AinvFu - (self.B @ b) / self.A[:, None]
        b_full = np.vstack([np.zeros((1, b.shape[1])), b])
        return a, b_full

    def demean(self, V: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        squeeze = V.ndim == 1
        V2 = V[:, None] if squeeze else V
        a, b = self.coefficients(self.DuW @ V2, self.DtW @ V2)
        out = V2 - a[self.u_codes] - b[self.t_codes]
        return out[:, 0] if squeeze else out


def within_fit(
    y: np.ndarray,
    cell: np.ndarray,
    k_int: int,
    covariates: Optional[np.ndarray],
    u_codes: np.ndarray,
    t_codes: np.ndarray,
    n_u: int,
    n_t: int,
    w: np.ndarray,
    cluster_codes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Coefficients, bread and cluster meat of the within regression.

    ``cell[i]`` is the interaction column of row ``i`` (``-1`` when the row
    is in no estimated cell); ``covariates`` are extra dense regressors.
    Returns ``beta``, ``XtX``, ``XtX_inv`` (with the ``1e-10`` ridge the
    dense path used) and ``meat`` (``sum_c s_c s_c'``).
    """
    n = len(y)
    fe = TwoWayFE(u_codes, t_codes, n_u, n_t, w)
    in_cell = cell >= 0
    R = sparse.csr_matrix(
        (np.ones(int(in_cell.sum())), (np.flatnonzero(in_cell), cell[in_cell])),
        shape=(n, k_int),
    )
    Z = None if covariates is None or covariates.shape[1] == 0 else covariates
    k_cov = 0 if Z is None else Z.shape[1]
    k = k_int + k_cov

    # F = D'W R, and the quadratic form F'(D'WD)^{-1}F by elimination.
    Fu = (fe.DuW @ R).toarray()  # U x k_int
    Ft = (fe.DtW @ R).toarray()  # T x k_int
    AinvFu = Fu / fe.A[:, None]
    H = Ft[1:] - fe.B.T @ AinvFu  # (T-1) x k_int
    P = fe.S_pinv @ H
    RtWR = np.asarray(R.multiply(w[:, None]).sum(axis=0)).ravel()
    XtX = np.zeros((k, k))
    XtX[:k_int, :k_int] = np.diag(RtWR) - (Fu.T @ AinvFu + H.T @ P)

    y_dm = fe.demean(y)
    Xty = np.zeros(k)
    Xty[:k_int] = R.T @ (w * y_dm)
    Z_dm = None
    if Z is not None:
        Z_dm = fe.demean(Z)
        wZ = Z_dm * w[:, None]
        XtX[k_int:, k_int:] = wZ.T @ Z_dm
        XtX[k_int:, :k_int] = (R.T @ wZ).T
        XtX[:k_int, k_int:] = XtX[k_int:, :k_int].T
        Xty[k_int:] = wZ.T @ y_dm
    XtX = (XtX + XtX.T) / 2.0

    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(k))
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ Xty

    fitted_raw = R @ beta[:k_int]
    if Z is not None:
        fitted_raw = fitted_raw + Z @ beta[k_int:]
    u = y_dm - fe.demean(fitted_raw)

    # Cluster scores  C diag(w u) M R  =  C_wu R - C_wu D (D'WD)^{-1} F,
    # with (D'WD)^{-1} F = [A^{-1} Fu - A^{-1} B P ; P].
    G = int(cluster_codes.max()) + 1
    Cwu = sparse.csr_matrix((w * u, (cluster_codes, np.arange(n))), shape=(G, n))
    CwuIu = (Cwu @ fe.Iu).tocsr()  # G x U
    CwuIt = (Cwu @ fe.It).toarray()  # G x T
    top = CwuIu @ AinvFu - (CwuIu @ sparse.diags(1.0 / fe.A) @ fe.B) @ P
    S_cl = np.zeros((G, k))
    S_cl[:, :k_int] = (Cwu @ R).toarray() - (top + CwuIt[:, 1:] @ P)
    if Z_dm is not None:
        S_cl[:, k_int:] = Cwu @ Z_dm
    return dict(
        beta=beta, XtX=XtX, XtX_inv=XtX_inv, meat=S_cl.T @ S_cl, resid=u, n=n, k=k
    )
