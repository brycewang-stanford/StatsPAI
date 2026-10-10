"""
Shared primitives for linear and generalized linear mixed models.

Covers:
    * Formula / column parsing and per-group matrix pre-computation.
    * Parameterisation of random-effect covariance matrices
      (identity / diagonal / unstructured) via a packed theta vector.
    * Marginal covariance V_j = Z_j G Z_j' + Phi_j and its log-determinant
      through a Cholesky factorisation.
    * Pseudo-data stacking utilities used by both the LMM GLS step and
      the Laplace-approximation inner loop of the GLMM.

Kept deliberately free of plotting / result-object code so it can be
reused from ``lmm.py`` and ``glmm.py`` without creating cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Grouping utilities
# ---------------------------------------------------------------------------


def _as_str_list(x: Optional[str | Sequence[str]]) -> List[str]:
    """Normalise a column spec to a list of strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _build_group_keys(df: pd.DataFrame, groups: Sequence[str]) -> np.ndarray:
    """
    Return an (n,) array of hashable keys identifying the finest
    group crossing for *nested* multi-level designs.

    ``groups`` are ordered outermost → innermost, e.g.
    ``["school", "class"]`` means classes are nested within schools and
    the returned key identifies a single class uniquely.
    """
    if len(groups) == 0:
        raise ValueError("at least one grouping variable is required")
    if len(groups) == 1:
        return np.asarray(df[groups[0]].values)
    # Tuple-based key preserves hierarchical identity across levels.
    return np.asarray(pd.MultiIndex.from_frame(df[list(groups)]).values)


# ---------------------------------------------------------------------------
# Covariance parameterisation
# ---------------------------------------------------------------------------


def _n_cov_params(q: int, cov_type: str) -> int:
    """Number of free parameters for the chosen covariance structure."""
    if cov_type == "identity":
        return 1
    if cov_type == "diagonal":
        return q
    if cov_type == "unstructured":
        return q * (q + 1) // 2
    raise ValueError(f"unknown cov_type {cov_type!r}")


def _unpack_G(theta: np.ndarray, q: int, cov_type: str) -> np.ndarray:
    """
    Rebuild a q×q PSD covariance matrix from its packed parameters.

    The parameterisation keeps all structures **unconstrained**, so
    numerical optimisation does not need box constraints:

    * identity     — single log-variance σ²   applied to I_q.
    * diagonal     — q log-variances.
    * unstructured — lower-triangular Cholesky L with log-diagonal,
                     G = L L'.
    """
    if cov_type == "identity":
        s2 = float(np.exp(theta[0]))
        return np.asarray(s2 * np.eye(q), dtype=float)

    if cov_type == "diagonal":
        return np.asarray(np.diag(np.exp(theta[:q])), dtype=float)

    if cov_type == "unstructured":
        L = np.zeros((q, q))
        idx = 0
        for i in range(q):
            for j in range(i + 1):
                if i == j:
                    L[i, j] = np.exp(theta[idx])
                else:
                    L[i, j] = theta[idx]
                idx += 1
        return np.asarray(L @ L.T, dtype=float)

    raise ValueError(f"unknown cov_type {cov_type!r}")


def _initial_theta(q: int, cov_type: str, s2_init: float) -> np.ndarray:
    """Sensible starting values for the covariance parameters."""
    # Aim for a small random-effect variance (~10% of residual var) at
    # the start; this is conservative and easy to escape from.
    log_s2 = np.log(max(s2_init, 1e-6))

    if cov_type == "identity":
        return np.array([log_s2], dtype=float)

    if cov_type == "diagonal":
        return np.full(q, log_s2, dtype=float)

    if cov_type == "unstructured":
        theta = np.zeros(_n_cov_params(q, "unstructured"))
        idx = 0
        for i in range(q):
            for j in range(i + 1):
                if i == j:
                    theta[idx] = 0.5 * log_s2  # diag of L
                idx += 1
        return theta

    raise ValueError(f"unknown cov_type {cov_type!r}")


# ---------------------------------------------------------------------------
# Per-group pre-processed block
# ---------------------------------------------------------------------------


@dataclass
class _GroupBlock:
    """
    Pre-computed matrices for a single level-2 group.

    These are expensive to construct — touching pandas for each group on
    every likelihood evaluation would be prohibitive.  We do it once and
    hand the raw numpy arrays to the hot loop.

    ``row_idx`` is the positional index of each observation in the
    **original** training dataframe (post-``dropna``).  It lets
    ``predict(data=None)`` return predictions aligned with that
    dataframe rather than the group-iteration order.
    """

    key: object
    y: np.ndarray  # (n_j,)
    X: np.ndarray  # (n_j, p)
    Z: np.ndarray  # (n_j, q)  (intercept-first ordering)
    n: int
    row_idx: Optional[np.ndarray] = None  # positions into original df

    def V(self, G: np.ndarray, sigma2: float) -> np.ndarray:
        """Marginal covariance V_j = Z_j G Z_j' + sigma² I."""
        return np.asarray(self.Z @ G @ self.Z.T + sigma2 * np.eye(self.n))


def _group_blocks(
    df: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    x_random: Sequence[str],
    group_col_name: str,
) -> Tuple[List[_GroupBlock], List[str], List[str]]:
    """
    Split a dataframe into per-group numeric blocks.

    Returns the list of blocks plus the column-name vectors for fixed
    and random effects (with the intercept name prepended).  An explicit
    column named ``__intercept__`` is expected to exist in ``df``.
    """
    fixed_names = ["_cons"] + list(x_fixed)
    random_names = ["_cons"] + list(x_random)

    y_all = df[y].to_numpy(dtype=float)
    X_all = df[["__intercept__"] + list(x_fixed)].to_numpy(dtype=float)
    Z_all = df[["__intercept__"] + list(x_random)].to_numpy(dtype=float)

    grouped = df.groupby(group_col_name, sort=False)
    if isinstance(df[group_col_name].dtype, pd.CategoricalDtype):
        # A categorical key can yield groups with no rows (unobserved
        # categories); let pandas decide which, and in what order.
        positions = np.arange(len(df))
        parts = [
            (key, positions[df.index.get_indexer(sub.index)]) for key, sub in grouped
        ]
        keys: List[object] = [key for key, _ in parts]
        row_sets = [idx for _, idx in parts]
    else:
        # One stable sort instead of one pandas slice per group: groups in
        # order of first appearance, rows in their original order.
        codes = grouped.ngroup().to_numpy()
        order = np.argsort(codes, kind="stable")
        sizes = np.bincount(codes, minlength=int(grouped.ngroups))
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        keys = df[group_col_name].iloc[order[bounds[:-1]]].tolist()
        row_sets = [order[bounds[j] : bounds[j + 1]] for j in range(len(sizes))]

    # Column-major blocks, as a per-group ``DataFrame.to_numpy()`` returns
    # them: BLAS rounds a product differently for the two layouts, and the
    # GLMM optimisers downstream are compared with Stata at 1e-6.
    blocks: List[_GroupBlock] = [
        _GroupBlock(
            key=key,
            y=y_all[idx],
            X=np.asfortranarray(X_all[idx]),
            Z=np.asfortranarray(Z_all[idx]),
            n=len(idx),
            row_idx=idx,
        )
        for key, idx in zip(keys, row_sets)
    ]
    return blocks, fixed_names, random_names


# ---------------------------------------------------------------------------
# Marginal-covariance Cholesky helper
# ---------------------------------------------------------------------------


def _solve_V(V: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve V x = B via Cholesky and return (x, log|V|).

    Raises ``np.linalg.LinAlgError`` if V is not positive-definite.
    """
    L = np.linalg.cholesky(V)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    # solve L z = B, then L' x = z
    z = np.linalg.solve(L, B)
    x = np.linalg.solve(L.T, z)
    return x, logdet


# ---------------------------------------------------------------------------
# Reduced (cross-product) form of the Gaussian likelihood
# ---------------------------------------------------------------------------


@dataclass
class _LMMReduced:
    """
    Everything the profiled (RE)ML criterion needs from the data.

    With the thin QR ``Z_j = Q_j R_j``, rotating group *j* by an orthogonal
    matrix whose first columns are ``Q_j`` turns ``V_j`` into
    ``blockdiag(R_j G R_j' + sigma2 I_q, sigma2 I)``.  The likelihood then
    depends on the data only through ``R_j``, ``C_j = Q_j' [X_j, e_j]`` and
    the pooled cross-product of the part of ``[X, e]`` orthogonal to every
    ``Z_j``.  Nothing here requires ``G`` to be invertible, and no term is
    formed as a difference of large numbers.

    ``e = y - X beta0`` (pooled OLS) stands in for ``y``: the criterion is
    unchanged, and the GLS step then solves for a small correction, so the
    residual quadratic form carries no cancellation from the level of
    ``X beta``.  Groups with fewer than ``q`` rows are zero-padded; the
    padding contributes ``log sigma2`` per row to ``log|W_j|``, which the
    ``(n_j - q) log sigma2`` term takes back.
    """

    R: np.ndarray  # (J, q, q)
    C: np.ndarray  # (J, q, p + 1)
    U: np.ndarray  # (k, p + 1) triangular factor of the orthogonal part
    beta0: np.ndarray  # (p,)
    n_total: int
    p: int
    q: int

    @property
    def n_groups(self) -> int:
        return int(self.R.shape[0])


def _lmm_reduce(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    sizes: np.ndarray,
    beta0: Optional[np.ndarray] = None,
) -> _LMMReduced:
    """
    Build :class:`_LMMReduced` from rows stored group after group.

    ``sizes[j]`` is the number of consecutive rows belonging to group *j*.
    Groups of equal size are factorised together, so the work is a handful
    of batched QR calls rather than one per group.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    sizes = np.asarray(sizes, dtype=np.intp)
    n, p = X.shape
    q = Z.shape[1]
    J = len(sizes)
    if beta0 is None:
        beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    D = np.column_stack([X, y - X @ beta0])

    R = np.zeros((J, q, q))
    C = np.zeros((J, q, p + 1))
    D_perp = np.zeros((n, p + 1))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.intp)
    for m in np.unique(sizes):
        if m == 0:
            continue
        idx = np.flatnonzero(sizes == m)
        rows = starts[idx][:, None] + np.arange(m)[None, :]
        Q, R_m = np.linalg.qr(Z[rows])
        D_m = D[rows]
        C_m = np.swapaxes(Q, 1, 2) @ D_m
        k = R_m.shape[1]  # min(m, q)
        R[idx, :k, :] = R_m
        C[idx, :k, :] = C_m
        D_perp[rows] = D_m - Q @ C_m
    # Only D_perp' D_perp matters; keep its triangular factor so the
    # residual sum of squares is formed as a norm, not as a difference.
    U = np.asarray(np.linalg.qr(D_perp, mode="r"))
    return _LMMReduced(R=R, C=C, U=U, beta0=np.asarray(beta0), n_total=int(n), p=p, q=q)


def _lmm_reduce_blocks(
    blocks: Sequence[_GroupBlock], beta0: Optional[np.ndarray] = None
) -> _LMMReduced:
    """:func:`_lmm_reduce` for a list of per-group blocks."""
    return _lmm_reduce(
        np.concatenate([b.y for b in blocks]),
        np.vstack([b.X for b in blocks]),
        np.vstack([b.Z for b in blocks]),
        np.array([b.n for b in blocks], dtype=np.intp),
        beta0,
    )


@dataclass
class _LMMSolve:
    """GLS quantities at one value of (G, sigma2); see :func:`_lmm_solve`."""

    L: np.ndarray  # (J, q, q) Cholesky factors of W_j = R_j G R_j' + sigma2 I
    T: np.ndarray  # (J, q, p + 1)  L_j^{-1} C_j
    XtVinvX: np.ndarray  # (p, p)
    delta: np.ndarray  # (p,)  beta_hat - beta0
    logdet_V: float  # sum_j log|V_j|
    quad: float  # sum_j r_j' V_j^{-1} r_j at beta_hat


def _lmm_solve(red: _LMMReduced, G: np.ndarray, sigma2: float) -> _LMMSolve:
    """
    Profile the fixed effects out at (G, sigma2).

    Raises ``np.linalg.LinAlgError`` when some ``W_j`` is not positive
    definite or the GLS normal equations are singular.
    """
    p = red.p
    W = (red.R @ G) @ np.swapaxes(red.R, 1, 2)
    diag = np.arange(red.q)
    W[:, diag, diag] += sigma2
    L = np.linalg.cholesky(W)
    T = np.linalg.solve(L, red.C)
    Tm = T.reshape(-1, p + 1)
    Us = red.U / np.sqrt(sigma2)
    F = Tm.T @ Tm + Us.T @ Us
    XtVinvX = F[:p, :p]
    delta = np.linalg.solve(XtVinvX, F[:p, p])
    r_par = Tm[:, p] - Tm[:, :p] @ delta
    r_perp = Us[:, p] - Us[:, :p] @ delta
    quad = float(r_par @ r_par + r_perp @ r_perp)
    logdet_V = float(
        (red.n_total - red.n_groups * red.q) * np.log(sigma2)
        + 2.0 * np.sum(np.log(L[:, diag, diag]))
    )
    return _LMMSolve(
        L=L, T=T, XtVinvX=XtVinvX, delta=delta, logdet_V=logdet_V, quad=quad
    )


# ---------------------------------------------------------------------------
# Formula helpers (intercept handling)
# ---------------------------------------------------------------------------


def _prepare_frame(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group_cols: Sequence[str],
    x_random: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate columns and return a cleaned copy with ``__intercept__``
    already attached.
    """
    all_cols = [y] + list(x_fixed) + list(group_cols)
    if x_random:
        all_cols.extend(x_random)
    if weights:
        all_cols.append(weights)
    missing = [c for c in all_cols if c not in data.columns]
    if missing:
        raise KeyError(f"missing columns in data: {missing}")

    # Reject non-hashable group values early — a list/array in the
    # group column silently poisons the ``blups`` dict at fit time.
    for g in group_cols:
        sample = data[g].iloc[0] if len(data) else None
        if sample is not None:
            try:
                hash(sample)
            except TypeError as exc:
                raise TypeError(
                    f"group column {g!r} contains unhashable values "
                    f"(e.g. {type(sample).__name__}); cast it to a "
                    "hashable type (str, int, tuple) before calling mixed()."
                ) from exc
    # Dedup while preserving order — random-slope variables frequently
    # also appear in ``x_fixed``, and pandas creates duplicated columns
    # when asked to index a column twice.
    seen: set = set()
    unique_cols: List[str] = []
    for c in all_cols:
        if c not in seen:
            unique_cols.append(c)
            seen.add(c)
    df = data[unique_cols].dropna().copy()
    df["__intercept__"] = 1.0
    return df


__all__ = [
    "_as_str_list",
    "_build_group_keys",
    "_n_cov_params",
    "_unpack_G",
    "_initial_theta",
    "_GroupBlock",
    "_group_blocks",
    "_solve_V",
    "_LMMReduced",
    "_LMMSolve",
    "_lmm_reduce",
    "_lmm_reduce_blocks",
    "_lmm_solve",
    "_prepare_frame",
]
