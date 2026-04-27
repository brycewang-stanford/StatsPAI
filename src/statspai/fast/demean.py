"""Fast multi-way HDFE demean kernel.

``sp.fast.demean`` exposes the Phase 1 Rust alternating-projection
within-transformation. Compared to the user-facing ``sp.demean`` /
``sp.Absorber``, this is a thinner, lower-level API closer to the FFI
boundary; it is the building block that Phase 2+ ``sp.fast.fepois`` and
``sp.fast.feols`` will sit on top of.

Design
------
1. Inputs (X, FE codes) are factorised on the Python side via
   ``pd.factorize`` so the Rust path receives dense, contiguous int64
   codes — the FFI shape it expects.
2. We *prefer* the Rust kernel when the compiled extension is loadable;
   otherwise we fall through to a NumPy implementation that mirrors the
   same algorithm bit-for-bit (Aitken layout, double-threshold tol),
   giving identical numerical results to within float-rounding.
3. Singleton pruning runs upstream of the AP loop. It is iterative
   (Correia 2015) and runs until no codes appear with count 1.
4. The block (multi-column) Rust path uses Rayon to parallelise across
   columns; per-column it is single-threaded so within-column data
   dependencies aren't violated.

References
----------
Correia, S. (2015). Singletons, Cluster-Robust Standard Errors and Fixed
Effects. Working paper.
Bergé, L. (2018). Efficient estimation of maximum likelihood models with
multiple fixed-effects: the R package FENmlm. CREA DP 13.
Varadhan, R. and Roland, C. (2008). Simple and globally convergent
methods for accelerating the convergence of any EM algorithm.
Scandinavian Journal of Statistics 35(2), 335–353.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Optional Rust backend. Failure to import is silent: the NumPy fallback
# always works, just slower on large panels.
try:
    import statspai_hdfe as _rust  # type: ignore
    _HAS_RUST = True
    _RUST_VERSION: Optional[str] = getattr(_rust, "__version__", None)
except ImportError:  # pragma: no cover  - exercised in CI on no-Rust wheels
    _rust = None  # type: ignore
    _HAS_RUST = False
    _RUST_VERSION = None


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class DemeanInfo:
    """Diagnostics returned from :func:`demean`.

    Attributes
    ----------
    n : int
        Original row count (before any singleton drop).
    n_kept : int
        Rows retained after singleton pruning.
    n_dropped : int
        ``n - n_kept``.
    iters : list of int
        AP iterations per output column.
    converged : list of bool
        Whether each column hit the tolerance threshold within max_iter.
    max_dx : list of float
        Final ``max|dx|`` per column (post-stop).
    keep_mask : ndarray of bool, shape (n,)
        True for rows that survived singleton pruning (input alignment).
    backend : str
        ``"rust"`` or ``"numpy"``.
    accel : str
        ``"aitken"`` or ``"none"``.
    n_fe : list of int
        Surviving group cardinality per FE dimension (post-prune,
        post-densification).
    """

    n: int
    n_kept: int
    n_dropped: int
    iters: List[int]
    converged: List[bool]
    max_dx: List[float]
    keep_mask: np.ndarray
    backend: str
    accel: str
    n_fe: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FE prep
# ---------------------------------------------------------------------------

def _coerce_fe_columns(
    fe: Union[pd.DataFrame, np.ndarray, Sequence[np.ndarray]],
    n: int,
) -> List[np.ndarray]:
    """Return a list of K 1-D arrays, one per FE column, length n each."""
    if isinstance(fe, pd.DataFrame):
        return [fe.iloc[:, k].values for k in range(fe.shape[1])]
    if isinstance(fe, np.ndarray):
        if fe.ndim == 1:
            return [fe]
        if fe.ndim == 2:
            return [fe[:, k] for k in range(fe.shape[1])]
        raise ValueError("fe ndarray must be 1-D or 2-D")
    # Sequence of arrays
    cols = [np.asarray(c) for c in fe]
    for c in cols:
        if c.shape[0] != n:
            raise ValueError(
                f"fe column has length {c.shape[0]} but X has length {n}"
            )
    return cols


def _factorize_columns(cols: List[np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
    """Densify each FE column to int64 codes in [0, G_k)."""
    out_codes: List[np.ndarray] = []
    out_card: List[int] = []
    for c in cols:
        codes, uniq = pd.factorize(c, sort=False, use_na_sentinel=True)
        if (codes < 0).any():
            raise ValueError("fast.demean: NaN in FE column not allowed")
        out_codes.append(codes.astype(np.int64))
        out_card.append(len(uniq))
    return out_codes, out_card


def _detect_singletons_numpy(fe_codes_raw: List[np.ndarray], n: int) -> np.ndarray:
    """Iterative K-way singleton drop. Pure-NumPy reference path."""
    keep = np.ones(n, dtype=bool)
    while True:
        dropped = False
        for codes_k in fe_codes_raw:
            ck = codes_k[keep]
            counts = np.bincount(ck)
            singles = np.where(counts == 1)[0]
            if singles.size == 0:
                continue
            mask_local = np.isin(ck, singles)
            if mask_local.any():
                gidx = np.where(keep)[0]
                keep[gidx[mask_local]] = False
                dropped = True
        if not dropped:
            break
    return keep


def _detect_singletons(fe_codes_raw: List[np.ndarray], n: int) -> np.ndarray:
    """Singleton mask. Prefers the Rust path; falls back to NumPy."""
    if _HAS_RUST and len(fe_codes_raw) > 0:
        try:
            mask_u8 = _rust.singleton_mask(list(fe_codes_raw))  # type: ignore
            return mask_u8.astype(bool, copy=False)
        except Exception:
            pass
    return _detect_singletons_numpy(fe_codes_raw, n)


# ---------------------------------------------------------------------------
# AP loop — NumPy reference implementation
# ---------------------------------------------------------------------------

def _aitken(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Vector Irons-Tuck extrapolation (matches Rust impl bit-for-bit)."""
    dx1 = x1 - x0
    d2 = x2 - 2.0 * x1 + x0
    den = float(d2 @ d2)
    if den < 1e-30:
        return x2
    alpha = float(dx1 @ d2) / den
    return x0 - alpha * dx1


def _sweep_one_fe(col: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> None:
    """In-place: col -= group_mean(col, codes). NumPy via bincount."""
    sums = np.bincount(codes, weights=col, minlength=counts.size)
    means = sums / np.maximum(counts, 1.0)
    col -= means[codes]


def _sweep_all_fe(
    col: np.ndarray, fe_codes: List[np.ndarray], counts_list: List[np.ndarray]
) -> None:
    for k in range(len(fe_codes)):
        _sweep_one_fe(col, fe_codes[k], counts_list[k])


def _demean_numpy_one_column(
    x: np.ndarray,
    fe_codes: List[np.ndarray],
    counts_list: List[np.ndarray],
    *,
    max_iter: int,
    tol_abs: float,
    tol_rel: float,
    accelerate: bool,
    accel_period: int,
) -> Tuple[np.ndarray, int, bool, float]:
    """Pure-NumPy AP loop matching the Rust algorithm."""
    K = len(fe_codes)
    if K == 0:
        return x, 0, True, 0.0
    if K == 1:
        _sweep_one_fe(x, fe_codes[0], counts_list[0])
        return x, 1, True, 0.0

    base_scale = float(np.max(np.abs(x))) + 1e-30
    stop = tol_abs + tol_rel * base_scale
    hist: List[np.ndarray] = []
    max_dx = float("inf")
    iters = 0
    converged = False
    for it in range(max_iter):
        before = x.copy()
        _sweep_all_fe(x, fe_codes, counts_list)
        max_dx = float(np.max(np.abs(x - before)))
        iters = it + 1
        if max_dx <= stop:
            converged = True
            break
        if accelerate:
            hist.append(x.copy())
            if len(hist) >= 3 and (it + 1) % accel_period == 0:
                acc = _aitken(hist[-3], hist[-2], hist[-1])
                if np.max(np.abs(acc)) < 10.0 * base_scale:
                    x[:] = acc
                hist.clear()
    return x, iters, converged, max_dx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def demean(
    X: np.ndarray,
    fe: Union[pd.DataFrame, np.ndarray, Sequence[np.ndarray]],
    *,
    accel: str = "aitken",
    max_iter: int = 1_000,
    tol: float = 1e-8,
    tol_abs: float = 0.0,
    accel_period: int = 5,
    drop_singletons: bool = True,
    backend: str = "auto",
    jax_max_iter: Optional[int] = None,
) -> Tuple[np.ndarray, DemeanInfo]:
    """Multi-way HDFE within-transform with Aitken acceleration.

    Parameters
    ----------
    X : ndarray, shape (n,) or (n, p)
        Outcome / regressor(s) to residualise. Float64 is preferred;
        other dtypes are upcast.
    fe : DataFrame, ndarray (n, K), or list of K 1-D arrays
        Fixed-effect columns. Any dtype factorisable by ``pd.factorize``
        (numeric, str, categorical) is accepted; NaN raises.
    accel : {"aitken", "none"}
        Acceleration. "aitken" uses Irons-Tuck extrapolation every
        ``accel_period`` sweeps once 3 history iterates exist (SQUAREM
        layout); "none" disables it. Default "aitken".
    max_iter : int
        Cap on AP iterations per column.
    tol : float
        Relative convergence threshold; multiplied by ``max|x_initial|``
        to form the per-iteration ``max|dx|`` stop criterion.
    tol_abs : float
        Absolute floor added to the stop criterion. Combined: stop when
        ``max|dx| <= tol_abs + tol * base_scale``.
    accel_period : int
        Sweeps between Aitken jumps.
    drop_singletons : bool
        If True, iteratively remove rows whose FE level appears once.
    backend : {"auto", "rust", "numpy"}
        Force a specific kernel; "auto" prefers Rust if available.

    Returns
    -------
    X_dem : ndarray
        Residualised X. Shape ``(n_kept,)`` or ``(n_kept, p)`` after
        singleton drops (same row count as input if all kept).
    info : DemeanInfo
        Convergence + provenance diagnostics.

    Raises
    ------
    ValueError
        On NaN in X or FE columns, dimension mismatches, or unknown
        ``accel`` / ``backend`` values.
    RuntimeError
        If ``backend="rust"`` is forced but the extension is missing.
    """
    if accel not in ("aitken", "none"):
        raise ValueError(f"accel={accel!r}; expected 'aitken' or 'none'")
    if backend not in ("auto", "rust", "numpy", "jax"):
        raise ValueError(f"backend={backend!r}; expected 'auto'|'rust'|'numpy'|'jax'")

    X = np.asarray(X)
    if X.dtype != np.float64:
        X = X.astype(np.float64)

    if X.ndim == 1:
        squeeze = True
        X = X.reshape(-1, 1)
    elif X.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"X must be 1-D or 2-D, got ndim={X.ndim}")

    n, p = X.shape
    if not np.isfinite(X).all():
        raise ValueError("fast.demean: non-finite values in X")

    # Factorise FE columns
    fe_cols = _coerce_fe_columns(fe, n)
    fe_codes_raw, n_fe_raw = _factorize_columns(fe_cols)

    # Singleton pruning
    if drop_singletons:
        keep_mask = _detect_singletons(fe_codes_raw, n)
    else:
        keep_mask = np.ones(n, dtype=bool)
    n_kept = int(keep_mask.sum())
    n_dropped = n - n_kept

    # Re-densify codes on kept rows (so they're contiguous in [0, G_k))
    fe_codes: List[np.ndarray] = []
    counts_list: List[np.ndarray] = []
    n_fe_post: List[int] = []
    for codes_k in fe_codes_raw:
        ck = codes_k[keep_mask]
        dense, uniq = pd.factorize(ck, sort=False, use_na_sentinel=True)
        dense = dense.astype(np.int64)
        G = len(uniq)
        fe_codes.append(dense)
        counts_list.append(np.bincount(dense, minlength=G).astype(np.float64))
        n_fe_post.append(G)

    # Mask X to surviving rows
    if n_dropped > 0:
        X = X[keep_mask].copy()
    else:
        X = X.copy()

    accelerate = (accel == "aitken")

    # Hand off to the shared core (also used by ``WithinTransformer.transform``)
    X, iters_out, converged_out, max_dx_out, backend_used = _demean_core(
        X, fe_codes, counts_list,
        max_iter=max_iter, tol=tol, tol_abs=tol_abs,
        accelerate=accelerate, accel_period=accel_period,
        backend=backend, jax_max_iter=jax_max_iter,
    )

    info = DemeanInfo(
        n=n, n_kept=n_kept, n_dropped=n_dropped,
        iters=iters_out, converged=converged_out, max_dx=max_dx_out,
        keep_mask=keep_mask, backend=backend_used, accel=accel,
        n_fe=n_fe_post,
    )
    if squeeze:
        return X.ravel(), info
    return X, info


def _demean_core(
    X: np.ndarray,
    fe_codes: List[np.ndarray],
    counts_list: List[np.ndarray],
    *,
    max_iter: int,
    tol: float,
    tol_abs: float,
    accelerate: bool,
    accel_period: int,
    backend: str,
    jax_max_iter: Optional[int] = None,
) -> Tuple[np.ndarray, List[int], List[bool], List[float], str]:
    """Run the demean kernel on already-prepared dense codes/counts.

    Caller is responsible for: factorising FE columns, applying any
    singleton mask, and providing a *mutable copy* of X (we may write
    into it in place via the Rust path). Returns the residualised X
    plus per-column iter / converged / max_dx and the backend label.

    This split lets ``WithinTransformer.transform`` skip the
    factorise / singleton pipeline that the cached transformer has
    already done — Round 4 audit fix.
    """
    iters_out: List[int] = []
    converged_out: List[bool] = []
    max_dx_out: List[float] = []
    if X.ndim == 1:
        # caller guarantees 2-D, but be defensive
        X = X.reshape(-1, 1)
    n, p = X.shape

    if backend == "jax":
        # JAX backend (Phase 7): structurally GPU-ready; on CPU it's the
        # slowest of the three options. We trust the user's explicit ask.
        from .jax_backend import demean_jax, _HAS_JAX
        if not _HAS_JAX:
            raise RuntimeError(
                "backend='jax' requested but jax is not installed; "
                "pip install jax jaxlib to enable, or use a different backend."
            )
        X_dem, converged_jax = demean_jax(
            X, fe_codes, counts_list,
            max_iter=jax_max_iter or max_iter,
            tol=tol, accelerate=accelerate, accel_period=accel_period,
        )
        if X_dem.ndim == 1:
            X_dem = X_dem.reshape(-1, 1)
        return (
            X_dem,
            [0] * X_dem.shape[1],          # JAX doesn't expose iter counts
            list(converged_jax),
            [0.0] * X_dem.shape[1],
            "jax",
        )

    use_rust = _HAS_RUST and (backend in ("auto", "rust"))
    if backend == "rust" and not _HAS_RUST:
        raise RuntimeError(
            "backend='rust' requested but statspai_hdfe extension not installed; "
            "build via `cd rust/statspai_hdfe && maturin develop --release` "
            "or use backend='numpy'/'auto'."
        )
    if backend == "numpy":
        use_rust = False

    if use_rust and len(fe_codes) > 0:
        # Avoid an extra copy when X is already F-contig (a common case
        # when WithinTransformer caches things; Round 5 audit fix).
        if X.flags["F_CONTIGUOUS"]:
            X_f = X
        else:
            X_f = np.asfortranarray(X)
        infos = _rust.demean_2d(  # type: ignore
            X_f,
            fe_codes,
            counts_list,
            int(max_iter),
            float(tol_abs),
            float(tol),
            bool(accelerate),
            int(accel_period),
        )
        # Return the F-contig array directly. NumPy can read it without
        # a second copy; downstream code that assumes C-contig must
        # ascontiguous itself (callers in this file are tolerant).
        for d in infos:
            iters_out.append(int(d["iters"]))
            converged_out.append(bool(d["converged"]))
            max_dx_out.append(float(d["max_dx"]))
        return X_f, iters_out, converged_out, max_dx_out, "rust"

    # NumPy fallback
    for j in range(p):
        col = np.ascontiguousarray(X[:, j])
        col, iters, converged, max_dx = _demean_numpy_one_column(
            col,
            fe_codes,
            counts_list,
            max_iter=max_iter,
            tol_abs=tol_abs,
            tol_rel=tol,
            accelerate=accelerate,
            accel_period=accel_period,
        )
        X[:, j] = col
        iters_out.append(iters)
        converged_out.append(converged)
        max_dx_out.append(max_dx)
    return X, iters_out, converged_out, max_dx_out, "numpy"


__all__ = ["demean", "DemeanInfo"]
