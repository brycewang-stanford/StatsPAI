"""Polars / PyArrow direct path for the Phase 1+ HDFE kernels.

Lets users pass ``polars.DataFrame`` or ``pyarrow.Table`` straight into
``sp.fast.demean`` / ``sp.fast.fepois`` without first materialising a
pandas DataFrame. For 1e8-row workloads this matters: pandas is the
memory bottleneck.

Today the Rust kernel still requires contiguous NumPy arrays at the FFI
boundary, so the "direct path" is currently a thin **adapter**: it
extracts each column as a NumPy array via the polars / arrow zero-copy
hooks (when possible) and dispatches to the existing kernels. A truly
zero-copy path that hands Arrow buffers directly to Rust is a
follow-up — it requires the C Data Interface bindings on the Rust
side, which we don't ship yet.

Public surface
--------------

    sp.fast.demean_polars(df_or_lf, X_cols, fe_cols, **kw)
    sp.fast.fepois_polars(df_or_lf, formula, **kw)

Both accept either ``polars.DataFrame`` or ``polars.LazyFrame`` (lazy
inputs are collected before column extraction; we don't fuse the demean
into Polars' query plan).
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except ImportError:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False

try:
    import pyarrow as pa  # type: ignore
    _HAS_ARROW = True
except ImportError:  # pragma: no cover
    pa = None  # type: ignore
    _HAS_ARROW = False

from .demean import demean as _fast_demean, DemeanInfo
from .fepois import fepois as _fast_fepois, FePoisResult


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _ensure_polars_eager(obj: Any):
    """Resolve ``LazyFrame`` to ``DataFrame``; identity for eager input."""
    if not _HAS_POLARS:
        raise RuntimeError(
            "polars is not installed; pip install polars to use this path"
        )
    if isinstance(obj, pl.LazyFrame):
        return obj.collect()
    if isinstance(obj, pl.DataFrame):
        return obj
    raise TypeError(
        f"expected polars DataFrame or LazyFrame, got {type(obj).__name__}"
    )


def _polars_to_numpy_zero_copy(series) -> np.ndarray:
    """Best-effort zero-copy conversion of a Polars Series to NumPy.

    Polars 1.x exposes ``.to_numpy(allow_copy=False)`` which succeeds
    for Series whose memory layout is already a contiguous NumPy-
    compatible buffer (most numeric dtypes, no nulls). When that fails
    we fall back to the safe copying path. The dtype-cast path
    (e.g. int → float) inevitably copies and we accept that.
    """
    try:
        return series.to_numpy(allow_copy=False)
    except (TypeError, RuntimeError, ValueError):
        # Older Polars versions may raise different exception types;
        # also: presence of nulls / non-contiguous storage forces copy.
        return series.to_numpy()


def _polars_columns_to_numpy(df, cols: Sequence[str]) -> np.ndarray:
    """Stack the named columns into a (n, len(cols)) float64 ndarray."""
    arrays: List[np.ndarray] = []
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"polars frame missing column {c!r}")
        arr = _polars_to_numpy_zero_copy(df[c])
        if arr.dtype != np.float64:
            # dtype cast forces a copy regardless; accept it
            arr = arr.astype(np.float64)
        arrays.append(arr)
    return np.column_stack(arrays) if arrays else np.empty((len(df), 0))


def _polars_columns_as_object(df, cols: Sequence[str]) -> List[np.ndarray]:
    """Extract columns as 1-D NumPy arrays (any dtype) for FE factorisation."""
    out: List[np.ndarray] = []
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"polars frame missing column {c!r}")
        out.append(_polars_to_numpy_zero_copy(df[c]))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def demean_polars(
    df: Any,
    X_cols: Sequence[str],
    fe_cols: Sequence[str],
    **kwargs: Any,
) -> Tuple[np.ndarray, DemeanInfo]:
    """Within-transform a polars DataFrame's columns by HDFE.

    Parameters
    ----------
    df : polars.DataFrame or polars.LazyFrame
    X_cols : list of str
        Continuous columns to residualise.
    fe_cols : list of str
        Fixed-effect columns. Any dtype factorisable by pd.factorize.
    **kwargs : forwarded to :func:`sp.fast.demean`.

    Returns
    -------
    (X_dem, info) : same as :func:`sp.fast.demean`.
    """
    df = _ensure_polars_eager(df)
    if len(X_cols) == 0:
        raise ValueError("X_cols must be non-empty")
    if len(fe_cols) == 0:
        raise ValueError("fe_cols must be non-empty")
    X = _polars_columns_to_numpy(df, list(X_cols))
    fe_arrays = _polars_columns_as_object(df, list(fe_cols))
    return _fast_demean(X, fe_arrays, **kwargs)


def fepois_polars(
    df: Any,
    formula: str,
    **kwargs: Any,
) -> FePoisResult:
    """Fit a Poisson HDFE model directly on a polars DataFrame.

    Internally collects the LazyFrame (if needed), materialises the
    referenced columns as a pandas DataFrame, and dispatches to
    ``sp.fast.fepois``. Materialisation is on a *projected* subset of
    columns — we don't pull the whole frame into pandas.
    """
    df = _ensure_polars_eager(df)

    from .fepois import _parse_fepois_formula
    lhs, rhs_terms, fe_terms = _parse_fepois_formula(formula)
    needed = [lhs] + [t for t in rhs_terms if t != "1"] + list(fe_terms)
    needed = list(dict.fromkeys(needed))  # de-dupe, preserve order
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"polars frame missing columns: {missing}")
    pandas_df = df.select(needed).to_pandas()
    return _fast_fepois(formula, pandas_df, **kwargs)


__all__ = ["demean_polars", "fepois_polars"]
