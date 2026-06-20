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

Both accept ``polars.DataFrame``, ``polars.LazyFrame`` (lazy inputs are
collected before column extraction), or ``pyarrow.Table``. We don't fuse
the demean into Polars' query plan.
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

from ..exceptions import MethodIncompatibility
from .demean import demean as _fast_demean, DemeanInfo
from .fepois import fepois as _fast_fepois, FePoisResult

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _normalize_columns(cols: Sequence[str] | str, *, name: str) -> List[str]:
    """Return a non-empty list of string column names."""
    if isinstance(cols, str):
        out = [cols]
    else:
        try:
            out = list(cols)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"{name} must be a non-empty sequence of column names"
            ) from exc
    if not out:
        raise MethodIncompatibility(f"{name} must be non-empty")
    bad = [c for c in out if not isinstance(c, str) or not c]
    if bad:
        raise MethodIncompatibility(
            f"{name} must contain only non-empty string column names"
        )
    return out


def _ensure_columnar_eager(obj: Any) -> Any:
    """Resolve supported lazy/eager columnar inputs.

    Polars LazyFrames are collected. Polars DataFrames and PyArrow Tables
    already expose eager column access and are returned unchanged.
    """
    if _HAS_POLARS and isinstance(obj, pl.LazyFrame):
        return obj.collect()
    if _HAS_POLARS and isinstance(obj, pl.DataFrame):
        return obj
    if _HAS_ARROW and isinstance(obj, pa.Table):
        return obj
    raise MethodIncompatibility(
        "expected polars DataFrame/LazyFrame or pyarrow Table, "
        f"got {type(obj).__name__}"
    )


def _polars_to_numpy_zero_copy(series: Any) -> np.ndarray:
    """Best-effort zero-copy conversion of a Polars Series to NumPy.

    Polars 1.x exposes ``.to_numpy(allow_copy=False)`` which succeeds
    for Series whose memory layout is already a contiguous NumPy-
    compatible buffer (most numeric dtypes, no nulls). When that fails
    we fall back to the safe copying path. The dtype-cast path
    (e.g. int → float) inevitably copies and we accept that.
    """
    try:
        return np.asarray(series.to_numpy(allow_copy=False))
    except (TypeError, RuntimeError, ValueError):
        # Older Polars versions may raise different exception types;
        # also: presence of nulls / non-contiguous storage forces copy.
        return np.asarray(series.to_numpy())


def _arrow_to_numpy_zero_copy(column: Any) -> np.ndarray:
    """Best-effort conversion of a PyArrow ChunkedArray to NumPy."""
    try:
        return np.asarray(column.to_numpy(zero_copy_only=True))
    except (pa.ArrowInvalid, ValueError, TypeError):
        return np.asarray(column.to_numpy(zero_copy_only=False))


def _column_names(df: Any) -> List[str]:
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return list(df.columns)
    if _HAS_ARROW and isinstance(df, pa.Table):
        return list(df.column_names)
    raise MethodIncompatibility(
        f"unsupported columnar object type: {type(df).__name__}"
    )


def _n_rows(df: Any) -> int:
    if _HAS_ARROW and isinstance(df, pa.Table):
        return int(df.num_rows)
    return len(df)


def _column_to_numpy(df: Any, column: str) -> np.ndarray:
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return _polars_to_numpy_zero_copy(df[column])
    if _HAS_ARROW and isinstance(df, pa.Table):
        return _arrow_to_numpy_zero_copy(df[column])
    raise MethodIncompatibility(
        f"unsupported columnar object type: {type(df).__name__}"
    )


def _select_to_pandas(df: Any, cols: Sequence[str]) -> pd.DataFrame:
    if _HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.select(cols).to_pandas()
    if _HAS_ARROW and isinstance(df, pa.Table):
        return df.select(cols).to_pandas()
    raise MethodIncompatibility(
        f"unsupported columnar object type: {type(df).__name__}"
    )


def _require_columns(df: Any, cols: Sequence[str], *, context: str) -> None:
    missing = [c for c in cols if c not in _column_names(df)]
    if missing:
        raise MethodIncompatibility(f"{context}: missing columns: {missing}")


def _columns_to_numpy(df: Any, cols: Sequence[str]) -> np.ndarray:
    """Stack the named columns into a (n, len(cols)) float64 ndarray."""
    _require_columns(df, cols, context="fast.demean_polars")
    arrays: List[np.ndarray] = []
    for c in cols:
        arr = _column_to_numpy(df, c)
        if arr.dtype != np.float64:
            # dtype cast forces a copy regardless; accept it
            arr = arr.astype(np.float64)
        arrays.append(arr)
    return np.column_stack(arrays) if arrays else np.empty((_n_rows(df), 0))


def _columns_as_object(df: Any, cols: Sequence[str]) -> List[np.ndarray]:
    """Extract columns as 1-D NumPy arrays (any dtype) for FE factorisation."""
    _require_columns(df, cols, context="fast.demean_polars")
    out: List[np.ndarray] = []
    for c in cols:
        out.append(_column_to_numpy(df, c))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def demean_polars(
    df: Any,
    X_cols: Sequence[str] | str,
    fe_cols: Sequence[str] | str,
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
    df = _ensure_columnar_eager(df)
    X_cols = _normalize_columns(X_cols, name="X_cols")
    fe_cols = _normalize_columns(fe_cols, name="fe_cols")
    X = _columns_to_numpy(df, X_cols)
    fe_arrays = _columns_as_object(df, fe_cols)
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
    df = _ensure_columnar_eager(df)

    from .fepois import _parse_fepois_formula

    lhs, rhs_terms, fe_terms = _parse_fepois_formula(formula)
    needed = [lhs] + [t for t in rhs_terms if t != "1"] + list(fe_terms)
    needed = list(dict.fromkeys(needed))  # de-dupe, preserve order
    _require_columns(df, needed, context="fast.fepois_polars")
    pandas_df = _select_to_pandas(df, needed)
    return _fast_fepois(formula, pandas_df, **kwargs)


__all__ = ["demean_polars", "fepois_polars"]
