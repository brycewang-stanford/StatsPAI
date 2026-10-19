"""Multi-threaded CSV reading for ``sp.read_data`` (pyarrow, when installed).

``pyarrow.csv`` parses several times faster than pandas' C reader (0.3 s
against 1.8 s on two million rows and six columns) and rounds decimal
text to the nearest double, which pandas' default converter does not
always do (on that file it was off by one unit in the last place for a
third of the values). The frame must still be the one pandas would give,
so the fast path

* only runs for a plain call (no reader keywords beyond the separator,
  ``usecols`` and an UTF-8 / ASCII encoding) on a file above
  ``_MIN_BYTES``;
* treats pandas' missing-value tokens as missing and parses no dates or
  times (date-looking text stays text, as pandas leaves it);
* turns all-missing columns into ``float64`` ``NaN``, as pandas does;
* compares the column types with pandas' own reading of the first rows
  and returns ``None`` (the caller then uses pandas) on any difference,
  on duplicate column names, and on any parse error.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

#: Below this size pandas' reader is as fast and the check costs more.
_MIN_BYTES = 4 * 1024 * 1024
_SAFE_KEYS = {"sep", "delimiter", "usecols"}
_ENCODINGS = {None, "utf-8", "utf8", "UTF-8", "ascii"}
_CHECK_ROWS = 2000


def _kind(dtype: Any) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "b"
    if pd.api.types.is_integer_dtype(dtype):
        return "i"
    if pd.api.types.is_float_dtype(dtype):
        return "f"
    return "O"


def read_csv_fast(
    path: Any, encoding: Optional[str], kwargs: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """The frame ``pd.read_csv(path, encoding=encoding, **kwargs)`` gives,
    read by pyarrow; ``None`` when that cannot be guaranteed."""
    if set(kwargs) - _SAFE_KEYS or encoding not in _ENCODINGS:
        return None
    sep = kwargs.get("sep", kwargs.get("delimiter", ","))
    if not isinstance(sep, str) or len(sep) != 1:
        return None
    usecols = kwargs.get("usecols")
    if usecols is not None and not (
        isinstance(usecols, (list, tuple)) and all(isinstance(c, str) for c in usecols)
    ):
        return None
    try:
        if os.path.getsize(path) < _MIN_BYTES:
            return None
        import pyarrow as pa
        import pyarrow.csv as pacsv
        from pandas._libs.parsers import STR_NA_VALUES
    except (ImportError, OSError, TypeError):
        return None

    def _read(column_types: Optional[Dict[str, Any]] = None) -> Any:
        return pacsv.read_csv(
            path,
            parse_options=pacsv.ParseOptions(delimiter=sep),
            convert_options=pacsv.ConvertOptions(
                null_values=sorted(STR_NA_VALUES),
                strings_can_be_null=True,
                timestamp_parsers=[],
                include_columns=list(usecols) if usecols else None,
                column_types=column_types or {},
            ),
        )

    try:
        table = _read()
        names = table.column_names
        if len(set(names)) != len(names) or any(n == "" for n in names):
            return None
        temporal = {
            f.name: pa.string() for f in table.schema if pa.types.is_temporal(f.type)
        }
        if temporal:
            table = _read(temporal)
        df = table.to_pandas()
    except (pa.ArrowException, ValueError, TypeError):
        return None
    for f in table.schema:
        if pa.types.is_null(f.type):
            df[f.name] = np.full(len(df), np.nan)
        elif df[f.name].dtype == object:
            # pandas marks a missing text cell NaN, pyarrow None
            arr = df[f.name].to_numpy(dtype=object)
            miss = pd.isna(arr)
            if miss.any():
                arr[miss] = np.nan
                df[f.name] = arr
    head = pd.read_csv(path, encoding=encoding, nrows=_CHECK_ROWS, **kwargs)
    if set(head.columns) != set(df.columns):
        return None
    df = df[list(head.columns)]  # usecols: pandas keeps the file's order
    for c in df.columns:
        if head[c].isna().all():
            continue  # all missing in the first rows: pandas' type says nothing
        if _kind(head[c].dtype) != _kind(df[c].dtype):
            return None
    return df


__all__ = ["read_csv_fast"]
