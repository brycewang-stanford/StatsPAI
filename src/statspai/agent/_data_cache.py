"""Server-side DataFrame handles for the MCP layer.

Before this module the only way to hand data to a tool was an absolute
file path. An agent that filtered, reshaped or imputed a frame could not
pass the derived frame to the next call, and a small hand-made table
needed a temporary file. Three additions close that gap:

* :data:`DATA_CACHE` — an LRU cache of DataFrames keyed by ``data_id``
  (``d_…``), the twin of :data:`~statspai.agent._result_cache.RESULT_CACHE`.
  Every tool accepts ``data_id`` wherever it accepts ``data_path``.
* Inline input — ``data_records`` (a JSON array of row objects) or
  ``data_csv`` (a CSV string) on any tool, bounded by the same byte cap
  as file loads, recorded in ``data_provenance`` as ``source_type:
  "inline"`` with a SHA-256 of the canonical bytes.
* Lineage — a handle produced by ``transform_data`` remembers its parent
  handle and the operations applied, and that chain rides along in
  ``data_provenance`` on every result fitted from it, so a table note can
  say exactly how the analysis sample was built.

Handles are process-local (they do not survive a server restart); a
missing handle is a recoverable error that tells the agent to reload.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._result_cache import ResultCache

_DEFAULT_DATA_CACHE_SIZE = 16

#: Head rows echoed in ``load_data`` / ``describe_data`` / the resource.
HEAD_ROWS = 5


def _data_cache_size() -> int:
    raw = os.environ.get("STATSPAI_MCP_DATA_CACHE_SIZE")
    if raw is None:
        return _DEFAULT_DATA_CACHE_SIZE
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_DATA_CACHE_SIZE


class DataCache(ResultCache):
    """LRU cache of DataFrames; ids are ``d_<hex>``.

    ``arguments`` on each entry holds the provenance / lineage record
    (``source_type``, ``parent_id``, ``operations`` …) so the resource
    view and ``data_provenance`` can be rebuilt from the handle alone.
    """

    _prefix = "d_"

    def __init__(self, max_size: Optional[int] = None) -> None:
        super().__init__(max_size=max_size or _data_cache_size())


DATA_CACHE = DataCache()


# ---------------------------------------------------------------------------
# Frame description
# ---------------------------------------------------------------------------


def _jsonable_scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def describe_frame(df: pd.DataFrame, *, head: int = HEAD_ROWS) -> Dict[str, Any]:
    """Compact, JSON-safe description of a DataFrame for an agent.

    Shape, dtypes, missing counts per column, the first ``head`` rows,
    and a numeric summary (mean / sd / min / max) for numeric columns.
    Bounded regardless of the frame's size.
    """
    n_rows, n_cols = df.shape
    dtypes = {str(c): str(t) for c, t in df.dtypes.items()}
    missing = {str(c): int(v) for c, v in df.isna().sum().items() if int(v) > 0}
    head_rows = [
        {str(k): _jsonable_scalar(v) for k, v in row.items()}
        for row in df.head(head).to_dict(orient="records")
    ]
    numeric: Dict[str, Dict[str, Any]] = {}
    num = df.select_dtypes(include="number")
    if not num.empty:
        desc = num.describe().T
        for col, row in desc.iterrows():
            numeric[str(col)] = {
                "mean": _jsonable_scalar(row.get("mean")),
                "sd": _jsonable_scalar(row.get("std")),
                "min": _jsonable_scalar(row.get("min")),
                "max": _jsonable_scalar(row.get("max")),
                "n_unique": int(num[col].nunique(dropna=True)),
            }
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": [str(c) for c in df.columns],
        "dtypes": dtypes,
        "missing": missing,
        "head": head_rows,
        "numeric_summary": numeric,
    }


# ---------------------------------------------------------------------------
# Inline data
# ---------------------------------------------------------------------------


def inline_frame(
    *,
    records: Optional[Any] = None,
    csv_text: Optional[str] = None,
    max_bytes: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a DataFrame from inline tool arguments.

    Exactly one of ``records`` (list of row objects) or ``csv_text``
    (CSV string) must be given. The serialised size is capped by
    ``max_bytes`` — the same budget as file loads — so a client cannot
    push an arbitrarily large table through the JSON-RPC channel.

    Returns the frame and a provenance record (``source_type="inline"``,
    ``format``, ``n_rows``, ``sha256`` of the canonical bytes).
    """
    from ..exceptions import MethodIncompatibility

    if records is not None and csv_text is not None:
        raise MethodIncompatibility(
            "Pass either data_records or data_csv, not both.",
            recovery_hint="Send the table once, as records or as CSV text.",
        )
    if records is not None:
        if not isinstance(records, list) or not all(
            isinstance(r, dict) for r in records
        ):
            raise MethodIncompatibility(
                "data_records must be a JSON array of row objects "
                "(e.g. [{'y': 1.0, 'x': 2.0}, ...]).",
                recovery_hint="Send rows as objects keyed by column name.",
            )
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(canonical) > max_bytes:
            raise MethodIncompatibility(
                f"data_records is {len(canonical):,} bytes, over the "
                f"{max_bytes:,}-byte inline cap.",
                recovery_hint=(
                    "Write the table to a file and pass data_path, or raise "
                    "STATSPAI_MCP_MAX_DATA_BYTES on the server."
                ),
            )
        if not records:
            raise MethodIncompatibility(
                "data_records is empty.",
                recovery_hint="Send at least one row.",
            )
        df = pd.DataFrame.from_records(records)
        fmt = "records"
    elif csv_text is not None:
        if not isinstance(csv_text, str):
            raise MethodIncompatibility(
                "data_csv must be a CSV string.",
                recovery_hint="Send the table as CSV text with a header row.",
            )
        canonical = csv_text.encode("utf-8")
        if len(canonical) > max_bytes:
            raise MethodIncompatibility(
                f"data_csv is {len(canonical):,} bytes, over the "
                f"{max_bytes:,}-byte inline cap.",
                recovery_hint=(
                    "Write the table to a file and pass data_path, or raise "
                    "STATSPAI_MCP_MAX_DATA_BYTES on the server."
                ),
            )
        if not csv_text.strip():
            raise MethodIncompatibility(
                "data_csv is empty.",
                recovery_hint="Send CSV text with a header row and data rows.",
            )
        df = pd.read_csv(io.StringIO(csv_text))
        fmt = "csv"
    else:
        raise MethodIncompatibility(
            "No inline data given.",
            recovery_hint="Pass data_records or data_csv.",
        )
    prov = {
        "source": f"inline:{fmt}",
        "source_type": "inline",
        "format": fmt,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "hash_status": "hashed_inline",
    }
    return df, prov


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


def register_frame(
    df: pd.DataFrame,
    *,
    provenance: Dict[str, Any],
    parent_id: Optional[str] = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    tool: str = "load_data",
) -> str:
    """Put ``df`` in :data:`DATA_CACHE` and return its ``data_id``.

    ``provenance`` is the record describing where the frame came from
    (a file, inline data, or — for a transform — the parent handle).
    ``operations`` lists the transform steps applied to the parent.
    """
    record: Dict[str, Any] = {
        "source_provenance": dict(provenance),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
    }
    if parent_id is not None:
        record["parent_id"] = parent_id
    if operations:
        record["operations"] = list(operations)
    return DATA_CACHE.put(df, tool=tool, arguments=record)


def lineage(data_id: str, *, max_depth: int = 32) -> List[Dict[str, Any]]:
    """Walk parent links from ``data_id`` to the root, newest first.

    Each item: ``{"data_id", "tool", "operations", "n_rows", "n_cols"}``
    plus the root's ``source_provenance``. Stops at a missing parent
    (evicted from the cache) with a ``"missing": True`` marker so the
    agent can see the chain is truncated rather than complete.
    """
    chain: List[Dict[str, Any]] = []
    cur: Optional[str] = data_id
    depth = 0
    while cur is not None and depth < max_depth:
        entry = DATA_CACHE.get_entry(cur)
        if entry is None:
            chain.append({"data_id": cur, "missing": True})
            break
        rec = entry.arguments
        item: Dict[str, Any] = {
            "data_id": cur,
            "tool": entry.tool,
            "n_rows": rec.get("n_rows"),
            "n_cols": rec.get("n_cols"),
        }
        if rec.get("operations"):
            item["operations"] = rec["operations"]
        if "parent_id" not in rec:
            item["source_provenance"] = rec.get("source_provenance", {})
        chain.append(item)
        cur = rec.get("parent_id")
        depth += 1
    return chain


def handle_provenance(data_id: str) -> Dict[str, Any]:
    """``data_provenance`` block for a tool call that consumed a handle."""
    chain = lineage(data_id)
    root = chain[-1] if chain else {}
    out: Dict[str, Any] = {
        "source": f"handle:{data_id}",
        "source_type": "handle",
        "data_id": data_id,
        "lineage": chain,
    }
    root_prov = root.get("source_provenance") if isinstance(root, dict) else None
    if isinstance(root_prov, dict) and root_prov:
        out["root"] = root_prov
    return out


def missing_handle_error(data_id: str) -> Dict[str, Any]:
    """Structured error for an unresolvable ``data_id``."""
    reason = DATA_CACHE.miss_reason(data_id)
    hints = {
        "ttl": "data_id expired (server data-cache TTL); reload with load_data.",
        "lru": (
            "data_id evicted (the data cache keeps only the most recent "
            "frames); reload with load_data."
        ),
        "explicit": "data_id was released; reload with load_data.",
    }
    return {
        "error": f"Unknown data_id: {data_id!r}",
        "error_kind": "missing_data_handle",
        "miss_reason": reason,
        "hint": hints.get(
            reason,
            "No such handle in this server process. Handles do not survive "
            "a restart; call load_data (data_path / data_records / data_csv) "
            "to get a fresh data_id.",
        ),
    }


__all__ = [
    "DATA_CACHE",
    "DataCache",
    "HEAD_ROWS",
    "describe_frame",
    "handle_provenance",
    "inline_frame",
    "lineage",
    "missing_handle_error",
    "register_frame",
]
