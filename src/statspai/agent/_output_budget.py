"""Output shaping for MCP ``tools/call`` results.

Two concerns live here, both applied to the JSON object that becomes a
result's ``structuredContent`` (and, serialised compactly, its ``text``
block):

* **Non-finite values.** JSON has no NaN / Infinity, so they are sent as
  ``null`` — which on its own is indistinguishable from "missing". The
  paths of every value that was non-finite are listed under a top-level
  ``_nonfinite`` key (only when there are any), so an agent can tell an
  infinite standard error from an absent one.
* **Byte budget.** A result can carry a coefficient table with thousands
  of rows, per-unit weights or bootstrap draws. :func:`apply_budget`
  keeps the serialised object under ``max_bytes`` by shortening the
  largest lists (and table-like dicts, and very long strings) first and
  records each cut under a top-level ``truncated`` key as
  ``{"path", "total", "shown"}``. Headline fields (estimate, standard
  error, confidence interval, p-value, error fields, handles) are never
  cut.

Paths are JSON Pointers (RFC 6901): ``/coefficients/x/std_error``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

#: Env var: default output byte budget per ``tools/call`` (``0`` disables).
MAX_OUTPUT_BYTES_ENV = "STATSPAI_MCP_MAX_OUTPUT_BYTES"

#: Default budget for the ``structuredContent`` object (the ``text`` block
#: carries the same object once more, compactly serialised).
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024

#: At most this many non-finite paths are listed; the total is reported
#: separately when the list is capped.
MAX_NONFINITE_PATHS = 100

#: Top-level keys never truncated (headline numbers, error fields, handles).
PROTECTED_KEYS = frozenset(
    {
        "estimate",
        "std_error",
        "se",
        "ci",
        "conf_int",
        "conf_low",
        "conf_high",
        "ci_lower",
        "ci_upper",
        "p_value",
        "pvalue",
        "pval",
        "estimand",
        "method",
        "n_obs",
        "error",
        "error_kind",
        "message",
        "hint",
        "miss_reason",
        "result_id",
        "result_uri",
        "data_id",
        "data_uri",
        "replay",
        "tool",
        "truncated",
        "_nonfinite",
        "_nonfinite_total",
    }
)

#: Dicts with more keys than this are treated as tables and may be cut.
_TABLE_DICT_MIN_KEYS = 20

#: Strings longer than this may be shortened.
_LONG_STRING = 2048


def max_output_bytes(override: Any = None) -> Optional[int]:
    """Resolve the budget: explicit ``override`` > env var > default.

    Returns ``None`` when the budget is disabled (``0``).
    """
    raw: Any = override
    if raw is None:
        raw = os.environ.get(MAX_OUTPUT_BYTES_ENV)
    if raw is None:
        return DEFAULT_MAX_OUTPUT_BYTES
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_OUTPUT_BYTES
    return v if v > 0 else None


def _pointer_token(key: Any) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def scrub_nonfinite(obj: Any) -> Tuple[Any, List[Dict[str, str]], int]:
    """Replace NaN / ±Inf floats with ``None`` and record where they were.

    Returns ``(clean, paths, total)`` where ``paths`` lists up to
    :data:`MAX_NONFINITE_PATHS` entries ``{"path": <pointer>, "value":
    "NaN" | "Infinity" | "-Infinity"}`` and ``total`` counts them all.
    """
    found: List[Dict[str, str]] = []
    count = [0]

    def _walk(o: Any, path: str) -> Any:
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                count[0] += 1
                if len(found) < MAX_NONFINITE_PATHS:
                    label = (
                        "NaN"
                        if math.isnan(o)
                        else ("Infinity" if o > 0 else "-Infinity")
                    )
                    found.append({"path": path or "/", "value": label})
                return None
            return o
        if isinstance(o, dict):
            return {k: _walk(v, f"{path}/{_pointer_token(k)}") for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_walk(v, f"{path}/{i}") for i, v in enumerate(o)]
        return o

    clean = _walk(obj, "")
    return clean, found, count[0]


def json_size(o: Any) -> int:
    """Bytes of ``o`` serialised compactly (as the ``text`` block is)."""
    return len(json.dumps(o, separators=(",", ":"), allow_nan=False))


def _measure(
    node: Any,
    path: str,
    depth: int,
    cands: List[Tuple[int, str, Any, Any, Any]],
    parent: Any,
    key: Any,
) -> int:
    """Exact compact-JSON size of ``node``; collects truncation candidates.

    A candidate is ``(size, path, node, parent, key)``.
    """
    if isinstance(node, dict):
        size = 2 + max(len(node) - 1, 0)
        for k, v in node.items():
            child_path = f"{path}/{_pointer_token(k)}"
            protected = depth == 0 and k in PROTECTED_KEYS
            if protected:
                size += json_size(str(k)) + 1 + json_size(v)
                continue
            size += (
                json_size(str(k))
                + 1
                + _measure(v, child_path, depth + 1, cands, node, k)
            )
        if depth > 0 and len(node) > _TABLE_DICT_MIN_KEYS:
            cands.append((size, path, node, parent, key))
        return size
    if isinstance(node, list):
        size = 2 + max(len(node) - 1, 0)
        for i, v in enumerate(node):
            size += _measure(v, f"{path}/{i}", depth + 1, cands, node, i)
        if len(node) > 1:
            cands.append((size, path, node, parent, key))
        return size
    size = json_size(node)
    if isinstance(node, str) and len(node) > _LONG_STRING and parent is not None:
        cands.append((size, path, node, parent, key))
    return size


def apply_budget(
    obj: Dict[str, Any], max_bytes: Optional[int]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Shrink ``obj`` in place until it serialises to at most ``max_bytes``.

    The largest truncatable container is cut first, to the number of
    items that removes the current excess (never below one item); the
    loop repeats until the object fits or nothing is left to cut.
    Returns ``(obj, truncated)`` where ``truncated`` lists
    ``{"path", "total", "shown"}`` per cut container (``total`` / ``shown``
    count items, or characters for strings).
    """
    if not max_bytes or not isinstance(obj, dict):
        return obj, []
    records: Dict[str, Dict[str, Any]] = {}
    for _ in range(500):
        cands: List[Tuple[int, str, Any, Any, Any]] = []
        total = _measure(obj, "", 0, cands, None, None)
        # Reserve room for the ``truncated`` ledger itself.
        excess = (
            total
            - max_bytes
            + (json_size(list(records.values())) + 16 if records else 64)
        )
        if excess <= 0:
            break
        cands = [c for c in cands if c[2] is not None and len(c[2]) > 1]
        if not cands:
            break
        size, path, node, parent, key = max(cands, key=lambda c: c[0])
        n = len(node)
        if isinstance(node, str):
            keep = max(256, n - excess - 32)
            if keep >= n:
                keep = n // 2
            new: Any = node[:keep] + "...[truncated]"
            shown = keep
        else:
            per_item = max(size / n, 1.0)
            drop = int(math.ceil(excess / per_item)) + 1
            shown = max(1, n - drop)
            if shown >= n:
                shown = n - 1
            if isinstance(node, dict):
                new = {k: node[k] for k in list(node)[:shown]}
            else:
                new = node[:shown]
        parent[key] = new
        rec = records.get(path)
        if rec is None:
            records[path] = {"path": path, "total": n, "shown": shown}
        else:
            rec["shown"] = shown
    return obj, list(records.values())


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "MAX_OUTPUT_BYTES_ENV",
    "PROTECTED_KEYS",
    "apply_budget",
    "json_size",
    "max_output_bytes",
    "scrub_nonfinite",
]
