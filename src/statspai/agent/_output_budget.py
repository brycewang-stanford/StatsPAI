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
* **Risk fields are cut last, and never silently.** ``violations``,
  ``runtime_warnings``, ``degradations`` and ``warnings`` are shortened
  only after every other container, and when they are the object gains
  ``risk_details_complete: false`` and a ``risk_summary`` giving, per
  field, ``total`` / ``shown`` / ``omitted``, counts by severity and by
  category, and how to obtain the full list. A result that lost
  risk detail therefore cannot be read as a clean one.
* **The budget is a contract with a reported outcome.** Whenever the
  object was cut, or could not be brought under the budget because the
  never-cut fields alone exceed it, ``output_budget`` states ``status``
  (``truncated`` / ``unavoidable_overflow``), ``max_bytes``,
  ``actual_bytes`` and ``scope``. Its absence means the object fit
  untouched. The budget covers the ``structuredContent`` object only:
  the ``text`` block repeats it and an image block is sent on top.

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
        "replay_completeness",
        "isolation",
        "tool",
        "truncated",
        "risk_summary",
        "risk_details_complete",
        "output_budget",
        "_nonfinite",
        "_nonfinite_total",
    }
)

#: Top-level lists that carry statistical risk. Cut only when nothing else
#: is left to cut, and then summarised under ``risk_summary``.
RISK_KEYS = ("violations", "runtime_warnings", "degradations", "warnings")

#: Nested subtrees never cut: the configuration-level evidence of a result
#: card says which outputs were validated, and half of it is worse than none.
_PROTECTED_PATH_PREFIXES = ("/result_card/evidence",)

#: What ``output_budget.scope`` reports.
BUDGET_SCOPE = "structuredContent"

#: At most this many categories are counted per risk field.
_MAX_RISK_CATEGORIES = 20

#: Item fields read, in order, as the category of a risk entry.
_RISK_CATEGORY_FIELDS = ("test", "kind", "category", "section", "error_type")

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
            protected = (depth == 0 and k in PROTECTED_KEYS) or (
                child_path in _PROTECTED_PATH_PREFIXES
            )
            if protected:
                size += json_size(str(k)) + 1 + json_size(v)
                continue
            if depth == 0 and k in RISK_KEYS:
                # Only the list itself may be cut, and only as a last
                # resort: apply_budget picks these up separately.
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


def _risk_summary_entry(items: List[Any]) -> Dict[str, Any]:
    """Counts that survive when a risk list is shortened."""
    by_severity: Dict[str, int] = {}
    categories: Dict[str, int] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        sev = it.get("severity")
        if sev is not None:
            by_severity[str(sev)] = by_severity.get(str(sev), 0) + 1
        for field in _RISK_CATEGORY_FIELDS:
            cat = it.get(field)
            if cat is not None:
                categories[str(cat)] = categories.get(str(cat), 0) + 1
                break
    entry: Dict[str, Any] = {"total": len(items), "shown": len(items), "omitted": 0}
    if by_severity:
        entry["by_severity"] = by_severity
    if categories:
        ranked = sorted(categories.items(), key=lambda kv: (-kv[1], kv[0]))
        entry["categories"] = dict(ranked[:_MAX_RISK_CATEGORIES])
        if len(ranked) > _MAX_RISK_CATEGORIES:
            entry["categories_omitted"] = len(ranked) - _MAX_RISK_CATEGORIES
    return entry


def note_risk_omission(obj: Dict[str, Any], key: str, total: int) -> None:
    """Record that ``obj[key]`` already holds only part of ``total`` entries.

    For a producer that caps a risk list before the budget sees it (the
    server keeps the first distinct runtime warnings): the cap is reported
    the same way as a budget cut.
    """
    items = obj.get(key)
    if not isinstance(items, list) or total <= len(items):
        return
    entry = _risk_summary_entry(items)
    entry["total"] = int(total)
    entry["omitted"] = int(total) - len(items)
    entry["counts_cover"] = "shown"
    _mark_risk_incomplete(obj, key, entry)


def _mark_risk_incomplete(obj: Dict[str, Any], key: str, entry: Dict[str, Any]) -> None:
    summary = obj.get("risk_summary")
    if not isinstance(summary, dict):
        summary = {}
        obj["risk_summary"] = summary
    summary[key] = entry
    obj["risk_details_complete"] = False
    # Warnings and degradations belong to the call, not to the cached fit,
    # so a result handle cannot stand in for the cut entries.
    summary["full_details"] = (
        "repeat the call with a larger max_output_bytes (0 = no limit)"
    )


def _set_budget_ledger(
    obj: Dict[str, Any], status: str, max_bytes: int, oversized: List[Dict[str, Any]]
) -> None:
    """Write ``output_budget`` with an ``actual_bytes`` that counts itself."""
    ledger: Dict[str, Any] = {
        "status": status,
        "max_bytes": int(max_bytes),
        "actual_bytes": 0,
        "scope": BUDGET_SCOPE,
    }
    if oversized:
        ledger["oversized_fields"] = oversized
    obj["output_budget"] = ledger
    for _ in range(4):
        actual = json_size(obj)
        if ledger["actual_bytes"] == actual:
            break
        ledger["actual_bytes"] = actual


def apply_budget(
    obj: Dict[str, Any], max_bytes: Optional[int]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Shrink ``obj`` in place until it serialises to at most ``max_bytes``.

    The largest truncatable container is cut first, to the number of
    items that removes the current excess (never below one item); the
    loop repeats until the object fits or nothing is left to cut. Risk
    lists (:data:`RISK_KEYS`) are cut only once nothing else can be, and
    leave ``risk_summary`` + ``risk_details_complete: false`` behind.

    Returns ``(obj, truncated)`` where ``truncated`` lists
    ``{"path", "total", "shown"}`` per cut container (``total`` / ``shown``
    count items, or characters for strings). The same list is written to
    ``obj["truncated"]``, and ``obj["output_budget"]`` reports the outcome
    whenever the object was cut or still exceeds the budget; both are
    counted in the size, so ``status == "truncated"`` guarantees
    ``actual_bytes <= max_bytes``.
    """
    if not max_bytes or not isinstance(obj, dict):
        return obj, []
    if json_size(obj) <= max_bytes:
        return obj, []
    records: Dict[str, Dict[str, Any]] = {}
    fits = False
    for _ in range(500):
        # The ledgers are written before measuring, so the size is exact.
        if records:
            obj["truncated"] = list(records.values())
        _set_budget_ledger(obj, "truncated", max_bytes, [])
        cands: List[Tuple[int, str, Any, Any, Any]] = []
        total = _measure(obj, "", 0, cands, None, None)
        # Slack for the digits ``shown`` / ``actual_bytes`` may still gain.
        excess = total - max_bytes + (0 if records else 48)
        if total <= max_bytes:
            fits = True
            break
        cands = [c for c in cands if c[2] is not None and len(c[2]) > 1]
        risk = False
        if not cands:
            cands = [
                (json_size(obj[k]), f"/{k}", obj[k], obj, k)
                for k in RISK_KEYS
                if isinstance(obj.get(k), list) and len(obj[k]) > 1
            ]
            risk = True
        if not cands:
            break
        size, path, node, parent, key = max(cands, key=lambda c: c[0])
        n = len(node)
        if risk:
            summary = obj.get("risk_summary")
            if not isinstance(summary, dict) or key not in summary:
                _mark_risk_incomplete(obj, key, _risk_summary_entry(node))
                # The summary costs bytes too: re-measure before cutting.
                continue
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
        if risk:
            entry = obj["risk_summary"][key]
            entry["omitted"] = entry["total"] - shown
            entry["shown"] = shown
        rec = records.get(path)
        if rec is None:
            records[path] = {"path": path, "total": n, "shown": shown}
        else:
            rec["shown"] = shown
    if records:
        obj["truncated"] = list(records.values())
    if not fits:
        sizes = sorted(
            ((json_size(v), k) for k, v in obj.items() if k != "output_budget"),
            reverse=True,
        )
        oversized = [{"path": f"/{k}", "bytes": b} for b, k in sizes[:3]]
        _set_budget_ledger(obj, "unavoidable_overflow", max_bytes, oversized)
    else:
        _set_budget_ledger(obj, "truncated", max_bytes, [])
    return obj, list(records.values())


__all__ = [
    "BUDGET_SCOPE",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "MAX_OUTPUT_BYTES_ENV",
    "PROTECTED_KEYS",
    "RISK_KEYS",
    "apply_budget",
    "note_risk_omission",
    "json_size",
    "max_output_bytes",
    "scrub_nonfinite",
]
