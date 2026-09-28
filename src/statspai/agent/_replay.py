"""Python replay strings for agent tool calls.

Every estimator reached through the agent layer (curated tools and the
auto-registered registry tools) reports ``replay``: the ``sp.<fn>(...)``
call that reproduces the fit in a Python session, built from the
arguments that actually reached the estimator (arguments dropped as
unsupported are not in it). The dataset is written as ``data=data``; the
MCP layer appends a comment naming where ``data`` came from (the
``data_path``, ``data_id`` or inline table), so a human can re-run the
analysis outside the agent loop.
"""

from __future__ import annotations

import keyword
from typing import Any, Dict, Mapping, Optional


def _literal(value: Any) -> str:
    """Python literal for a JSON-like value; a placeholder otherwise."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_literal(v) for v in value)
        if isinstance(value, tuple):
            return f"({inner}{',' if len(value) == 1 else ''})"
        return f"[{inner}]"
    if isinstance(value, dict):
        inner = ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items())
        return "{" + inner + "}"
    return f"<{type(value).__name__}>"


def build_replay(
    fn_name: str,
    kwargs: Mapping[str, Any],
    *,
    result_id: Optional[str] = None,
) -> str:
    """Return ``sp.<fn_name>(data=data, key=value, ...)``.

    Parameters
    ----------
    fn_name : str
        Public ``statspai`` function name.
    kwargs : mapping
        Keyword arguments as passed to the function. ``data`` (any value)
        renders as ``data=data``; a fitted ``result`` injected from a
        handle renders as ``result=result_<result_id>``; keys starting
        with ``_`` (server-internal) are skipped.
    result_id : str, optional
        Handle the ``result`` argument came from.
    """
    parts = []
    if "data" in kwargs:
        parts.append("data=data")
    for key, value in kwargs.items():
        if key == "data" or str(key).startswith("_"):
            continue
        if not str(key).isidentifier() or keyword.iskeyword(str(key)):
            continue
        if key == "result" and result_id:
            parts.append(f"result=result_{result_id}")
            continue
        parts.append(f"{key}={_literal(value)}")
    return f"sp.{fn_name}({', '.join(parts)})"


def data_comment(
    *,
    data_path: Optional[str] = None,
    data_id: Optional[str] = None,
    inline_provenance: Optional[Dict[str, Any]] = None,
    data_columns: Optional[Any] = None,
    data_sample_n: Optional[Any] = None,
) -> str:
    """Comment naming the source of ``data`` in a replay string."""
    if data_id:
        src = f"data_id {data_id!r} (lineage in data_provenance)"
    elif data_path:
        src = f"data_path {data_path!r}"
    elif inline_provenance:
        sha = str(inline_provenance.get("sha256", ""))[:12]
        src = f"inline {inline_provenance.get('format', 'table')} (sha256 {sha}...)"
    else:
        return ""
    extras = []
    if data_columns:
        extras.append(f"columns={list(data_columns)!r}")
    if data_sample_n is not None:
        extras.append(f"sample_n={data_sample_n!r}, seed=0")
    tail = f" [{', '.join(extras)}]" if extras else ""
    return f"  # data = {src}{tail}"


__all__ = ["build_replay", "data_comment"]
