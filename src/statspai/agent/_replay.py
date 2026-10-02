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
import re
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


#: ``replay_completeness.level`` values, weakest first.
REPLAY_LEVELS = ("call_only", "session_replayable", "standalone")

_PLACEHOLDER = re.compile(r"=<[A-Za-z_][A-Za-z0-9_]*>")


def replay_completeness(
    replay: str,
    data_provenance: Optional[Mapping[str, Any]] = None,
    *,
    result_id: Optional[str] = None,
) -> Dict[str, Any]:
    """How far a ``replay`` string goes towards reproducing the call.

    Returns ``{"level", "needs"}``. ``level`` is

    * ``"standalone"`` -- a new Python process can re-run the call from
      the string, given what ``needs`` lists (a local file, identified by
      path and SHA-256, or nothing for a call that takes no data);
    * ``"session_replayable"`` -- it depends on a handle held by this
      server process (a ``data_id`` whose lineage is in
      ``data_provenance``, or a fitted ``result_id``);
    * ``"call_only"`` -- the string documents the call but cannot re-run
      it: an argument had no literal form, or the data was sent inline
      or fetched from a URL and only a hash (or nothing) was kept.

    The weakest component decides the level.
    """
    rank = len(REPLAY_LEVELS) - 1
    needs = []

    def _cap(level: str, need: str) -> None:
        nonlocal rank
        rank = min(rank, REPLAY_LEVELS.index(level))
        needs.append(need)

    if _PLACEHOLDER.search(replay):
        _cap("call_only", "argument values shown as <TypeName> have no literal form")
    if result_id and f"result_{result_id}" in replay:
        _cap("session_replayable", f"fitted result {result_id} from this session")
    if "data=data" in replay:
        prov = data_provenance or {}
        kind = prov.get("source_type")
        if kind == "handle":
            _cap(
                "session_replayable",
                f"dataset handle {prov.get('data_id')}; re-apply "
                "data_provenance.lineage to rebuild it in a new process",
            )
        elif kind == "local":
            sha = prov.get("sha256")
            needs.append(
                f"file {prov.get('source')}"
                + (f" (sha256 {sha})" if sha else " (not hashed)")
            )
        elif kind == "inline":
            _cap("call_only", "inline data: only its hash was recorded")
        elif kind == "remote":
            _cap("call_only", "remote data is not hashed; its content may change")
        else:
            _cap("call_only", "the source of `data` was not recorded")
    return {"level": REPLAY_LEVELS[rank], "needs": needs}


__all__ = ["REPLAY_LEVELS", "build_replay", "data_comment", "replay_completeness"]
