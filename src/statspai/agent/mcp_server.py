"""
Model Context Protocol (MCP) server for StatsPAI.

Exposes StatsPAI's estimator catalogue as MCP tools so any MCP-capable
client (Claude Desktop, Copilot CLI, Cursor, custom agents) can call
``sp.iv()``, ``sp.did()``, ``sp.causal()``, etc. directly from a
natural-language workflow.

The server speaks JSON-RPC 2.0 over stdio — the transport required by
the MCP spec (https://modelcontextprotocol.io/specification). It is
implemented in pure Python with no external dependencies so it can
ship inside the StatsPAI wheel.

Quick start
-----------
Launch from a shell::

    python -m statspai.agent.mcp_server

For Claude Desktop, add to ``claude_desktop_config.json``::

    {
      "mcpServers": {
        "statspai": {
          "command": "python",
          "args": ["-m", "statspai.agent.mcp_server"]
        }
      }
    }

Tool contract
-------------
Every tool takes a ``data_path`` argument — an absolute CSV path on
the local filesystem — plus whatever column-name arguments the
underlying StatsPAI function expects. The server loads the CSV, runs
the estimator, and returns the result as a JSON object.

Resources
---------
The server also exposes ``statspai://catalog`` — a resource enumerating
every registered estimator with its description and citation. Clients
can fetch this once during session setup to give the LLM structured
context about what's available.
"""

from __future__ import annotations

import json
import sys
import os
import traceback
from typing import Any, Dict, Iterable, List, Optional

from .tools import tool_manifest, execute_tool


MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "statspai"


# ═══════════════════════════════════════════════════════════════════════
#  Typed RPC errors → mapped to canonical JSON-RPC / MCP error codes
# ═══════════════════════════════════════════════════════════════════════
#
# JSON-RPC 2.0 reserves ``-32xxx`` codes; MCP 2024-11-05 names
# ``-32002`` for resource-not-found. Using untyped ValueError + a
# blanket ``-32000`` would force MCP clients to regex the message to
# decide whether to retry, prompt the user, or surface a friendly
# error — typing the exception keeps the protocol semantically rich.

class _RpcError(Exception):
    """Internal exception carrying an explicit JSON-RPC error code."""
    code: int = -32000  # generic server-defined error


class _InvalidParamsError(_RpcError):
    """``-32602`` per JSON-RPC 2.0 — caller-supplied params are wrong."""
    code = -32602


class _ResourceNotFoundError(_RpcError):
    """``-32002`` per MCP 2024-11-05 — URI does not resolve to a resource."""
    code = -32002


def _resolve_server_version() -> str:
    """Pull the server version from ``statspai.__version__``.

    Keeps the MCP server in lock-step with the package on every release
    — avoids the drift we hit when ``SERVER_VERSION`` was a hand-edited
    literal that fell behind the project version bump.
    """
    try:
        import statspai as _sp
        v = getattr(_sp, "__version__", None)
        if isinstance(v, str) and v:
            return v
    except Exception:  # pragma: no cover — statspai must import
        pass
    return "0.0.0"


SERVER_VERSION = _resolve_server_version()


# ═══════════════════════════════════════════════════════════════════════
#  JSON-RPC helpers
# ═══════════════════════════════════════════════════════════════════════

def _jsonrpc_result(request_id: Any, result: Any) -> str:
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }, default=_json_default)


def _jsonrpc_error(request_id: Any, code: int, message: str,
                   data: Any = None) -> str:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": err,
    }, default=_json_default)


def _json_default(o: Any) -> Any:
    """Best-effort JSON encoder for numpy / pandas / std-lib scalars.

    Covers every type we've actually seen leak out of estimator dicts:

    * numpy: ``integer`` / ``floating`` / ``bool_`` / ``complex_`` /
      ``datetime64`` / ``timedelta64`` / ``ndarray``
    * pandas: ``Series`` / ``DataFrame`` / ``Index`` / ``Timestamp`` /
      ``Timedelta`` / ``Categorical`` / ``Interval``
    * stdlib: ``set`` / ``frozenset`` / ``bytes`` / ``Decimal`` /
      ``Path`` / dataclasses / Enums

    A bare ``__dict__`` fallback is risky on heavy result objects (live
    DataFrames recursing into themselves), so it's reached last and only
    walks public attributes one level deep.
    """
    # NaN/Inf — JSON has no representation; emit ``None`` so json.dumps
    # without ``allow_nan=False`` doesn't silently round-trip 'NaN'.
    if isinstance(o, float):
        import math
        if math.isnan(o) or math.isinf(o):
            return None

    try:
        import numpy as _np
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            v = float(o)
            import math
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(o, _np.complexfloating):
            return {"real": float(o.real), "imag": float(o.imag)}
        if isinstance(o, _np.datetime64):
            # ns-precision ISO-8601 string; stable across pandas versions
            return str(o)
        if isinstance(o, _np.timedelta64):
            return str(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass

    try:
        import pandas as _pd
        if isinstance(o, _pd.DataFrame):
            return o.to_dict(orient="list")
        if isinstance(o, _pd.Series):
            return o.to_dict()
        if isinstance(o, _pd.Index):
            return o.tolist()
        if isinstance(o, _pd.Timestamp):
            return o.isoformat()
        if isinstance(o, _pd.Timedelta):
            return o.isoformat()
        if isinstance(o, _pd.Categorical):
            return list(o)
        if isinstance(o, _pd.Interval):
            return {"left": o.left, "right": o.right, "closed": o.closed}
    except ImportError:  # pragma: no cover
        pass

    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if isinstance(o, bytes):
        # Round-trippable; agents reading JSON shouldn't get garbled UTF-8
        import base64
        return {"__bytes_b64__": base64.b64encode(o).decode("ascii")}

    try:
        from decimal import Decimal
        if isinstance(o, Decimal):
            return float(o)
    except Exception:  # pragma: no cover
        pass

    try:
        from pathlib import PurePath
        if isinstance(o, PurePath):
            return str(o)
    except Exception:  # pragma: no cover
        pass

    try:
        from enum import Enum
        if isinstance(o, Enum):
            return o.value
    except Exception:  # pragma: no cover
        pass

    # dataclasses (without using asdict, which recurses and re-hits us)
    if hasattr(o, "__dataclass_fields__"):
        return {f: getattr(o, f, None) for f in o.__dataclass_fields__}

    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items()
                if not k.startswith("_")}
    return str(o)


# ═══════════════════════════════════════════════════════════════════════
#  Tool spec transformation: StatsPAI manifest → MCP tools/list spec
# ═══════════════════════════════════════════════════════════════════════

#: Reserved argument names the MCP server consumes itself before
#: dispatching to the estimator. ``data_path`` becomes a DataFrame;
#: ``detail`` controls the result-serialisation level (see
#: ``CausalResult.to_dict``). Each entry is the (single) source of
#: truth for both the schema injection in :func:`_build_mcp_tools`
#: and the argument stripping in :func:`_handle_tools_call`.
_RESERVED_ARG_NAMES = ("data_path", "detail")

#: Allowed values for ``detail`` (mirrors ``CausalResult.to_dict``).
_DETAIL_LEVELS = ("minimal", "standard", "agent")

#: Tools whose underlying StatsPAI function does NOT take a DataFrame
#: as input (they consume pre-computed statistics or string handles).
#: ``data_path`` is still injected into their schema as an OPTIONAL
#: convenience for clients that always send it, but it MUST NOT be
#: marked required — strict-schema MCP clients (e.g. Claude Desktop)
#: would otherwise refuse to dispatch the call without a CSV path that
#: the estimator never reads.
#:
#: This is the *manual override* set — names listed here are forced
#: dataless even if the registry says otherwise. The runtime also
#: auto-derives dataless tools from the registry (any spec without a
#: required ``data`` ParamSpec) via :func:`_dataless_tool_names`, so the
#: hand-curated list only carries entries the registry can't reach
#: (e.g. tools backed by an auto-generated stub or whose dataframe
#: dependency was added after the schema was frozen).
_DATALESS_OVERRIDES = frozenset({"honest_did", "sensitivity",
                                  "audit_result", "brief_result",
                                  "sensitivity_from_result",
                                  "honest_did_from_result",
                                  "plot_from_result",
                                  "bibtex",
                                  "from_stata", "from_r"})


#: Backwards-compatible alias for the old hand-curated set. New code
#: should call :func:`_dataless_tool_names` to get the registry-derived
#: union; tests / external callers that imported this constant continue
#: to see a stable surface.
_DATALESS_TOOLS = _DATALESS_OVERRIDES


def _dataless_tool_names() -> "frozenset[str]":
    """Names of tools that take no DataFrame.

    Auto-derived from the registry: any registered function without a
    required ``data`` parameter is dataless. Falls back to
    :data:`_DATALESS_OVERRIDES` alone if registry introspection fails.
    """
    derived: "set[str]" = set(_DATALESS_OVERRIDES)
    try:
        from ..registry import _REGISTRY, _ensure_full_registry
        _ensure_full_registry()
        for name, spec in _REGISTRY.items():
            params = getattr(spec, "params", None) or []
            has_required_data = any(
                p.name == "data" and p.required for p in params
            )
            if not has_required_data:
                # No required `data` param → safe to mark dataless. Tools
                # that take an OPTIONAL data still get data_path injected
                # for client convenience but won't be required.
                derived.add(name)
    except Exception:
        pass
    return frozenset(derived)


def _build_mcp_tools() -> List[Dict[str, Any]]:
    """Convert the StatsPAI agent-tool manifest into MCP tool specs.

    We inject server-handled arguments into every tool's schema so the
    LLM can supply them via the standard ``tools/call`` arguments
    object:

    * ``data_path`` (required for data-bound tools) — absolute path or
      ``s3://`` / ``gs://`` / ``https://`` URL to a CSV / Parquet / Stata
      / Feather / JSON file the server loads into a DataFrame.
    * ``data_columns`` (optional) — column projection for Parquet /
      Stata reads to skip loading unused columns.
    * ``data_sample_n`` (optional) — random subsample size for fast
      iteration on huge files.
    * ``result_id`` (optional) — pointer to a previously-fitted result
      cached by the server. When supplied, it can replace ``data_path``
      for tools that operate on a fitted result (audit, sensitivity,
      brief, honest_did from result, …).
    * ``as_handle`` (optional) — when ``true``, the server caches the
      fitted result and returns ``result_id`` / ``result_uri`` so the
      next call can reference it without re-running the estimator.
    * ``detail`` (optional, default ``"agent"``) — payload depth,
      forwarded to ``result.to_dict(detail=...)``.
    """
    manifest = tool_manifest()
    dataless = _dataless_tool_names()
    out: List[Dict[str, Any]] = []
    for t in manifest:
        schema = dict(t.get("input_schema") or {})
        props = dict(schema.get("properties") or {})
        required = list(schema.get("required") or [])
        if "data_path" not in props:
            props["data_path"] = {
                "type": "string",
                "description": (
                    "Absolute path or URL to a data file. Supported: "
                    ".csv / .tsv / .txt (delimited), .parquet / .pq, "
                    ".feather / .arrow, .xlsx / .xls, .dta (Stata), "
                    ".json / .jsonl. Schemes: file://, s3://, gs://, "
                    "https://."
                ),
            }
            # Mark required ONLY for tools whose underlying function
            # actually takes a DataFrame; dataless tools leave
            # ``data_path`` optional so strict-schema MCP clients don't
            # refuse to dispatch them.
            if t["name"] not in dataless:
                required.append("data_path")
        if "data_columns" not in props:
            props["data_columns"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional column projection. Parquet/Feather/Stata "
                    "loaders honour this for fast partial reads."
                ),
            }
        if "data_sample_n" not in props:
            props["data_sample_n"] = {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional uniform random subsample size "
                    "(seed=0, deterministic) — useful on huge panels."
                ),
            }
        if "result_id" not in props:
            props["result_id"] = {
                "type": "string",
                "description": (
                    "Optional handle to a previously-fitted result "
                    "(returned by an earlier call when as_handle=true). "
                    "Tools that operate on a fitted object accept this "
                    "in place of re-supplying data_path + columns."
                ),
            }
        if "as_handle" not in props:
            props["as_handle"] = {
                "type": "boolean",
                "default": False,
                "description": (
                    "If true, cache the fitted result on the server "
                    "and return result_id + result_uri alongside the "
                    "JSON payload so a subsequent tools/call can chain "
                    "without re-running."
                ),
            }
        # Unconditional overwrite: ``detail`` is a server-handled control
        # arg (forwarded to ``result.to_dict(detail=...)``) — if a
        # registry estimator happens to have its own ``detail`` parameter
        # (e.g. ``oaxaca`` uses it as a bool), we hide it so the manifest
        # schema is uniform across tools. Reaching that estimator's
        # ``detail`` requires the direct Python API.
        if True:
            props["detail"] = {
                "type": "string",
                "enum": ["minimal", "standard", "agent"],
                "default": "agent",
                "description": (
                    "Payload depth: 'minimal' (~150 tokens) for "
                    "sub-step calls where only the point estimate is "
                    "needed; 'standard' (~1K tokens) for diagnostics "
                    "+ coefficient table; 'agent' (~2K tokens, "
                    "default) adds violations / next_steps / "
                    "suggested_functions so the LLM can plan its "
                    "next call without another round-trip."
                ),
            }
        schema["type"] = schema.get("type", "object")
        schema["properties"] = props
        schema["required"] = sorted(set(required))
        out.append({
            "name": t["name"],
            "description": t["description"],
            "inputSchema": schema,
        })
    return out


#: Default max file size (bytes) the server will load. A misconfigured
#: client pointing at a 50GB parquet will OOM the host otherwise.
#: Override via ``STATSPAI_MCP_MAX_DATA_BYTES`` (e.g. ``5_000_000_000``);
#: set to ``0`` to disable the check.
_DEFAULT_MAX_DATA_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _max_data_bytes() -> int:
    raw = os.environ.get("STATSPAI_MCP_MAX_DATA_BYTES")
    if raw is None:
        return _DEFAULT_MAX_DATA_BYTES
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_DATA_BYTES


def _is_remote_url(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "https://", "http://",
                             "file://"))


def _load_dataframe(path: str,
                    columns: Optional[List[str]] = None,
                    sample_n: Optional[int] = None):
    """Load a DataFrame from a local path or remote URL.

    Parameters
    ----------
    path : str
        Absolute filesystem path or one of: ``file://``, ``s3://``,
        ``gs://``, ``https://``, ``http://``.
    columns : list of str, optional
        Column projection. Honoured by parquet / feather / stata
        readers; for CSV we read all columns then sub-select (read_csv's
        ``usecols`` would also work but mismatched names raise; we want
        the server to be permissive and let the estimator surface
        column-name errors with rich remediation).
    sample_n : int, optional
        Uniform random subsample size (seed=0, deterministic).
        Applied AFTER the projection.

    Notes
    -----
    Caches the materialised frame keyed by ``(path, mtime, columns)``
    so repeated tool calls on the same file are O(1) after the first
    load. The cache is bounded — see ``_LOAD_CACHE_SIZE``.
    """
    if _is_remote_url(path):
        # Remote — defer all guard rails to the underlying loader; we
        # can't ``os.path.exists`` an s3 URL, and pandas/storage_options
        # error messages are rich enough.
        df = _load_remote(path, columns=columns)
    else:
        if not os.path.isabs(path):
            raise ValueError(
                f"data_path must be absolute or a URL, got {path!r}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")
        size = os.path.getsize(path)
        cap = _max_data_bytes()
        if cap and size > cap:
            raise ValueError(
                f"data file is {size:,} bytes; exceeds "
                f"STATSPAI_MCP_MAX_DATA_BYTES={cap:,}. "
                f"Pass data_sample_n=<N> for a random subsample, or "
                f"raise the limit with the env var."
            )
        mtime = os.path.getmtime(path)
        df = _load_local_cached(path, mtime, tuple(columns or ()))

    if columns:
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df[keep]
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=int(sample_n), random_state=0).reset_index(drop=True)
    return df


import functools as _functools


@_functools.lru_cache(maxsize=8)
def _load_local_cached(path: str, mtime: float,
                       columns_key: tuple):  # noqa: ARG001 — mtime invalidates
    """LRU-cached local loader. ``mtime`` busts the cache on file edits."""
    import pandas as pd
    lower = path.lower()
    cols = list(columns_key) or None
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep, usecols=cols)
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(path, columns=cols)
    if lower.endswith((".feather", ".arrow")):
        return pd.read_feather(path, columns=cols)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, usecols=cols)
    if lower.endswith(".dta"):
        # Stata native — alignment with Stata is StatsPAI's tagline,
        # so being able to read .dta is non-negotiable.
        return pd.read_stata(path, columns=cols)
    if lower.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
        return df[cols] if cols else df
    if lower.endswith(".json"):
        df = pd.read_json(path)
        return df[cols] if cols else df
    raise ValueError(
        f"Unsupported file extension: {path!r}. Supported: "
        ".csv/.tsv/.txt/.parquet/.pq/.feather/.arrow/.xlsx/.xls/.dta/"
        ".json/.jsonl"
    )


def _load_remote(url: str, columns: Optional[List[str]] = None):
    """Load a DataFrame from a remote URL via pandas storage backends.

    Pandas dispatches s3:// / gs:// / https:// to fsspec. Authentication
    is configured by the host environment (e.g. AWS credentials chain);
    we don't smuggle secrets through the MCP layer.
    """
    import pandas as pd
    lower = url.split("?", 1)[0].lower()
    cols = list(columns) if columns else None
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(url, sep=sep, usecols=cols)
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(url, columns=cols)
    if lower.endswith((".feather", ".arrow")):
        return pd.read_feather(url, columns=cols)
    if lower.endswith(".dta"):
        return pd.read_stata(url, columns=cols)
    if lower.endswith(".jsonl"):
        df = pd.read_json(url, lines=True)
        return df[cols] if cols else df
    if lower.endswith(".json"):
        df = pd.read_json(url)
        return df[cols] if cols else df
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(url, usecols=cols)
    raise ValueError(
        f"Unsupported remote extension in {url!r}. "
        f"See _load_dataframe docs for supported formats."
    )


# ═══════════════════════════════════════════════════════════════════════
#  Resources
# ═══════════════════════════════════════════════════════════════════════
#
# Three top-level URIs are exposed. ``statspai://catalog`` and
# ``statspai://functions`` are listable in ``resources/list``; the
# per-function ``statspai://function/<name>`` URIs are not enumerated
# (would be 100+ items in client UIs) but are readable on demand and
# documented in the catalog so agents know the pattern.
#
#   statspai://catalog              — Markdown summary of every tool
#   statspai://functions            — JSON array: name + 1-line description
#   statspai://function/<name>      — JSON: full agent_card for one tool
#                                     (description, input_schema,
#                                      assumptions, failure_modes,
#                                      alternatives, typical_n_min, example)

_FUNCTION_URI_PREFIX = "statspai://function/"
_RESULT_URI_PREFIX = "statspai://result/"


def _catalog_text() -> str:
    """Return a Markdown catalog of every StatsPAI tool."""
    manifest = tool_manifest()
    lines = [
        "# StatsPAI tool catalog",
        "",
        f"Version: {SERVER_VERSION}. {len(manifest)} tools registered.",
        "",
        "**Per-function detail**: read "
        f"`{_FUNCTION_URI_PREFIX}<name>` for the full agent card "
        "(assumptions, failure modes, alternatives, typical_n_min, "
        "example) of any tool listed below.",
        "",
        "**Machine-readable index**: read `statspai://functions` for a "
        "JSON array of `{name, description}` entries.",
        "",
    ]
    for t in manifest:
        lines.append(f"## {t['name']}")
        lines.append("")
        desc = t.get("description", "").strip()
        if desc:
            lines.append(desc)
            lines.append("")
    return "\n".join(lines)


def _functions_index() -> List[Dict[str, str]]:
    """Return a JSON-ready ``[{name, description}, …]`` list."""
    return [
        {"name": t["name"],
         "description": (t.get("description") or "").strip()}
        for t in tool_manifest()
    ]


def _function_detail(name: str) -> Optional[Dict[str, Any]]:
    """Return the rich agent card for one tool, or ``None`` if unknown.

    Prefers ``statspai.registry.agent_card`` (full card with
    assumptions / failure_modes / alternatives / typical_n_min) and
    falls back to the manifest entry for tools that exist in the
    auto-generated layer but lack a hand-curated registry spec.
    """
    # Try the registry first — it has the agent-native metadata.
    try:
        from ..registry import agent_card as _agent_card
        card = _agent_card(name)
        if card:
            return card
    except Exception:
        pass

    # Fallback: synthesise from the merged manifest so any registered
    # tool — even auto-generated ones without a curated spec — still
    # resolves to *something* readable.
    for t in tool_manifest():
        if t["name"] == name:
            return {
                "name": t["name"],
                "description": (t.get("description") or "").strip(),
                "signature": {
                    "name": t["name"],
                    "description": (t.get("description") or "").strip(),
                    "parameters": t.get("input_schema") or {},
                },
                "pre_conditions": [],
                "assumptions": [],
                "failure_modes": [],
                "alternatives": [],
                "typical_n_min": None,
                "reference": "",
                "example": "",
            }
    return None


# ═══════════════════════════════════════════════════════════════════════
#  JSON-RPC handlers
# ═══════════════════════════════════════════════════════════════════════

_SESSION_INSTRUCTIONS = (
    "StatsPAI MCP — agent-native causal inference & econometrics.\n\n"
    "Recommended workflow:\n"
    "  1. detect_design (or pass design= explicitly) to identify the "
    "study shape.\n"
    "  2. preflight + recommend on the data to surface design problems "
    "and pick an estimator.\n"
    "  3. Fit with as_handle=true so you get a result_id you can chain "
    "into downstream tools.\n"
    "  4. audit_result(result_id=...) to enumerate missing robustness "
    "checks; for each, call the suggest_function it emits.\n"
    "  5. honest_did_from_result / sensitivity_from_result for "
    "design-specific sensitivity (no need to ferry betas / sigma).\n"
    "  6. bibtex(keys=[...]) for verified citations — never invent "
    "references; paper.bib is the single source of truth.\n\n"
    "Token economy: pass detail='minimal' on cheap sub-step calls; "
    "default 'agent' carries violations + next_steps. Inline plots "
    "arrive as image content blocks for vision-capable clients."
)


def _handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
        "instructions": _SESSION_INSTRUCTIONS,
    }


def _handle_tools_list(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"tools": _build_mcp_tools()}


def _handle_tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    arguments = dict(params.get("arguments") or {})
    if not isinstance(name, str):
        raise _InvalidParamsError(
            "`name` is required and must be a string")

    # Server-handled args are stripped before estimator dispatch — the
    # estimator's signature has no ``data_path`` / ``detail`` etc. and
    # would crash with a "got an unexpected keyword argument" error.
    data_path = arguments.pop("data_path", None)
    data_columns = arguments.pop("data_columns", None) or None
    data_sample_n = arguments.pop("data_sample_n", None)
    result_id = arguments.pop("result_id", None)
    as_handle = bool(arguments.pop("as_handle", False))

    df = None
    if data_path:
        try:
            df = _load_dataframe(data_path,
                                  columns=data_columns,
                                  sample_n=data_sample_n)
        except (FileNotFoundError, ValueError) as e:
            # Surface as -32602 rather than a generic -32000 — a
            # bad/missing path is a caller-supplied params problem.
            raise _InvalidParamsError(str(e))

    detail = arguments.pop("detail", "agent")
    if detail not in _DETAIL_LEVELS:
        raise _InvalidParamsError(
            "detail must be one of "
            f"{', '.join(repr(v) for v in _DETAIL_LEVELS)}; "
            f"got {detail!r}"
        )

    result = execute_tool(name, arguments,
                           data=df,
                           detail=detail,
                           result_id=result_id,
                           as_handle=as_handle)
    text = json.dumps(result, indent=2, default=_json_default)

    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

    # Image content: estimators can attach a PNG plot under ``_plot_png``
    # for the MCP layer to surface as an image content block. Claude
    # vision (and any MCP client supporting image content) will render
    # it inline; the bytes are stripped from the JSON payload above.
    plot_bytes = None
    if isinstance(result, dict):
        plot_bytes = result.get("_plot_png")
    if isinstance(plot_bytes, (bytes, bytearray)):
        import base64
        content.append({
            "type": "image",
            "data": base64.b64encode(plot_bytes).decode("ascii"),
            "mimeType": "image/png",
        })
        # Re-serialise the JSON payload without the binary blob so the
        # text content stays readable.
        clean = {k: v for k, v in result.items() if k != "_plot_png"}
        content[0]["text"] = json.dumps(clean, indent=2,
                                          default=_json_default)

    return {
        "content": content,
        "isError": bool(isinstance(result, dict) and result.get("error")),
    }


def _handle_resources_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """Enumerate the top-level resources only.

    Per-function URIs (``statspai://function/<name>``) are intentionally
    *not* listed — there are 100+ of them and putting each in a client
    UI is noise. The catalog explicitly documents the pattern, and
    ``resources/read`` accepts any valid name on demand.
    """
    return {
        "resources": [
            {
                "uri": "statspai://catalog",
                "name": "StatsPAI estimator catalog",
                "mimeType": "text/markdown",
                "description": "Markdown list of every registered "
                               "StatsPAI estimator with its description "
                               "and a pointer to the per-function "
                               "agent-card URI pattern.",
            },
            {
                "uri": "statspai://functions",
                "name": "StatsPAI tool index (machine-readable)",
                "mimeType": "application/json",
                "description": "JSON array of {name, description} "
                               "entries. Read this once during session "
                               "setup to enumerate available tools.",
            },
        ],
    }


def _handle_resources_read(params: Dict[str, Any]) -> Dict[str, Any]:
    uri = params.get("uri")
    if not isinstance(uri, str):
        raise _InvalidParamsError(
            f"`uri` must be a string; got {uri!r}")

    if uri == "statspai://catalog":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": _catalog_text(),
                },
            ],
        }
    if uri == "statspai://functions":
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(_functions_index(),
                                       default=_json_default),
                },
            ],
        }
    if uri.startswith(_FUNCTION_URI_PREFIX):
        name = uri[len(_FUNCTION_URI_PREFIX):]
        if not name:
            raise _InvalidParamsError(
                f"Function name is empty in URI {uri!r}; "
                f"expected {_FUNCTION_URI_PREFIX}<name>")
        # Embedded slashes are not part of the {name} template — surface
        # the malformed-URI condition as -32602 (invalid params), not
        # -32002 (resource not found), so clients don't auto-retry with
        # a "did you mean" prompt.
        if "/" in name:
            raise _InvalidParamsError(
                f"Function name in URI {uri!r} must not contain '/'; "
                f"the URI template is {_FUNCTION_URI_PREFIX}{{name}}.")
        # Named ``card`` (not ``detail``) so reading this function
        # in isolation doesn't suggest a connection to the
        # serialisation-level ``detail`` parameter threaded elsewhere
        # through the MCP layer.
        card = _function_detail(name)
        if card is None:
            raise _ResourceNotFoundError(
                f"Unknown StatsPAI tool: {name!r}. "
                f"Read statspai://functions for the full index.")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(card, default=_json_default),
                },
            ],
        }

    if uri.startswith(_RESULT_URI_PREFIX):
        rid = uri[len(_RESULT_URI_PREFIX):]
        if not rid or "/" in rid:
            raise _InvalidParamsError(
                f"Result handle in URI {uri!r} is empty or malformed; "
                f"expected {_RESULT_URI_PREFIX}<id>.")
        from ._result_cache import RESULT_CACHE
        entry = RESULT_CACHE.get_entry(rid)
        if entry is None:
            raise _ResourceNotFoundError(
                f"Result {rid!r} not in server cache. LRU cache evicts "
                f"oldest entries; re-fit with as_handle=true for a "
                f"fresh handle.")
        # Render the result the same way an agent would have seen it
        # at fit time: registry-style ``to_dict(detail='agent')`` if
        # available, else a structural summary.
        from .tools import _default_serializer
        try:
            payload = _default_serializer(entry.obj, detail="agent")
        except Exception:  # pragma: no cover — fallback for odd objects
            payload = {"result_class": type(entry.obj).__name__}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        payload["provenance"] = entry.to_metadata()
        payload["result_id"] = rid
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(payload, default=_json_default),
                },
            ],
        }

    raise _ResourceNotFoundError(f"Unknown resource: {uri!r}")


# ═══════════════════════════════════════════════════════════════════════
#  Prompts: canned workflow templates
# ═══════════════════════════════════════════════════════════════════════
#
# MCP clients (Claude Desktop, Cursor) surface ``prompts/list`` entries
# in their UI as one-click "use this prompt" buttons. We ship a small
# set of curated workflow templates so users can spin up a typical
# StatsPAI agent loop without writing the prompt from scratch.
#
# Per spec:
# - ``prompts/list`` returns a list of {name, description, arguments[]}
# - ``prompts/get`` takes {name, arguments} and returns
#   {description, messages: [{role, content}]}

_PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "audit_did_result",
        "description": ("Run a DID estimator on a CSV, surface the "
                         "estimate, and walk through every "
                         "reviewer-checklist gap. Uses pipeline_did "
                         "to consolidate preflight + estimate + audit "
                         "+ honest-DID + Bacon into one call."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True, "description": "Outcome column."},
            {"name": "treat", "required": True,
             "description": "Binary 0/1 treatment indicator."},
            {"name": "time", "required": True, "description": "Time column."},
        ],
        "_template": (
            "Call `pipeline_did` with data_path={data_path}, y={y}, "
            "treat={treat}, time={time}. The pipeline returns a "
            "markdown narrative with the canonical reviewer-grade "
            "DID workflow already executed (preflight, estimator, "
            "audit, honest-DID, Bacon decomposition, brief). Quote "
            "the narrative verbatim; for any high-importance check "
            "the audit flagged as missing, dispatch the corresponding "
            "entry in `next_calls`. End with a `bibtex` lookup of the "
            "keys in `citations.keys` so the user gets verified "
            "references."
        ),
    },
    {
        "name": "audit_iv_result",
        "description": ("End-to-end IV workflow: 2SLS + first-stage F + "
                         "Anderson-Rubin CI + e-value sensitivity, all "
                         "wrapped in pipeline_iv."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "formula", "required": True,
             "description": "'y ~ x + (d ~ z)' Wilkinson-style."},
        ],
        "_template": (
            "Call `pipeline_iv` with data_path={data_path}, "
            "formula='{formula}'. Read `effective_F` from the "
            "response: < 10 means the Staiger-Stock weak-IV threshold "
            "is breached and you should foreground the "
            "Anderson-Rubin CI in your reply (it is in the "
            "`anderson_rubin` field) instead of the 2SLS point "
            "estimate. Cite via `bibtex(keys=...)` from the "
            "`citations.keys` list."
        ),
    },
    {
        "name": "audit_rd_result",
        "description": ("End-to-end RD workflow: rdrobust + rdplot "
                         "(image content) + density test + bandwidth "
                         "sensitivity via pipeline_rd."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True,
             "description": "Outcome column."},
            {"name": "x", "required": True,
             "description": "Running variable column."},
            {"name": "c", "required": False,
             "description": "Cutoff value (default 0)."},
        ],
        "_template": (
            "Call `pipeline_rd` with data_path={data_path}, y={y}, "
            "x={x}, c={c}. Use the `rdplot` image content block (PNG) "
            "to anchor your reply visually. If the McCrary-style "
            "density test rejects (`rddensity` p < 0.05) flag "
            "manipulation; recommend `rdplacebo` and `rdrbounds` "
            "(emit them via `next_calls`)."
        ),
    },
    {
        "name": "design_then_estimate",
        "description": ("Given an unfamiliar CSV, auto-detect the "
                         "study design, recommend an estimator, run "
                         "it with diagnostics."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "outcome", "required": True,
             "description": "Outcome column."},
            {"name": "treatment", "required": False,
             "description": "Treatment column (optional)."},
        ],
        "_template": (
            "1. Call `detect_design` with data_path={data_path}.\n"
            "2. Call `recommend` with y={outcome} (and "
            "treatment={treatment} when supplied). Read the top "
            "recommendation's `reasoning`.\n"
            "3. If the recommendation is DID/CS, call `pipeline_did`. "
            "If RD, call `pipeline_rd`. If IV, call `pipeline_iv`. "
            "Otherwise, call the recommended estimator with "
            "as_handle=true and follow up with `audit_result`.\n"
            "4. Quote the resulting `narrative`; emit the first "
            "two entries of `next_calls` for the user to consider."
        ),
    },
    {
        "name": "robustness_followup",
        "description": ("Take an existing fitted result handle and "
                         "run all high-importance follow-up "
                         "sensitivities the audit identifies as "
                         "missing."),
        "arguments": [
            {"name": "result_id", "required": True,
             "description": ("Handle from an earlier estimator call "
                              "(as_handle=true).")},
        ],
        "_template": (
            "1. Call `audit_result` with result_id={result_id}; read "
            "`items` (or `checks`) and collect every entry with "
            "status='missing' AND importance in {{'high', 'critical'}}.\n"
            "2. For each, dispatch the `suggest_function` it names. "
            "If the function takes a fitted result, pass "
            "result_id={result_id}; otherwise re-load the data via "
            "data_path.\n"
            "3. For each follow-up result, call `brief_result` and "
            "report whether the new estimate overturns the original "
            "conclusion (sign change / CI exclusion of zero)."
        ),
    },
    {
        "name": "paper_render",
        "description": ("Compose a paper-style memo from a fitted "
                         "result handle: estimate, diagnostics, "
                         "robustness, BibTeX. The output is a "
                         "ready-to-paste markdown section."),
        "arguments": [
            {"name": "result_id", "required": True,
             "description": ("Handle to a fitted result (returned by an "
                              "earlier estimator call with as_handle=true).")},
        ],
        "_template": (
            "Given result_id={result_id}:\n"
            "1. Call `brief_result` for a one-paragraph summary.\n"
            "2. Call `audit_result`; pull the audit's items with "
            "status='present' (the diagnostics that DID run) into a "
            "bulleted list.\n"
            "3. Call `plot_from_result` (auto-detects the right plot) "
            "and embed the resulting image.\n"
            "4. Call `bibtex(keys=...)` on the citation keys returned "
            "earlier; include the BibTeX bodies in a final "
            "`### References` section.\n"
            "5. Format as: '## Estimate' / '## Diagnostics' / "
            "'## Robustness' / '## Figure' / '## References'."
        ),
    },
    {
        "name": "compare_methods",
        "description": ("Run two or more estimators on the same data "
                         "and compare conclusions side by side."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True,
             "description": "Outcome column."},
            {"name": "treat", "required": True,
             "description": "Binary treatment indicator."},
            {"name": "time", "required": False,
             "description": "Time column for panel methods."},
        ],
        "_template": (
            "Run all three: `did`, `callaway_santanna` (if cohort/id "
            "available), `did_imputation`. Use as_handle=true for "
            "each so you collect three result_ids. Then call "
            "`brief_result` on each, and report a markdown table "
            "with rows = method, columns = (estimate, 95% CI, "
            "violations flagged). Highlight any sign disagreement."
        ),
    },
    {
        "name": "policy_evaluation",
        "description": ("Causal-forest-driven policy evaluation: "
                         "fit causal_forest, summarise CATE, evaluate "
                         "a candidate policy."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "formula", "required": True,
             "description": "'y ~ d | x1 + x2 + ...' (treatment | covariates)"},
        ],
        "_template": (
            "1. Call `causal_forest` with formula='{formula}' and "
            "as_handle=true.\n"
            "2. Call `cate_summary` with the result_id; report ATE + "
            "the CATE quantiles.\n"
            "3. Call `blp_test` to test whether heterogeneity is "
            "real, and `calibration_test` to check predictive quality.\n"
            "4. Call `policy_value` to estimate the value of treating "
            "everyone with positive predicted CATE."
        ),
    },
    {
        "name": "synth_full",
        "description": ("End-to-end Synthetic Control workflow: synth "
                         "fit + placebo + synthdid + permutation."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "outcome", "required": True,
             "description": "Outcome column."},
            {"name": "unit", "required": True,
             "description": "Unit identifier column."},
            {"name": "time", "required": True,
             "description": "Time column."},
            {"name": "treated_unit", "required": True,
             "description": "Identifier of the treated unit."},
            {"name": "treatment_time", "required": True,
             "description": "First post-treatment period."},
        ],
        "_template": (
            "1. Call `synth` with the canonical args; as_handle=true.\n"
            "2. Call `synthdid_estimate` for the synthetic-DID "
            "alternative — the two estimates should bracket the truth.\n"
            "3. Call `synthdid_placebo` for in-space placebo "
            "inference.\n"
            "4. Call `plot_from_result` (kind='synth_gap') to "
            "visualise the treated-vs-synthetic series."
        ),
    },
    {
        "name": "decompose_inequality",
        "description": ("RIF / FFL / Oaxaca-Blinder decomposition of "
                         "an outcome gap."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True,
             "description": "Outcome column."},
            {"name": "group", "required": True,
             "description": "Binary group indicator (e.g. gender, race)."},
            {"name": "covariates", "required": False,
             "description": "Comma-separated covariate columns."},
        ],
        "_template": (
            "Call `decompose` with method='oaxaca' (or method='rif' "
            "for distributional decomposition). Report explained vs "
            "unexplained share. If the user mentions wage gap, also "
            "run method='ffl' for the Firpo-Fortin-Lemieux variant."
        ),
    },
]


def _handle_prompts_list(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompts": [
            {
                "name": p["name"],
                "description": p["description"],
                "arguments": p["arguments"],
            }
            for p in _PROMPTS
        ],
    }


def _handle_prompts_get(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    if not isinstance(name, str):
        raise _InvalidParamsError("`name` is required and must be a string")
    spec = next((p for p in _PROMPTS if p["name"] == name), None)
    if spec is None:
        raise _ResourceNotFoundError(
            f"Unknown prompt: {name!r}. Read prompts/list for the "
            "available templates."
        )
    args = dict(params.get("arguments") or {})
    # Validate required arguments (omit MCP would otherwise leave the
    # template with literal ``{x}`` placeholders).
    missing = [
        a["name"] for a in spec["arguments"]
        if a.get("required") and a["name"] not in args
    ]
    if missing:
        raise _InvalidParamsError(
            f"prompt {name!r} missing required arguments: {missing}"
        )
    # Fill the template safely. ``str.format_map`` is single-pass —
    # it scans the *template* once for placeholders and substitutes
    # values verbatim without re-parsing the substituted text. So a
    # user value containing a literal ``{y}`` is preserved as-is in
    # the output (verified by ``test_get_with_brace_in_user_value...``).
    # ``_SafeDict`` keeps unknown placeholders literal so missing
    # required-arg bugs surface instead of being silently dropped.
    template = spec["_template"]
    try:
        rendered = template.format_map(_SafeDict(args))
    except Exception as e:
        raise _InvalidParamsError(
            f"Failed to render prompt {name!r}: "
            f"{type(e).__name__}: {e}"
        )
    return {
        "description": spec["description"],
        "messages": [
            {"role": "user",
             "content": {"type": "text", "text": rendered}},
        ],
    }


class _SafeDict(dict):
    """Format-map helper that leaves unknown placeholders literal."""
    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


def _handle_resources_templates_list(
        params: Dict[str, Any]) -> Dict[str, Any]:
    """Expose the parameterised ``statspai://function/{name}`` URI.

    Per MCP 2024-11-05, ``resources/templates/list`` is the protocol-
    level mechanism for parameterised resources. Clients that do
    autocomplete on resource URIs use this; the static ``resources/list``
    entries above don't enumerate per-function URIs (would be 100+
    items in client UIs) so a template is the right vehicle.
    """
    return {
        "resourceTemplates": [
            {
                "uriTemplate": _FUNCTION_URI_PREFIX + "{name}",
                "name": "StatsPAI function agent card",
                "mimeType": "application/json",
                "description": (
                    "Agent-native detail card for one tool: "
                    "description, JSON-schema signature, identifying "
                    "assumptions, common failure modes with recovery "
                    "hints, ranked alternatives, typical_n_min, and "
                    "an example call. Read "
                    "statspai://functions for the list of valid "
                    "{name} values."
                ),
            },
            {
                "uriTemplate": _RESULT_URI_PREFIX + "{id}",
                "name": "StatsPAI fitted-result handle",
                "mimeType": "application/json",
                "description": (
                    "Read a server-cached fitted result by id. The id "
                    "is returned by any tools/call invoked with "
                    "as_handle=true. Body shape mirrors the original "
                    "tool output (estimate / SE / CI / diagnostics) "
                    "plus a provenance block tagging the tool + args "
                    "that produced it. Cache is LRU; missing handles "
                    "raise -32002 (resource not found) — re-fit with "
                    "as_handle=true to refresh."
                ),
            },
        ],
    }


_METHODS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/templates/list": _handle_resources_templates_list,
    "resources/read": _handle_resources_read,
    "prompts/list": _handle_prompts_list,
    "prompts/get": _handle_prompts_get,
}


def handle_request(line: str) -> Optional[str]:
    """Process a single JSON-RPC request line; return the response line.

    Returns ``None`` for notifications — both the JSON-RPC 2.0 form
    (``id`` field entirely absent) and the MCP convention of any
    method whose name starts with ``"notifications/"`` (e.g.
    ``notifications/initialized`` sent by Claude Desktop / Cursor
    immediately after the handshake). The MCP spec mandates servers
    MUST NOT respond to those.
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        return _jsonrpc_error(None, -32700, f"Parse error: {e}")

    request_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    # JSON-RPC 2.0: a notification has no ``id`` field at all.
    if request_id is None and "id" not in msg:
        return None
    # MCP convention: ``notifications/<x>`` is a notification regardless
    # of whether the client erroneously included an ``id``. Silently
    # drop it instead of replying with -32601, which would generate
    # protocol noise on every session.
    if isinstance(method, str) and method.startswith("notifications/"):
        return None

    handler = _METHODS.get(method)
    if handler is None:
        return _jsonrpc_error(
            request_id, -32601, f"Method not found: {method!r}")

    try:
        result = handler(params)
    except _RpcError as exc:
        # Typed error → preserve the canonical JSON-RPC / MCP code
        # (``-32602`` invalid params, ``-32002`` resource not found,
        # ``-32000`` generic). No traceback for these — they're
        # expected / actionable on the client side.
        return _jsonrpc_error(request_id, exc.code, str(exc))
    except Exception as exc:
        # Tracebacks expose internal paths and class names; only emit
        # them when the operator opts in via STATSPAI_MCP_DEBUG=1. Plain
        # ``"<class>: <msg>"`` is enough for the agent to remediate in
        # the common case.
        data = None
        if os.environ.get("STATSPAI_MCP_DEBUG", "").strip() in {"1", "true",
                                                                  "True", "yes"}:
            data = {"traceback": traceback.format_exc()}
        return _jsonrpc_error(
            request_id, -32000, f"{type(exc).__name__}: {exc}",
            data=data,
        )
    return _jsonrpc_result(request_id, result)


# ═══════════════════════════════════════════════════════════════════════
#  stdio event loop
# ═══════════════════════════════════════════════════════════════════════

def serve_stdio(
    stdin: Optional[Iterable[str]] = None,
    stdout=None,
) -> None:
    """Run the JSON-RPC loop on stdio until stdin closes.

    Parameters
    ----------
    stdin, stdout : file-like, optional
        Defaults to ``sys.stdin`` / ``sys.stdout``. Tests can supply
        in-memory buffers instead.
    """
    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        response = handle_request(line)
        if response is None:
            continue
        stdout.write(response + "\n")
        stdout.flush()


def main() -> None:  # pragma: no cover
    """Entry point for ``python -m statspai.agent.mcp_server``."""
    serve_stdio()


__all__ = [
    "serve_stdio",
    "handle_request",
    "tool_manifest",
    "MCP_PROTOCOL_VERSION",
    "SERVER_NAME",
    "SERVER_VERSION",
]


if __name__ == "__main__":  # pragma: no cover
    main()
