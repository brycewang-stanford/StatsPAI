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

# JSON-RPC error taxonomy lives in ``_errors`` so split helper modules
# (``_resources``, ``_prompts``, …) can raise the same typed errors
# without forming a circular import through ``mcp_server``. The
# underscore-prefixed aliases preserve the v1.x private surface for
# tests / agents that subclass.
from ._errors import (
    RpcError as _RpcError,
    InvalidParamsError as _InvalidParamsError,
    ResourceNotFoundError as _ResourceNotFoundError,
)


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
            # Use POSIX form so JSON output is byte-stable across
            # OSes (Windows would otherwise emit ``\\tmp\\x`` which
            # breaks downstream consumers and round-trip tests).
            return o.as_posix()
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



# Data-file loading moved to ``_data_loader``. The shim below
# preserves the v1.x private names tests + downstream callers
# reach for.
from ._data_loader import (
    DEFAULT_MAX_DATA_BYTES as _DEFAULT_MAX_DATA_BYTES,
    max_data_bytes as _max_data_bytes,
    is_remote_url as _is_remote_url,
    load_dataframe as _load_dataframe,
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


# Resource catalog / function detail / templates moved to
# ``_resources``. The shim below preserves the v1.x private names
# the test suite + downstream callers reach for.
from ._resources import (
    FUNCTION_URI_PREFIX as _FUNCTION_URI_PREFIX,
    RESULT_URI_PREFIX as _RESULT_URI_PREFIX,
    catalog_text as _catalog_text_impl,
    functions_index as _functions_index,
    function_detail as _function_detail,
    handle_resources_list as _handle_resources_list,
    handle_resources_read as _resources_read_impl,
    handle_resources_templates_list as _handle_resources_templates_list,
)


def _catalog_text():
    return _catalog_text_impl(SERVER_VERSION)


def _handle_resources_read(params):
    return _resources_read_impl(
        params,
        json_default=_json_default,
        server_version=SERVER_VERSION,
        InvalidParamsError=_InvalidParamsError,
        ResourceNotFoundError=_ResourceNotFoundError,
    )


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
    # Snapshot the client's capability advertisement so server-side
    # sampling helpers can route to ``sampling/createMessage`` when
    # supported. ``_sampling.set_capability(False)`` is the safe
    # default (the LLM helpers fall through to the user-API-key
    # fallback path).
    from . import _sampling
    client_caps = (params.get("capabilities") or {}) if isinstance(params, dict) else {}
    has_sampling = isinstance(client_caps, dict) and "sampling" in client_caps
    _sampling.set_capability(has_sampling)
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


#: Module-global pointer to the active stdout sink, set by
#: :func:`serve_stdio`. ``_handle_tools_call`` reads it to write
#: ``notifications/progress`` mid-call without going through the
#: per-request return value (which is reserved for the final result).
#: ``None`` when the server is invoked outside the stdio loop (e.g.
#: by a unit test calling ``handle_request`` directly) — in that case
#: progress notifications are dropped silently, which is the right
#: thing for in-process tests.
_PROGRESS_SINK = None


def _make_progress_drain():
    """Return a callable that writes a progress notification to the
    active stdio sink. Returns a no-op if no sink is registered."""
    sink = _PROGRESS_SINK
    if sink is None:
        return lambda payload: None

    def _drain(payload):
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": payload,
        }, default=_json_default)
        try:
            sink.write(msg + "\n")
            sink.flush()
        except Exception:
            # If stdout is closed mid-call, drop the notification —
            # the next handle_request will surface the real error.
            pass

    return _drain


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

    # MCP ``_meta.progressToken`` is the standard handshake the client
    # uses to opt in to receiving progress notifications. It's set
    # OUTSIDE the ``arguments`` block (per spec) — pull it from
    # ``params['_meta']``.
    meta = params.get("_meta") or {}
    progress_token = meta.get("progressToken") if isinstance(meta, dict) else None

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

    # Run the actual estimator under the timeout-enforcing runner so
    # MCP can stay responsive during long calls (BCF / spec_curve /
    # synthdid_placebo / dml). Tools that don't hit ``progress(...)``
    # see no behaviour change.
    from ._runner import run_with_progress, tool_timeout

    def _do():
        return execute_tool(name, arguments,
                             data=df,
                             detail=detail,
                             result_id=result_id,
                             as_handle=as_handle)

    drain = _make_progress_drain() if progress_token is not None else None

    ok, payload = run_with_progress(
        _do,
        progress_token=progress_token,
        timeout=tool_timeout(),
        drain=drain,
    )

    if not ok:
        if isinstance(payload, TimeoutError):
            raise _RpcError(str(payload))
        # Unexpected exception — re-raise so the outer ``handle_request``
        # turns it into a clean ``-32000`` JSON-RPC error.
        raise payload

    result = payload
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

from ._prompts import (
    PROMPTS as _PROMPTS,
    SafeDict as _SafeDict,
    handle_prompts_list as _prompts_list_impl,
    handle_prompts_get as _prompts_get_impl,
)


def _handle_prompts_list(params):
    return _prompts_list_impl(params)


def _handle_prompts_get(params):
    return _prompts_get_impl(params, _InvalidParamsError, _ResourceNotFoundError)



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

    # JSON-RPC reply (no ``method``) — likely a response to a
    # server-initiated ``sampling/createMessage`` request. Route it
    # to the sampling matcher; if no pending request matches, fall
    # through to the regular notification-drop path.
    if isinstance(msg, dict) and "method" not in msg and "id" in msg:
        from . import _sampling
        if _sampling.route_response(msg):
            return None

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

    global _PROGRESS_SINK
    _PROGRESS_SINK = stdout

    # Register a writer for server-initiated ``sampling/createMessage``
    # requests. Helpers that need to invoke the client's LLM go through
    # ``_sampling.request_sampling`` which fails closed (raises
    # ``UnsupportedSamplingError``) when this writer isn't set OR the
    # client never advertised the capability — i.e. server-side
    # sampling is opt-in on both sides.
    from . import _sampling

    def _writer(line: str) -> None:
        stdout.write(line + "\n")
        stdout.flush()

    _sampling.set_writer(_writer)
    try:
        for raw in stdin:
            line = raw.strip()
            if not line:
                continue
            response = handle_request(line)
            if response is None:
                continue
            stdout.write(response + "\n")
            stdout.flush()
    finally:
        _PROGRESS_SINK = None
        _sampling.set_writer(None)
        _sampling.set_capability(False)


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
