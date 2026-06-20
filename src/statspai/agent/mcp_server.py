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
the estimator, and returns the result both as a human-readable JSON
``text`` block and — for clients on protocol ``2025-06-18`` and up — as
machine-readable ``structuredContent`` validated against each tool's
``outputSchema``.

Protocol features
-----------------
The server negotiates its protocol revision with the client
(:data:`SUPPORTED_PROTOCOL_VERSIONS`, newest preferred) and, on top of
the original ``2024-11-05`` surface, advertises:

* **Tool annotations** (``2025-03-26``) — every tool is tagged
  ``readOnlyHint=true`` / ``openWorldHint=false`` (StatsPAI estimators
  read the supplied data and compute; they never mutate state), so a
  client can auto-approve calls.
* **Structured tool output** (``2025-06-18``) — ``outputSchema`` on every
  tool plus ``structuredContent`` on every result.

Older clients negotiate ``2024-11-05`` and simply ignore the extra
fields, so the additions are fully backward-compatible.

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
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO, cast

from ._data_loader import (
    DEFAULT_MAX_DATA_BYTES as _DEFAULT_MAX_DATA_BYTES,
    max_data_bytes as _max_data_bytes,
    is_remote_url as _is_remote_url,
    load_dataframe as _load_dataframe,
)
from ._errors import (
    RpcError as _RpcError,
    InvalidParamsError as _InvalidParamsError,
    ResourceNotFoundError as _ResourceNotFoundError,
)
from ._prompts import (
    PROMPTS as _PROMPTS,
    SafeDict as _SafeDict,
    handle_prompts_list as _prompts_list_impl,
    handle_prompts_get as _prompts_get_impl,
)
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

_PRIVATE_COMPAT_EXPORTS = (
    _DEFAULT_MAX_DATA_BYTES,
    _max_data_bytes,
    _is_remote_url,
    _FUNCTION_URI_PREFIX,
    _RESULT_URI_PREFIX,
    _functions_index,
    _function_detail,
    _PROMPTS,
    _SafeDict,
)


#: Protocol revision this server *prefers* (the latest it implements).
#: Bumped from ``2024-11-05`` once the server gained the two features that
#: revision lacks: per-tool ``annotations`` (added in ``2025-03-26``) and
#: structured tool output — ``outputSchema`` + ``structuredContent`` (added
#: in ``2025-06-18``). Both are now emitted by :func:`_build_mcp_tools` /
#: :func:`_handle_tools_call`, so advertising the newer revision is honest.
MCP_PROTOCOL_VERSION = "2025-06-18"

#: Every protocol revision this server can speak, newest first. The
#: handshake (:func:`_handle_initialize`) negotiates against this set: if
#: the client asks for one we support we echo it verbatim (per spec, the
#: server MUST reply with the requested version when supported); otherwise
#: we fall back to :data:`MCP_PROTOCOL_VERSION` (the latest). The added
#: tool fields (``annotations`` / ``outputSchema`` / ``structuredContent``)
#: are backward-compatible — older clients negotiating ``2024-11-05`` simply
#: ignore the extra keys, so there is no behavioural downside to a client
#: that only knows the original revision.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

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
    except (ImportError, AttributeError):  # pragma: no cover — statspai must import
        pass
    return "0.0.0"


SERVER_VERSION = _resolve_server_version()


def tool_manifest(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """Lazy proxy for the agent tool manifest.

    Keeping this import lazy matters for MCP cold start: the live
    registry path imports pandas / scipy / sklearn-heavy modules, while
    most MCP clients first need only the static schema snapshot.
    """
    from .tools import tool_manifest as _tool_manifest

    return _tool_manifest(*args, **kwargs)


def execute_tool(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Lazy proxy for runtime tool dispatch."""
    from .tools import execute_tool as _execute_tool

    return _execute_tool(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  JSON-RPC helpers
# ═══════════════════════════════════════════════════════════════════════


def _jsonrpc_result(request_id: Any, result: Any) -> str:
    return json.dumps(
        _clean_floats(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        ),
        default=_json_default,
        allow_nan=False,
        separators=(",", ":"),
    )


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> str:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return json.dumps(
        _clean_floats(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": err,
            }
        ),
        default=_json_default,
        allow_nan=False,
        separators=(",", ":"),
    )


def _jsonrpc_result_preencoded(request_id: Any, result_json: str) -> str:
    """Build a JSON-RPC result from an already JSON-encoded result body."""
    encoded_id = json.dumps(
        _clean_floats(request_id),
        default=_json_default,
        allow_nan=False,
        separators=(",", ":"),
    )
    return f'{{"jsonrpc":"2.0","id":{encoded_id},"result":{result_json}}}'


def _clean_floats(o: Any) -> Any:
    """Recursively replace native float NaN/Inf with ``None``.

    json.dumps' ``default=`` callback is **not** invoked for native
    Python ``float`` values — they are "natively serialisable" as the
    non-standard literals ``NaN`` / ``Infinity`` / ``-Infinity``. Strict
    JSON parsers (RFC 8259, including Claude Desktop's ``JSON.parse``)
    reject those tokens — typically with "No number after minus sign"
    when they hit ``-Infinity``.

    Walk dicts / lists / tuples (the only Python-native containers
    json.dumps recurses into) before serialising so nan/inf can never
    reach the output. Strings, bytes, and any non-container leaf pass
    through untouched — numpy arrays / DataFrames / etc. are still
    routed through :func:`_json_default`, which itself returns cleaned
    Python structures.
    """
    if isinstance(o, float):
        import math

        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, dict):
        return {k: _clean_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_floats(v) for v in o]
    return o


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

    Any branch that produces a Python list/dict (numpy arrays, pandas
    Series/DataFrame/Index/Categorical) routes its return through
    :func:`_clean_floats` so nan/inf inside the produced container are
    scrubbed before json.dumps re-walks them.
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
            return _clean_floats({"real": float(o.real), "imag": float(o.imag)})
        if isinstance(o, _np.datetime64):
            # ns-precision ISO-8601 string; stable across pandas versions
            return str(o)
        if isinstance(o, _np.timedelta64):
            return str(o)
        if isinstance(o, _np.ndarray):
            return _clean_floats(o.tolist())
    except ImportError:  # pragma: no cover
        pass

    try:
        import pandas as _pd

        if isinstance(o, _pd.DataFrame):
            return _clean_floats(o.to_dict(orient="list"))
        if isinstance(o, _pd.Series):
            return _clean_floats(o.to_dict())
        if isinstance(o, _pd.Index):
            return _clean_floats(o.tolist())
        if isinstance(o, _pd.Timestamp):
            return o.isoformat()
        if isinstance(o, _pd.Timedelta):
            return o.isoformat()
        if isinstance(o, _pd.Categorical):
            return _clean_floats(list(o))
        if isinstance(o, _pd.Interval):
            return _clean_floats({"left": o.left, "right": o.right, "closed": o.closed})
    except ImportError:  # pragma: no cover
        pass

    if isinstance(o, (set, frozenset)):
        return _clean_floats(sorted(o, key=str))
    if isinstance(o, bytes):
        # Round-trippable; agents reading JSON shouldn't get garbled UTF-8
        import base64

        return {"__bytes_b64__": base64.b64encode(o).decode("ascii")}

    from decimal import Decimal

    if isinstance(o, Decimal):
        v = float(o)
        import math

        return None if (math.isnan(v) or math.isinf(v)) else v

    from pathlib import PurePath

    if isinstance(o, PurePath):
        # Use POSIX form so JSON output is byte-stable across OSes (Windows
        # would otherwise emit ``\\tmp\\x`` which breaks downstream consumers
        # and round-trip tests).
        return o.as_posix()

    from enum import Enum

    if isinstance(o, Enum):
        return _clean_floats(o.value)

    # dataclasses (without using asdict, which recurses and re-hits us)
    if hasattr(o, "__dataclass_fields__"):
        return _clean_floats({f: getattr(o, f, None) for f in o.__dataclass_fields__})

    if hasattr(o, "__dict__"):
        return _clean_floats(
            {k: v for k, v in vars(o).items() if not k.startswith("_")}
        )
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
_DATALESS_OVERRIDES = frozenset(
    {
        "honest_did",
        "sensitivity",
        "audit_result",
        "brief_result",
        "interpret_result",
        "sensitivity_from_result",
        "honest_did_from_result",
        "plot_from_result",
        "bibtex",
        "from_stata",
        "from_r",
    }
)


#: Backwards-compatible alias for the old hand-curated set. New code
#: should call :func:`_dataless_tool_names` to get the registry-derived
#: union; tests / external callers that imported this constant continue
#: to see a stable surface.
_DATALESS_TOOLS = _DATALESS_OVERRIDES


#: Shared JSON Schema for the *structured* tool result (MCP ``2025-06-18``+
#: ``outputSchema`` / ``structuredContent``). Estimator payloads are
#: heterogeneous and vary by ``detail`` level, so this schema is
#: deliberately permissive — ``additionalProperties: true`` and no
#: ``required`` keys — while still *documenting* the common agent-facing
#: envelope so a client gets type hints for the fields it can rely on.
#: The same object the server serialises into the ``text`` content block is
#: also returned verbatim as ``structuredContent``; this schema is what a
#: spec-compliant client validates that object against. Every documented
#: property mirrors a real key emitted by ``CausalResult.to_dict`` /
#: ``_default_serializer`` / ``_enrichment.enrich_payload`` / the
#: ``execute_tool`` error envelope — no invented fields.
_RESULT_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Agent-facing estimator result. Shape depends on the tool and the "
        "`detail` level; only a subset of these keys appears on any given "
        "call, and tools may add estimator-specific keys (additionalProperties "
        "is permitted). On failure the object instead carries `error` "
        "(+ `remediation` / `error_kind` / `error_payload`)."
    ),
    "properties": {
        "estimate": {
            "type": ["number", "null"],
            "description": "Point estimate of the target effect.",
        },
        "std_error": {
            "type": ["number", "null"],
            "description": "Standard error of the estimate.",
        },
        "p_value": {"type": ["number", "null"]},
        "conf_low": {
            "type": ["number", "null"],
            "description": "Lower confidence bound.",
        },
        "conf_high": {
            "type": ["number", "null"],
            "description": "Upper confidence bound.",
        },
        "estimand": {
            "type": "string",
            "description": "Target estimand (e.g. ATT, ATE, LATE).",
        },
        "method": {"type": "string", "description": "Estimator / method name."},
        "n_obs": {
            "type": ["integer", "null"],
            "description": "Number of observations used.",
        },
        "coefficients": {
            "type": "object",
            "description": (
                "Per-regressor table (regression-style results): "
                "name → {estimate, std_error, p_value}."
            ),
            "additionalProperties": True,
        },
        "diagnostics": {
            "type": "object",
            "description": "Scalar diagnostic statistics keyed by name.",
            "additionalProperties": True,
        },
        "violations": {
            "type": "array",
            "description": (
                "Assumption violations flagged for this design "
                "(present at detail='agent')."
            ),
            "items": {"type": "object", "additionalProperties": True},
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "next_steps": {
            "type": "array",
            "description": "Suggested follow-up analyses (detail='agent').",
            "items": {"type": "object", "additionalProperties": True},
        },
        "suggested_functions": {
            "type": "array",
            "description": "StatsPAI functions worth calling next.",
            "items": {"type": "string"},
        },
        "next_calls": {
            "type": "array",
            "description": (
                "Ready-to-dispatch JSON-RPC tools/call payloads "
                "for the recommended follow-ups (enrichment)."
            ),
            "items": {"type": "object", "additionalProperties": True},
        },
        "citations": {
            "type": "array",
            "description": "Verified bib keys / BibTeX for the methods used.",
            "items": {"type": ["object", "string"]},
        },
        "narrative": {
            "type": "string",
            "description": "Short markdown digest of the result.",
        },
        "result_id": {
            "type": "string",
            "description": (
                "Server-side handle to the fitted result "
                "(present when as_handle=true)."
            ),
        },
        "result_uri": {
            "type": "string",
            "description": "statspai://result/<id> form of result_id.",
        },
        "error": {
            "type": "string",
            "description": "Error message when the call failed.",
        },
        "error_kind": {
            "type": "string",
            "description": (
                "Stable StatsPAIError code "
                "(e.g. assumption_violation, "
                "identification_failure) for programmatic "
                "branching."
            ),
        },
        "remediation": {
            "type": "object",
            "description": "Structured repair hints for the next call.",
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}


#: URI of the resource that serves the full :data:`_RESULT_OUTPUT_SCHEMA`.
RESULT_SCHEMA_URI = "statspai://schema/result"


#: The *compact* output schema actually injected into every tool's
#: ``outputSchema`` in ``tools/list``. The full documented envelope above
#: is byte-identical for all ~480 tools, so inlining it everywhere would
#: duplicate ~1.3 MB of the same schema across the manifest (half the
#: payload) for zero added information. Instead each tool advertises this
#: compact-but-valid schema — enough for a client to validate
#: ``structuredContent`` (any object passes) and to learn the result is an
#: object — and the full field-by-field reference is served **once** as the
#: :data:`RESULT_SCHEMA_URI` resource. The actual fields are also visible on
#: every call via the ``structuredContent`` payload itself.
_RESULT_OUTPUT_SCHEMA_COMPACT: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "description": (
        "Agent-facing estimator result (object). Shape varies by tool and "
        "the `detail` level; on failure it carries `error` / `error_kind` / "
        "`remediation` instead. Full typed field reference: read the "
        f"`{RESULT_SCHEMA_URI}` resource."
    ),
}


def _schema_snapshot_enabled() -> bool:
    raw = os.environ.get("STATSPAI_MCP_SCHEMA_SNAPSHOT")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _schema_snapshot_dirs() -> List[Path]:
    """Candidate directories containing offline schema exports."""
    here = Path(__file__).resolve()
    dirs: List[Path] = []
    for parent in here.parents:
        candidate = parent / "schemas"
        if (candidate / "tools.json").exists():
            dirs.append(candidate)
    return dirs


@lru_cache(maxsize=1)
def _load_schema_snapshot() -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Load the committed import-free schema bundle when it is available.

    The source tree ships ``schemas/tools.json`` + ``schemas/functions.json``
    and CI keeps it in sync with the live registry. Reading that bundle is
    much cheaper than importing the full registry tail just to answer the
    MCP client's first ``tools/list`` request. Operators can force the live
    path with ``STATSPAI_MCP_SCHEMA_SNAPSHOT=0`` while developing dynamic
    registries.
    """
    if not _schema_snapshot_enabled():
        return None
    for directory in _schema_snapshot_dirs():
        try:
            index_path = directory / "index.json"
            tools_path = directory / "tools.json"
            functions_path = directory / "functions.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("schema_version") != "1":
                continue
            if index.get("statspai_version") != SERVER_VERSION:
                continue
            tools = json.loads(tools_path.read_text(encoding="utf-8"))
            functions = json.loads(functions_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(tools, list) and isinstance(functions, list):
            return {
                "tools": cast(List[Dict[str, Any]], tools),
                "functions": cast(List[Dict[str, Any]], functions),
            }
    return None


def _agent_tool_manifest() -> List[Dict[str, Any]]:
    snapshot = _load_schema_snapshot()
    if snapshot is not None:
        return snapshot["tools"]
    return tool_manifest()


def _snapshot_dataless_tool_names() -> Optional["frozenset[str]"]:
    snapshot = _load_schema_snapshot()
    if snapshot is None:
        return None
    data_bound: "set[str]" = set()
    for item in snapshot["functions"]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        params = item.get("parameters") or {}
        required = set(params.get("required") or [])
        props = params.get("properties") or {}
        if "data" in required and "data" in props:
            data_bound.add(name)
    tool_names: "set[str]" = set()
    for tool in snapshot["tools"]:
        name = tool.get("name")
        if isinstance(name, str):
            tool_names.add(name)
    return frozenset(
        name
        for name in tool_names
        if name in _DATALESS_OVERRIDES or name not in data_bound
    )


@lru_cache(maxsize=1)
def _dataless_tool_names() -> "frozenset[str]":
    """Names of tools that take no DataFrame.

    Auto-derived from the registry: any registered function without a
    required ``data`` parameter is dataless. Falls back to
    :data:`_DATALESS_OVERRIDES` alone if registry introspection fails.
    """
    snapshot = _snapshot_dataless_tool_names()
    if snapshot is not None:
        return snapshot

    derived: "set[str]" = set(_DATALESS_OVERRIDES)
    try:
        from ..registry import _REGISTRY, _ensure_full_registry

        _ensure_full_registry()
        for name, spec in _REGISTRY.items():
            params = getattr(spec, "params", None) or []
            has_required_data = any(p.name == "data" and p.required for p in params)
            if not has_required_data:
                # No required `data` param → safe to mark dataless. Tools
                # that take an OPTIONAL data still get data_path injected
                # for client convenience but won't be required.
                derived.add(name)
    except (ImportError, AttributeError, TypeError):
        pass
    return frozenset(derived)


@lru_cache(maxsize=1)
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
    manifest = _agent_tool_manifest()
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
        props["detail"] = {
            "type": "string",
            "enum": list(_DETAIL_LEVELS),
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

        # Tool annotations (MCP ``2025-03-26``+). StatsPAI tools are
        # estimators / diagnostics / report builders: they read the
        # supplied dataset, compute, and return — they never mutate the
        # input file or any external state, and their "world" is the
        # closed StatsPAI library plus the one dataset handed to them
        # (not an open set of external entities). So ``readOnlyHint`` and
        # ``openWorldHint=False`` are honestly uniform; a client can use
        # ``readOnlyHint`` to auto-approve calls without a confirmation
        # prompt. A manifest entry may override either hint by carrying
        # its own ``annotations`` dict (none do today).
        annotations = dict(t.get("annotations") or {})
        annotations.setdefault("readOnlyHint", True)
        annotations.setdefault("openWorldHint", False)

        out.append(
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": schema,
                "annotations": annotations,
                "outputSchema": _RESULT_OUTPUT_SCHEMA_COMPACT,
            }
        )
    return out


@lru_cache(maxsize=1)
def _tools_list_result_json() -> str:
    """Cached JSON payload for the static ``tools/list`` result."""
    return json.dumps(
        _clean_floats({"tools": _build_mcp_tools()}),
        default=_json_default,
        allow_nan=False,
        separators=(",", ":"),
    )


def _clear_mcp_caches() -> None:
    """Clear cached MCP schema material for tests / dynamic registries."""
    _load_schema_snapshot.cache_clear()
    _dataless_tool_names.cache_clear()
    _build_mcp_tools.cache_clear()
    _tools_list_result_json.cache_clear()
    for name in ("_catalog_text_impl", "_functions_index", "_function_detail"):
        fn = globals().get(name)
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()


# Data-file loading moved to ``_data_loader``. The shim below
# preserves the v1.x private names tests + downstream callers
# reach for.


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


def _catalog_text() -> str:
    return _catalog_text_impl(SERVER_VERSION)


def _handle_resources_read(params: Dict[str, Any]) -> Dict[str, Any]:
    return _resources_read_impl(
        params,
        json_default=_json_default,
        server_version=SERVER_VERSION,
        InvalidParamsError=_InvalidParamsError,
        ResourceNotFoundError=_ResourceNotFoundError,
        clean_for_json=_clean_floats,
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

    # Version negotiation (MCP spec): when the client requests a revision
    # we support, the server MUST reply with that same revision; otherwise
    # we offer the latest we implement. A client that sends no
    # ``protocolVersion`` (or an unknown one) gets our preferred revision.
    requested = params.get("protocolVersion") if isinstance(params, dict) else None
    negotiated = (
        requested
        if isinstance(requested, str) and requested in SUPPORTED_PROTOCOL_VERSIONS
        else MCP_PROTOCOL_VERSION
    )
    return {
        "protocolVersion": negotiated,
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
_PROGRESS_SINK: Optional[TextIO] = None


def _make_progress_drain() -> Callable[[Dict[str, Any]], None]:
    """Return a callable that writes a progress notification to the
    active stdio sink. Returns a no-op if no sink is registered."""
    sink = _PROGRESS_SINK
    if sink is None:

        def _noop(payload: Dict[str, Any]) -> None:
            return None

        return _noop

    def _drain(payload: Dict[str, Any]) -> None:
        msg = json.dumps(
            _clean_floats(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": payload,
                }
            ),
            default=_json_default,
            allow_nan=False,
            separators=(",", ":"),
        )
        try:
            sink.write(msg + "\n")
            sink.flush()
        except (OSError, ValueError):
            # If stdout is closed mid-call, drop the notification —
            # the next handle_request will surface the real error.
            pass

    return _drain


def _handle_tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    arguments = dict(params.get("arguments") or {})
    if not isinstance(name, str):
        raise _InvalidParamsError("`name` is required and must be a string")

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
            df = _load_dataframe(
                data_path,
                columns=data_columns,
                sample_n=data_sample_n,
            )
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

    def _do() -> Dict[str, Any]:
        return execute_tool(
            name,
            arguments,
            data=df,
            detail=detail,
            result_id=result_id,
            as_handle=as_handle,
        )

    drain = _make_progress_drain()

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

    # Image content: estimators can attach a PNG plot under ``_plot_png``
    # for the MCP layer to surface as an image content block. Claude
    # vision (and any MCP client supporting image content) will render
    # it inline; the bytes are stripped from the JSON payload above.
    result = payload
    plot_bytes = None
    result_for_text = result
    if isinstance(result, dict):
        plot_bytes = result.get("_plot_png")
        if isinstance(plot_bytes, (bytes, bytearray)):
            result_for_text = {k: v for k, v in result.items() if k != "_plot_png"}

    text = json.dumps(
        _clean_floats(result_for_text), indent=2, default=_json_default, allow_nan=False
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

    if isinstance(plot_bytes, (bytes, bytearray)):
        import base64

        content.append(
            {
                "type": "image",
                "data": base64.b64encode(plot_bytes).decode("ascii"),
                "mimeType": "image/png",
            }
        )

    out: Dict[str, Any] = {
        "content": content,
        "isError": bool(isinstance(result, dict) and result.get("error")),
    }

    # Structured tool output (MCP ``2025-06-18``+). We advertise an
    # ``outputSchema`` on every tool, so we return the result object
    # *also* as ``structuredContent`` — the machine-readable twin of the
    # ``text`` block, which spec-compliant clients validate against the
    # schema and hand to the model as typed data instead of re-parsing
    # the serialised string. ``result_for_text`` is always a JSON object
    # (``execute_tool`` returns a dict; the image bytes are already
    # stripped), so it conforms to the ``type: object`` schema. The
    # surrounding ``_jsonrpc_result`` re-walks it through
    # ``_clean_floats`` / ``_json_default``, so numpy / nan values are
    # scrubbed here exactly as they are in the text block. Older clients
    # that negotiated an earlier revision simply ignore the extra key.
    if isinstance(result_for_text, dict):
        out["structuredContent"] = result_for_text

    return out


# ═══════════════════════════════════════════════════════════════════════
#  Prompts: canned workflow templates
# ═══════════════════════════════════════════════════════════════════════
#
# MCP clients (Claude Desktop, Cursor) surface ``prompts/list`` entries
# in their UI as prompt shortcut buttons. We ship a small
# set of curated workflow templates so users can spin up a typical
# StatsPAI agent loop without writing the prompt from scratch.
#
# Per spec:
# - ``prompts/list`` returns a list of {name, description, arguments[]}
# - ``prompts/get`` takes {name, arguments} and returns
#   {description, messages: [{role, content}]}


def _handle_prompts_list(params: Dict[str, Any]) -> Dict[str, Any]:
    return _prompts_list_impl(params)


def _handle_prompts_get(params: Dict[str, Any]) -> Dict[str, Any]:
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
        return _jsonrpc_error(request_id, -32601, f"Method not found: {method!r}")

    try:
        if method == "tools/list":
            return _jsonrpc_result_preencoded(
                request_id,
                _tools_list_result_json(),
            )
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
        if os.environ.get("STATSPAI_MCP_DEBUG", "").strip() in {
            "1",
            "true",
            "True",
            "yes",
        }:
            data = {"traceback": traceback.format_exc()}
        return _jsonrpc_error(
            request_id,
            -32000,
            f"{type(exc).__name__}: {exc}",
            data=data,
        )
    return _jsonrpc_result(request_id, result)


# ═══════════════════════════════════════════════════════════════════════
#  stdio event loop
# ═══════════════════════════════════════════════════════════════════════


def serve_stdio(
    stdin: Optional[Iterable[str]] = None,
    stdout: Optional[TextIO] = None,
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
    "SUPPORTED_PROTOCOL_VERSIONS",
    "RESULT_SCHEMA_URI",
    "SERVER_NAME",
    "SERVER_VERSION",
]


if __name__ == "__main__":  # pragma: no cover
    main()
