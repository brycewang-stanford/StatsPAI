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
Every tool takes one data source — ``data_path`` (absolute path on the
server), ``data_id`` (a handle from ``load_data`` / ``transform_data``)
or an inline ``data_records`` / ``data_csv`` table — plus whatever
column-name arguments the underlying StatsPAI function expects. The
server loads the data, runs the estimator, and returns the result as
``structuredContent`` (validated against each tool's ``outputSchema``)
plus the same object serialised compactly in a ``text`` block for
clients that predate structured output.

Failures while *executing* a call — an unknown or expired ``data_id`` /
``result_id``, an unreadable file, a refused path, a timeout, an
estimator error — come back as a normal result with ``isError: true``
and a structured payload (``error_kind``, ``message``, ``hint``, …), so
the model sees them and can repair its next call. Only protocol errors
(malformed JSON-RPC, unknown method or tool name, ``params`` /
``arguments`` not an object) are JSON-RPC errors.

Results are bounded by ``max_output_bytes`` (default 256 KiB,
``STATSPAI_MCP_MAX_OUTPUT_BYTES``): the longest lists / tables are cut
first and listed under ``truncated``; risk lists (``violations``,
``runtime_warnings``, ``degradations``, ``warnings``) are cut last and
leave ``risk_summary`` + ``risk_details_complete: false``; ``output_budget``
reports ``truncated`` or ``unavoidable_overflow`` (the budget covers
``structuredContent``, not the repeated ``text`` block or an image).
Non-finite numbers are sent as
``null`` and their paths listed under ``_nonfinite``. Estimator results
carry ``replay`` — the ``sp.<fn>(...)`` call that reproduces them.

Operator controls: ``STATSPAI_MCP_DATA_ROOTS`` restricts readable
directories, network URLs need ``STATSPAI_MCP_ALLOW_REMOTE=1``,
``STATSPAI_MCP_MAX_DATA_BYTES`` caps loads (local and remote),
``STATSPAI_MCP_TOOL_TIMEOUT_SECONDS`` bounds a call,
``STATSPAI_MCP_WORKERS`` sizes the ``tools/call`` pool, and
``STATSPAI_MCP_MAX_QUEUED_CALLS`` / ``STATSPAI_MCP_MAX_ORPHANED_CALLS`` /
``STATSPAI_MCP_MAX_QUEUE_SECONDS`` / ``STATSPAI_MCP_MAX_REQUEST_BYTES``
bound what the stdio loop admits (``server_busy`` / ``-32600`` beyond
them). ``STATSPAI_MCP_ISOLATION=process`` runs every call that needs no
data handle in a child process that is killed on timeout or cancel;
result handles travel both ways, so a fit that asks for one
(``as_handle``) and a follow-up that reads one (``result_id``) are both
isolated (:mod:`statspai.agent._process_worker`).

Protocol features
-----------------
The server negotiates its protocol revision with the client
(:data:`SUPPORTED_PROTOCOL_VERSIONS`, newest preferred) and, on top of
the original ``2024-11-05`` surface, advertises:

* **Tool annotations** (``2025-03-26``) — ``readOnlyHint`` is true
  except for tools that can write a file; ``openWorldHint`` is true only
  when network data URLs are enabled.
* **Structured tool output** (``2025-06-18``) — ``outputSchema`` on every
  tool plus ``structuredContent`` on every result.
* ``ping``, ``notifications/progress`` (with ``_meta.progressToken``) and
  ``notifications/cancelled`` — the stdio loop keeps answering while a
  tool runs (see :func:`serve_stdio`).

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
import os
import queue
import sys
import threading
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO, cast

from ._data_loader import DEFAULT_MAX_DATA_BYTES as _DEFAULT_MAX_DATA_BYTES
from ._data_loader import data_provenance as _data_provenance
from ._data_loader import is_remote_url as _is_remote_url
from ._data_loader import load_dataframe as _load_dataframe
from ._data_loader import max_data_bytes as _max_data_bytes
from ._errors import InvalidParamsError as _InvalidParamsError
from ._errors import ResourceNotFoundError as _ResourceNotFoundError
from ._errors import RpcError as _RpcError
from ._prompts import PROMPTS as _PROMPTS
from ._prompts import SafeDict as _SafeDict
from ._prompts import handle_prompts_get as _prompts_get_impl
from ._prompts import handle_prompts_list as _prompts_list_impl
from ._resources import FUNCTION_URI_PREFIX as _FUNCTION_URI_PREFIX
from ._resources import RESULT_URI_PREFIX as _RESULT_URI_PREFIX
from ._resources import catalog_text as _catalog_text_impl
from ._resources import function_detail as _function_detail
from ._resources import functions_index as _functions_index
from ._resources import handle_resources_list as _handle_resources_list
from ._resources import handle_resources_read as _resources_read_impl
from ._resources import (
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


def _make_json_default(clean: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Build a ``json.dumps(default=...)`` encoder.

    ``clean`` post-processes every container the encoder produces:
    :func:`_clean_floats` for wire output (non-finite → ``null``), or the
    identity for :data:`_json_default_raw`, which keeps NaN / ±Inf so the
    tool-result path can record where they were.
    """

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

            if clean is _clean_floats and (math.isnan(o) or math.isinf(o)):
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

                if clean is not _clean_floats:
                    return v
                return None if (math.isnan(v) or math.isinf(v)) else v
            if isinstance(o, _np.complexfloating):
                return clean({"real": float(o.real), "imag": float(o.imag)})
            if isinstance(o, _np.datetime64):
                # ns-precision ISO-8601 string; stable across pandas versions
                return str(o)
            if isinstance(o, _np.timedelta64):
                return str(o)
            if isinstance(o, _np.ndarray):
                return clean(o.tolist())
        except ImportError:  # pragma: no cover
            pass

        try:
            import pandas as _pd

            if isinstance(o, _pd.DataFrame):
                return clean(o.to_dict(orient="list"))
            if isinstance(o, _pd.Series):
                return clean(o.to_dict())
            if isinstance(o, _pd.Index):
                return clean(o.tolist())
            if isinstance(o, _pd.Timestamp):
                return o.isoformat()
            if isinstance(o, _pd.Timedelta):
                return o.isoformat()
            if isinstance(o, _pd.Categorical):
                return clean(list(o))
            if isinstance(o, _pd.Interval):
                return clean({"left": o.left, "right": o.right, "closed": o.closed})
        except ImportError:  # pragma: no cover
            pass

        if isinstance(o, (set, frozenset)):
            return clean(sorted(o, key=str))
        if isinstance(o, bytes):
            # Round-trippable; agents reading JSON shouldn't get garbled UTF-8
            import base64

            return {"__bytes_b64__": base64.b64encode(o).decode("ascii")}

        from decimal import Decimal

        if isinstance(o, Decimal):
            v = float(o)
            import math

            if clean is not _clean_floats:
                return v
            return None if (math.isnan(v) or math.isinf(v)) else v

        from pathlib import PurePath

        if isinstance(o, PurePath):
            # Use POSIX form so JSON output is byte-stable across OSes (Windows
            # would otherwise emit ``\\tmp\\x`` which breaks downstream consumers
            # and round-trip tests).
            return o.as_posix()

        from enum import Enum

        if isinstance(o, Enum):
            return clean(o.value)

        # dataclasses (without using asdict, which recurses and re-hits us)
        if hasattr(o, "__dataclass_fields__"):
            return clean({f: getattr(o, f, None) for f in o.__dataclass_fields__})

        if hasattr(o, "__dict__"):
            return clean({k: v for k, v in vars(o).items() if not k.startswith("_")})
        return str(o)

    return _json_default


#: Encoder used for every JSON-RPC message: non-finite floats become null.
_json_default = _make_json_default(_clean_floats)

#: Encoder that keeps NaN / ±Inf so :func:`_normalise_tool_result` can
#: record *where* they were before scrubbing them (``_nonfinite``).
_json_default_raw = _make_json_default(lambda x: x)


# ═══════════════════════════════════════════════════════════════════════
#  Tool spec transformation: StatsPAI manifest → MCP tools/list spec
# ═══════════════════════════════════════════════════════════════════════

#: Reserved argument names the MCP server consumes itself before
#: dispatching to the estimator. ``data_path`` becomes a DataFrame;
#: ``detail`` controls the result-serialisation level (see
#: ``CausalResult.to_dict``). Each entry is the (single) source of
#: truth for both the schema injection in :func:`_build_mcp_tools`
#: and the argument stripping in :func:`_handle_tools_call`.
_RESERVED_ARG_NAMES = (
    "data_path",
    "data_id",
    "data_records",
    "data_csv",
    "detail",
)

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
        # Discovery meta-tools (search / describe never touch data;
        # call_function loads data only when the target function needs it).
        "search_functions",
        "describe_function",
        "call_function",
        # Data-handle tools take data_id / data_path / inline data — none of
        # the three is individually required.
        "load_data",
        "describe_data",
        "transform_data",
        "route_estimator",
    }
)

#: Tools that receive the resolved data handle / provenance of their input
#: (under ``_source_data_id`` / ``_source_provenance``) so they can record
#: lineage. Every other tool sees only the DataFrame.
_DATA_TOOLS = frozenset({"load_data", "describe_data", "transform_data"})


#: Tool-list profiles. ``tools/list`` with the ``full`` profile returns
#: every auto-generated tool (several hundred entries, megabytes on the
#: wire) — more than any client context window holds. ``curated`` (the
#: default everywhere) lists only the
#: hand-written estimator / workflow / pipeline tools plus the three
#: discovery meta-tools (``search_functions`` / ``describe_function`` /
#: ``call_function``) through which every other registered function
#: stays reachable. ``core`` is the smallest useful set for constrained
#: clients. Select with ``statspai-mcp --profile <name>`` or
#: ``STATSPAI_MCP_PROFILE``; ``tools/call`` accepts any tool name under
#: every profile — the profile only shapes the advertised list.
_PROFILES = ("core", "curated", "full")

_CORE_PROFILE_TOOLS = frozenset(
    {
        "search_functions",
        "describe_function",
        "call_function",
        "route_estimator",
        "detect_design",
        "preflight",
        "recommend",
        "regress",
        "did",
        "callaway_santanna",
        "rdrobust",
        "ivreg",
        "synth",
        "dml",
        "audit_result",
        "brief_result",
        "interpret_result",
        "honest_did_from_result",
        "sensitivity_from_result",
        "plot_from_result",
        "bibtex",
        "pipeline_did",
        "pipeline_iv",
        "pipeline_rd",
    }
)

#: Hand-listed tools whose underlying function writes a file when asked
#: to. The advertised set is this list *plus* every tool whose input
#: schema has a file-output parameter (:data:`_FILE_OUTPUT_PARAMS`), see
#: :func:`_is_file_writing_tool`. Their MCP annotations carry
#: ``readOnlyHint=False`` so a client does not auto-approve a call that
#: touches the filesystem.
_FILE_WRITING_TOOLS = frozenset(
    {
        "cs_report",
        "did_report",
        "influence_functions",
        "rd_dashboard",
        "synth_report_to_file",
        "synth_to_excel",
    }
)

#: Input-schema parameter names that name a file / directory to write.
_FILE_OUTPUT_PARAMS = frozenset(
    {
        "output",
        "output_path",
        "output_file",
        "output_dir",
        "outfile",
        "out",
        "out_path",
        "out_dir",
        "outdir",
        "path",
        "file",
        "filename",
        "file_path",
        "filepath",
        "fname",
        "save",
        "save_to",
        "save_path",
        "save_dir",
        "export",
        "export_path",
        "to_file",
        "dir",
        "directory",
    }
)


def _is_file_writing_tool(name: str, input_schema: Optional[Dict[str, Any]]) -> bool:
    """True if ``name`` may write a file: hand-listed or schema-derived."""
    if name in _FILE_WRITING_TOOLS:
        return True
    props = (input_schema or {}).get("properties") or {}
    return any(p in _FILE_OUTPUT_PARAMS for p in props)


def _file_writing_tool_names() -> "frozenset[str]":
    """Every advertised tool that carries ``readOnlyHint=False``."""
    return frozenset(
        t["name"]
        for t in _build_mcp_tools()
        if t.get("annotations", {}).get("readOnlyHint") is False
    )


#: Profile used when neither :func:`set_tool_profile` nor
#: ``STATSPAI_MCP_PROFILE`` chooses one — the same for the CLI and for
#: in-process callers of :func:`handle_request`.
DEFAULT_PROFILE = "curated"


def _normalise_profile(profile: Optional[str]) -> str:
    if profile is None:
        profile = os.environ.get("STATSPAI_MCP_PROFILE", DEFAULT_PROFILE)
    key = str(profile).strip().lower()
    if key not in _PROFILES:
        raise ValueError(
            f"Unknown MCP tool profile {profile!r}; choose one of {list(_PROFILES)}."
        )
    return key


#: Active profile for this server process (set by :func:`main` /
#: :func:`set_tool_profile`; ``None`` means "read ``STATSPAI_MCP_PROFILE``,
#: default :data:`DEFAULT_PROFILE`").
_ACTIVE_PROFILE: Optional[str] = None


def set_tool_profile(profile: Optional[str]) -> str:
    """Set the ``tools/list`` profile for this process and return it."""
    global _ACTIVE_PROFILE
    _ACTIVE_PROFILE = None if profile is None else _normalise_profile(profile)
    _build_mcp_tools.cache_clear()
    _tools_list_result_json.cache_clear()
    return _normalise_profile(_ACTIVE_PROFILE)


def _profile_tool_names(profile: str) -> Optional["frozenset[str]"]:
    """Names advertised under ``profile``; ``None`` means every tool."""
    if profile == "full":
        return None
    from .tools import tool_manifest as _tool_manifest

    curated = frozenset(t["name"] for t in _tool_manifest(curated_only=True))
    if profile == "curated":
        return curated
    return frozenset(n for n in curated if n in _CORE_PROFILE_TOOLS)


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
        "result_card": {
            "type": "object",
            "description": (
                "sp.result_card(result): estimand, sample (rows used vs "
                "input, exclusions), specification (formula, weights, call "
                "arguments), inference (covariance, t/normal, CI level), "
                "provenance (versions, data hash), evidence (configuration-"
                "level validation_scope where mapped, else the function tier "
                "flagged as such), assumptions (declared, not verified), "
                "limitations. Absent at detail='minimal'."
            ),
            "additionalProperties": True,
        },
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
                "JSON-RPC tools/call payloads for recommended "
                "follow-ups. Each item carries ready=true when it can "
                "be dispatched as-is; otherwise ready=false plus "
                "missing_arguments lists the fields an agent must fill."
            ),
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object", "additionalProperties": True},
                    "ready": {"type": "boolean"},
                    "missing_arguments": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "rationale": {"type": "string"},
                    "hint": {"type": "string"},
                },
            },
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
        "data_provenance": {
            "type": "object",
            "description": (
                "MCP data_path source summary when a tool call loaded data "
                "server-side: sanitized source, scheme, format, requested "
                "columns/sample, and local file size/mtime/SHA-256 when "
                "available."
            ),
            "additionalProperties": True,
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
        "replay": {
            "type": "string",
            "description": (
                "The sp.<fn>(...) call that was run, with the data source "
                "named in a trailing comment."
            ),
        },
        "replay_completeness": {
            "type": "object",
            "description": (
                "What re-running `replay` needs. level: 'standalone' (a new "
                "process can re-run it given the file in needs), "
                "'session_replayable' (depends on a data_id / result_id "
                "held by this server), 'call_only' (documents the call but "
                "cannot re-run it: inline or remote data, or an argument "
                "without a literal form). needs: the dependencies."
            ),
            "additionalProperties": True,
        },
        "isolation": {
            "type": "object",
            "description": (
                "Present when the call ran in a child process "
                "(STATSPAI_MCP_ISOLATION=process): mode; adopted_handles "
                "when the result handle was handed over to this server and "
                "works in follow-up calls; dropped_handles (with a reason) "
                "for handles that could not outlive the child."
            ),
            "additionalProperties": True,
        },
        "runtime_warnings": {
            "type": "array",
            "description": (
                "Python warnings raised during the call, as {category, "
                "message}; at most 20 distinct ones (risk_summary gives the "
                "count when more were raised)."
            ),
            "items": {"type": "object", "additionalProperties": True},
        },
        "truncated": {
            "type": "array",
            "description": (
                "Containers shortened to meet max_output_bytes: {path, "
                "total, shown}. Absent when nothing was cut."
            ),
            "items": {"type": "object", "additionalProperties": True},
        },
        "risk_details_complete": {
            "type": "boolean",
            "description": (
                "false when violations / runtime_warnings / degradations / "
                "warnings were shortened: the lists shown are not all the "
                "risks raised. Absent when they are complete."
            ),
        },
        "risk_summary": {
            "type": "object",
            "description": (
                "Present with risk_details_complete=false. Per shortened "
                "risk field: total, shown, omitted, by_severity, categories; "
                "plus full_details (how to obtain every entry)."
            ),
            "additionalProperties": True,
        },
        "output_budget": {
            "type": "object",
            "description": (
                "Outcome of max_output_bytes when the result did not fit "
                "untouched: status ('truncated' = cut to fit; "
                "'unavoidable_overflow' = fields that are never cut exceed "
                "the budget, listed under oversized_fields), max_bytes, "
                "actual_bytes, scope ('structuredContent': the text block "
                "repeats the object and an image is sent on top). Absent "
                "when the result fit."
            ),
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}


#: URI of the resource that serves the full :data:`_RESULT_OUTPUT_SCHEMA`.
RESULT_SCHEMA_URI = "statspai://schema/result"


#: The *compact* output schema actually injected into every tool's
#: ``outputSchema`` in ``tools/list``. The full documented envelope above
#: is byte-identical for all ~580 tools, so inlining it everywhere would
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


@lru_cache(maxsize=4)
def _build_mcp_tools(profile: Optional[str] = None) -> List[Dict[str, Any]]:
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

    ``profile`` defaults to the active profile (:func:`set_tool_profile`);
    the resources layer passes ``"full"`` so ``statspai://functions``
    indexes every tool whatever the advertised list is.
    """
    from ._data_loader import ALLOW_REMOTE_ENV, DATA_ROOTS_ENV, remote_loading_enabled

    remote_ok = remote_loading_enabled()
    data_path_description = (
        "Absolute path to a data file on the server (or file:// URL). "
        "Supported: .csv / .tsv / .txt (delimited), .parquet / .pq, "
        ".feather / .arrow, .xlsx / .xls, .dta (Stata), .json / .jsonl. "
        + (
            "Network URLs (s3://, gs://, https://) are enabled on this server."
            if remote_ok
            else f"Network URLs are disabled (operator opt-in: {ALLOW_REMOTE_ENV}=1)."
        )
        + f" The operator may restrict readable directories ({DATA_ROOTS_ENV})."
    )
    manifest = _agent_tool_manifest()
    dataless = _dataless_tool_names()
    profile = _normalise_profile(_ACTIVE_PROFILE if profile is None else profile)
    allowed = _profile_tool_names(profile)
    # The committed snapshot may predate a curated tool added in this
    # release; make sure every curated entry is advertised regardless.
    present = {t["name"] for t in manifest}
    from .tools import tool_manifest as _tool_manifest

    manifest = list(manifest) + [
        t for t in _tool_manifest(curated_only=True) if t["name"] not in present
    ]
    if allowed is not None:
        manifest = [t for t in manifest if t["name"] in allowed]
    out: List[Dict[str, Any]] = []
    for t in manifest:
        schema = dict(t.get("input_schema") or {})
        props = dict(schema.get("properties") or {})
        required = list(schema.get("required") or [])
        writes_files = _is_file_writing_tool(t["name"], schema)
        if "data_path" not in props:
            props["data_path"] = {
                "type": "string",
                "description": data_path_description,
            }
            # Mark required ONLY for tools whose underlying function
            # actually takes a DataFrame; dataless tools leave
            # ``data_path`` optional so strict-schema MCP clients don't
            # refuse to dispatch them.
            if t["name"] not in dataless:
                required.append("data_path")
        if "data_id" not in props:
            props["data_id"] = {
                "type": "string",
                "description": (
                    "Handle (d_…) of a dataset already loaded with load_data "
                    "or derived with transform_data. Use instead of "
                    "data_path to avoid re-sending the file; the handle's "
                    "lineage is recorded in data_provenance."
                ),
            }
        if "data_records" not in props:
            props["data_records"] = {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
                "description": (
                    "Inline table as a JSON array of row objects (small "
                    "data only; same byte cap as file loads)."
                ),
            }
        if "data_csv" not in props:
            props["data_csv"] = {
                "type": "string",
                "description": "Inline table as CSV text with a header row.",
            }
        if "data_columns" not in props:
            props["data_columns"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional column projection, honoured by every "
                    "reader that supports it (CSV/Parquet/Feather/Stata) "
                    "and by streamed sampling."
                ),
            }
        if "data_sample_n" not in props:
            props["data_sample_n"] = {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional uniform random subsample size "
                    "(seed=0, deterministic, file order kept). Files over "
                    "the server's size cap are sampled in one streamed "
                    "pass for .csv/.tsv/.txt/.parquet/.jsonl/.dta — useful "
                    "on huge panels."
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
        if "max_output_bytes" not in props:
            props["max_output_bytes"] = {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "Byte budget for this result (default 262144, server env "
                    "STATSPAI_MCP_MAX_OUTPUT_BYTES; 0 = no limit). Over "
                    "budget, the longest lists / tables are shortened first "
                    "and listed under `truncated`; headline numbers are "
                    "never cut. Risk lists are cut last and leave "
                    "`risk_summary`; `output_budget` reports the outcome."
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
        # supplied dataset, compute, and return. ``readOnlyHint`` is
        # False for tools that can write a file (hand list + schema-
        # derived output parameters). ``openWorldHint`` is True only when
        # the tool can reach beyond the server — i.e. it accepts
        # ``data_path`` and the operator enabled network URLs
        # (``STATSPAI_MCP_ALLOW_REMOTE``); otherwise the tool's world is
        # the closed library plus local / inline data. A manifest entry
        # may override either hint by carrying its own ``annotations``.
        annotations = dict(t.get("annotations") or {})
        annotations.setdefault("readOnlyHint", not writes_files)
        annotations.setdefault("openWorldHint", bool(remote_ok))

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
    "  6. For any next_calls item, dispatch it directly only when "
    "ready=true; when ready=false, fill missing_arguments first.\n"
    "  7. bibtex(keys=[...]) for verified citations — never invent "
    "references; paper.bib is the single source of truth.\n\n"
    "Economist migration helpers: use prompts/list for "
    "stata_command_workflow, r_command_workflow, and "
    "cross_language_command_check. For cross-software evidence, read "
    "statspai://parity/track-a-summary; it summarizes committed artifacts "
    "only and is not a live Stata/R execution.\n\n"
    "Data handling: load_data(data_path=... | data_records=[...] | "
    "data_csv='...') returns a data_id; pass data_id to any tool instead "
    "of re-sending the file. transform_data(data_id=..., operations=[...]) "
    "filters / reshapes / winsorises / imputes and returns a new handle "
    "whose lineage rides in every result's data_provenance; "
    "describe_data profiles a handle; statspai://data/{id} reads it.\n\n"
    "Data provenance: tools/call responses that load data include "
    "data_provenance. Local files carry size, mtime, and SHA-256; remote "
    "URLs are sanitized and not re-hashed; inline tables are hashed; "
    "handles carry their lineage.\n\n"
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


class _ToolCallError(Exception):
    """A failure while *executing* a ``tools/call``.

    MCP separates protocol errors (unknown method, malformed request,
    unknown tool — JSON-RPC ``error``) from tool-execution errors, which
    are returned as a normal result with ``isError: true`` so the model
    sees them and can correct its next call. Many clients never show a
    JSON-RPC error to the model at all, so a bad ``data_id``, an
    unreadable file, a timeout or an estimator failure must travel as a
    result. ``kind`` becomes ``error_kind``; ``fields`` (``hint``,
    ``miss_reason``, ...) are merged into ``structuredContent``.
    """

    def __init__(self, kind: str, message: str, **fields: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.fields = fields


def _tool_error_result(kind: str, message: str, **fields: Any) -> Dict[str, Any]:
    """``tools/call`` result for a tool-execution failure (``isError``)."""
    payload: Dict[str, Any] = {"error": message, "error_kind": kind, "message": message}
    for k, v in fields.items():
        if v is not None:
            payload[k] = v
    # The text block is the compact JSON of the same (small) payload, as
    # for successful results, so a client that ignores structuredContent
    # can still parse ``error_kind`` / ``hint``.
    payload = _clean_floats(payload)
    text = json.dumps(payload, separators=(",", ":"), default=_json_default)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": True,
        "structuredContent": payload,
    }


def _exception_kind(exc: BaseException, default: str) -> str:
    """Stable ``error_kind`` for a data-loading exception."""
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    from ..exceptions import StatsPAIError

    code = getattr(exc, "code", None) if isinstance(exc, StatsPAIError) else None
    if isinstance(code, str) and code not in ("", "method_incompatibility"):
        return code
    return default


def _debug_enabled() -> bool:
    return os.environ.get("STATSPAI_MCP_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _normalise_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Native-JSON copy of ``result`` with non-finite values recorded.

    numpy / pandas values are converted with :data:`_json_default_raw`
    (which keeps NaN / ±Inf), then every non-finite float becomes
    ``null`` and its JSON Pointer is listed under ``_nonfinite`` so an
    agent can tell "infinite" or "undefined" from "missing".
    """
    from ._output_budget import scrub_nonfinite

    try:
        native = json.loads(
            json.dumps(result, default=_json_default_raw, allow_nan=True)
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise _ToolCallError(
            "serialization_error",
            f"The tool ran but its result could not be serialised to JSON: "
            f"{type(exc).__name__}: {exc}",
            hint="Retry with detail='minimal', or report this as a bug.",
        ) from exc
    clean, found, total = scrub_nonfinite(native)
    if not isinstance(clean, dict):
        clean = {"value": clean}
    if found:
        clean["_nonfinite"] = found
        if total > len(found):
            clean["_nonfinite_total"] = total
    return clean


def _handle_tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    if not isinstance(name, str) or not name:
        raise _InvalidParamsError("`name` is required and must be a string")
    raw_args = params.get("arguments")
    if raw_args is None:
        arguments: Dict[str, Any] = {}
    elif isinstance(raw_args, dict):
        arguments = dict(raw_args)
    else:
        raise _InvalidParamsError("`arguments` must be a JSON object")
    try:
        from . import _process_worker

        if _process_worker.isolation_mode() == "process" and _process_worker.eligible(
            name, arguments
        ):
            isolated = _run_isolated_call(name, arguments, params)
            if isolated is not None:
                return isolated
        try:
            # An isolated worker first takes the results its call reads.
            _process_worker.import_inbound()
            return _run_tools_call(name, arguments, params)
        finally:
            # ... and leaves the ones it produced for the parent.
            _process_worker.export_handles()
    except _ToolCallError as err:
        return _tool_error_result(err.kind, err.message, **err.fields)


def _run_isolated_call(
    name: str, arguments: Dict[str, Any], params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Run a call in a child process that can be killed.

    Returns ``None`` when the call reads a result that cannot be handed
    to a worker; the caller then runs it on the thread runner.
    """
    from . import _process_worker
    from ._runner import current_cancel_event, tool_timeout

    meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else None
    sink = _PROGRESS_SINK

    def _forward(line: str) -> None:
        if sink is not None:
            sink.write(line + "\n")
            sink.flush()

    import shutil
    import tempfile

    timeout = tool_timeout()
    import secrets

    handoff = tempfile.mkdtemp(prefix="statspai-mcp-handoff-")
    handoff_key = secrets.token_hex(32)
    reads = arguments.get("result_id")
    if isinstance(reads, str) and reads:
        if not _process_worker.export_inbound(handoff, handoff_key, [reads]):
            # Not in the cache, or not picklable: answer from this process.
            shutil.rmtree(handoff, ignore_errors=True)
            return None
    try:
        status, payload = _process_worker.run_isolated(
            name,
            arguments,
            meta,
            timeout=timeout,
            cancel_event=current_cancel_event(),
            forward=_forward,
            handoff_dir=handoff,
            handoff_key=handoff_key,
        )
        adopted: List[str] = []
        failed: Dict[str, str] = {}
        if status == "result":
            adopted, failed = _process_worker.import_handles(handoff, handoff_key)
            if isinstance(reads, str) and reads:
                # the handle the call read is still held here
                adopted = list(adopted) + [reads]
    finally:
        shutil.rmtree(handoff, ignore_errors=True)
    if status == "result":
        return _process_worker.annotate(payload, adopted=adopted, failed=failed)
    if status == "rpc_error":
        code = payload.get("code") if isinstance(payload, dict) else None
        message = str(payload.get("message")) if isinstance(payload, dict) else ""
        if code == -32602:
            raise _InvalidParamsError(message)
        raise _ToolCallError("internal_error", message, tool=name)
    if status == "timeout":
        raise _ToolCallError(
            "timeout",
            f"tool exceeded {timeout:.0f}s timeout "
            "(env: STATSPAI_MCP_TOOL_TIMEOUT_SECONDS)",
            hint=(
                "Retry on a sample (data_sample_n) or with cheaper options, or "
                "raise the timeout. The worker process was killed: nothing is "
                "left running."
            ),
            timeout_seconds=timeout,
            worker_may_still_be_running=False,
            worker_killed=True,
            isolation="process",
        )
    if status == "cancelled":
        raise _ToolCallError(
            "cancelled",
            "The client cancelled this tools/call.",
            worker_may_still_be_running=False,
            worker_killed=True,
            isolation="process",
        )
    raise _ToolCallError(
        "internal_error",
        f"isolated worker exited without an answer "
        f"(returncode {payload.get('returncode')})",
        tool=name,
        stderr=payload.get("stderr") if _debug_enabled() else None,
        isolation="process",
    )


def _resolve_tool_data(
    name: str, arguments: Dict[str, Any]
) -> "tuple[Any, Optional[Dict[str, Any]], str]":
    """Pop the data-source arguments and load the frame.

    Returns ``(df, data_provenance, replay_comment)``; raises
    :class:`_ToolCallError` for every caller-fixable failure.
    """
    data_path = arguments.pop("data_path", None)
    data_columns = arguments.pop("data_columns", None) or None
    data_sample_n = arguments.pop("data_sample_n", None)
    data_id = arguments.pop("data_id", None)
    data_records = arguments.pop("data_records", None)
    data_csv = arguments.pop("data_csv", None)

    from ._replay import data_comment

    sources = [
        k
        for k, v in (
            ("data_id", data_id),
            ("data_path", data_path),
            ("data_records", data_records),
            ("data_csv", data_csv),
        )
        if v is not None and v != ""
    ]
    if len(sources) > 1:
        raise _ToolCallError(
            "invalid_arguments",
            f"Pass exactly one data source; got {sources}.",
            hint=(
                "Use data_id for a handle from load_data / transform_data, "
                "data_path for a file or URL, data_records / data_csv for an "
                "inline table."
            ),
        )
    if data_columns is not None and (
        not isinstance(data_columns, list)
        or not all(isinstance(c, str) for c in data_columns)
    ):
        raise _ToolCallError(
            "invalid_arguments", "data_columns must be a list of column names."
        )

    df = None
    data_prov: Optional[Dict[str, Any]] = None
    comment = ""
    if data_id:
        from ._data_cache import DATA_CACHE, handle_provenance, missing_handle_error

        if not isinstance(data_id, str):
            raise _ToolCallError(
                "invalid_arguments", "data_id must be a string handle (d_...)."
            )
        cached = DATA_CACHE.get(data_id)
        if cached is None:
            miss = missing_handle_error(data_id)
            raise _ToolCallError(
                miss["error_kind"],
                miss["error"],
                hint=miss["hint"],
                miss_reason=miss["miss_reason"],
                data_id=data_id,
            )
        df = cached
        data_prov = handle_provenance(data_id)
        if data_columns:
            missing_cols = [c for c in data_columns if c not in df.columns]
            if missing_cols:
                raise _ToolCallError(
                    "column_not_found",
                    f"data_columns not in handle {data_id}: {missing_cols}",
                    hint="describe_data(data_id=...) lists the columns.",
                )
            df = df[list(data_columns)]
            data_prov["columns_requested"] = list(data_columns)
        comment = data_comment(data_id=data_id, data_columns=data_columns)
    elif data_records is not None or data_csv is not None:
        from ._data_cache import inline_frame

        try:
            df, data_prov = inline_frame(
                records=data_records, csv_text=data_csv, max_bytes=_max_data_bytes()
            )
        except (ValueError, TypeError) as e:  # MethodIncompatibility / parse errors
            raise _ToolCallError(
                _exception_kind(e, "invalid_data"),
                str(e),
                hint=getattr(e, "recovery_hint", None),
            ) from e
        comment = data_comment(inline_provenance=data_prov)
    elif data_path:
        if not isinstance(data_path, str):
            raise _ToolCallError(
                "invalid_arguments", "data_path must be a string path or URL."
            )
        try:
            df = _load_dataframe(
                data_path,
                columns=data_columns,
                sample_n=data_sample_n,
            )
            data_prov = _data_provenance(
                data_path,
                columns=data_columns,
                sample_n=data_sample_n,
            )
        except (OSError, ValueError, ImportError) as e:
            raise _ToolCallError(
                _exception_kind(e, "data_load_error"),
                str(e) or type(e).__name__,
                hint=getattr(e, "recovery_hint", None),
                data_path=data_path,
            ) from e
        comment = data_comment(
            data_path=data_path, data_columns=data_columns, data_sample_n=data_sample_n
        )
    if name in _DATA_TOOLS:
        # Let the handle tools record where their input came from.
        if data_id:
            arguments["_source_data_id"] = data_id
        if data_prov is not None:
            arguments["_source_provenance"] = data_prov
    return df, data_prov, comment


def _run_tools_call(
    name: str, arguments: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    # Server-handled args are stripped before estimator dispatch — the
    # estimator's signature has no ``data_path`` / ``detail`` etc. and
    # would crash with a "got an unexpected keyword argument" error.
    result_id = arguments.pop("result_id", None)
    as_handle = bool(arguments.pop("as_handle", False))
    raw_budget = arguments.pop("max_output_bytes", None)

    # MCP ``_meta.progressToken`` is the standard handshake the client
    # uses to opt in to receiving progress notifications. It's set
    # OUTSIDE the ``arguments`` block (per spec) — pull it from
    # ``params['_meta']``.
    meta = params.get("_meta") or {}
    progress_token = meta.get("progressToken") if isinstance(meta, dict) else None

    # Loading the data is supervised like the estimator: a read that never
    # returns (a FIFO, a stalled network mount, a slow URL) would otherwise
    # hold the worker forever, outside any timeout.
    from ._runner import ToolCancelled as _LoadCancelled
    from ._runner import current_cancel_event as _load_cancel_event
    from ._runner import orphaned_threads as _load_orphans
    from ._runner import run_with_progress as _run_supervised
    from ._runner import tool_timeout as _load_timeout

    load_limit = _load_timeout()
    loaded_ok, loaded = _run_supervised(
        lambda: _resolve_tool_data(name, arguments),
        timeout=load_limit,
        cancel_event=_load_cancel_event(),
    )
    if not loaded_ok:
        if isinstance(loaded, TimeoutError):
            raise _ToolCallError(
                "timeout",
                f"loading the data exceeded the {load_limit:.0f}s timeout "
                "(env: STATSPAI_MCP_TOOL_TIMEOUT_SECONDS)",
                hint=(
                    "The source did not finish reading. Check that data_path "
                    "is a regular file or a reachable URL; the read cannot be "
                    "killed and keeps waiting in the background."
                ),
                timeout_seconds=load_limit,
                stage="data_load",
                worker_may_still_be_running=True,
                orphaned_tool_threads=_load_orphans(),
            )
        if isinstance(loaded, _LoadCancelled):
            raise _ToolCallError(
                "cancelled",
                "The client cancelled this tools/call.",
                stage="data_load",
                worker_may_still_be_running=True,
            )
        raise loaded
    df, data_prov, replay_comment = loaded

    detail = arguments.pop("detail", "agent")
    if detail not in _DETAIL_LEVELS:
        raise _ToolCallError(
            "invalid_arguments",
            "detail must be one of "
            f"{', '.join(repr(v) for v in _DETAIL_LEVELS)}; got {detail!r}",
        )
    if raw_budget is not None and (
        isinstance(raw_budget, bool)
        or not isinstance(raw_budget, int)
        or raw_budget < 0
    ):
        raise _ToolCallError(
            "invalid_arguments",
            f"max_output_bytes must be a non-negative integer (0 = no limit); "
            f"got {raw_budget!r}",
        )
    from ._output_budget import apply_budget, max_output_bytes, note_risk_omission

    budget = max_output_bytes(raw_budget)

    if result_id is not None and result_id != "":
        from ._result_cache import RESULT_CACHE, missing_result_error

        if not isinstance(result_id, str):
            raise _ToolCallError(
                "invalid_arguments", "result_id must be a string handle (r_...)."
            )
        if RESULT_CACHE.get_entry(result_id) is None:
            miss = missing_result_error(result_id)
            raise _ToolCallError(
                miss["error_kind"],
                miss["error"],
                hint=miss["hint"],
                miss_reason=miss["miss_reason"],
                result_id=result_id,
                available_result_ids=miss["available_result_ids"],
            )
    else:
        result_id = None

    # Run the actual estimator under the timeout-enforcing runner so
    # MCP can stay responsive during long calls (BCF / spec_curve /
    # synthdid_placebo / dml).
    from ._runner import (
        ToolCancelled,
        current_cancel_event,
        orphaned_threads,
        run_with_progress,
        tool_timeout,
    )

    def _do() -> Dict[str, Any]:
        # Two protections around the estimator call:
        #
        # * ``print()`` inside an estimator (``verbose=True`` defaults,
        #   ``rdsummary``, ``assumption_audit``) would land on the
        #   JSON-RPC channel and corrupt the stream, so stdout is
        #   redirected to stderr for the duration of the call.
        # * Python warnings (``ConvergenceWarning``, ``AssumptionWarning``,
        #   weak-instrument / few-cluster notices, LIML->2SLS fallbacks)
        #   were only ever visible on stderr, which an MCP client never
        #   reads. They are recorded and attached to the payload under
        #   ``runtime_warnings`` so the agent sees what a human at a
        #   terminal would (CLAUDE.md §3.7, 失败要响亮).
        import contextlib
        import warnings as _warnings

        with contextlib.redirect_stdout(sys.stderr):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                out = execute_tool(
                    name,
                    arguments,
                    data=df,
                    detail=detail,
                    result_id=result_id,
                    as_handle=as_handle,
                )
        if caught and isinstance(out, dict):
            seen: set = set()
            recorded: List[Dict[str, str]] = []
            for w in caught:
                key = (w.category.__name__, str(w.message))
                if key in seen:
                    continue
                seen.add(key)
                if len(recorded) < 20:
                    recorded.append({"category": key[0], "message": key[1]})
            out["runtime_warnings"] = recorded
            # The cap is reported, not silent: risk_summary carries the
            # count of distinct warnings that were raised.
            note_risk_omission(out, "runtime_warnings", len(seen))
        return out

    timeout = tool_timeout()
    ok, payload = run_with_progress(
        _do,
        progress_token=progress_token,
        timeout=timeout,
        drain=_make_progress_drain(),
        cancel_event=current_cancel_event(),
    )

    if not ok:
        if isinstance(payload, TimeoutError):
            raise _ToolCallError(
                "timeout",
                str(payload),
                hint=(
                    "Retry on a sample (data_sample_n) or with cheaper options, "
                    "or raise STATSPAI_MCP_TOOL_TIMEOUT_SECONDS. The computation "
                    "cannot be killed: it keeps running in the background until "
                    "it finishes or reaches a progress checkpoint."
                ),
                timeout_seconds=timeout,
                worker_may_still_be_running=True,
                orphaned_tool_threads=orphaned_threads(),
            )
        if isinstance(payload, ToolCancelled):
            raise _ToolCallError(
                "cancelled",
                "The client cancelled this tools/call.",
                worker_may_still_be_running=True,
            )
        if isinstance(payload, _RpcError) or not isinstance(payload, Exception):
            # Typed protocol errors keep their code; KeyboardInterrupt /
            # SystemExit propagate.
            raise payload
        # An exception that escaped the dispatcher's own error envelope
        # (dispatch / serializer bug). Report it to the model as a tool
        # error; tracebacks only with STATSPAI_MCP_DEBUG=1.
        fields: Dict[str, Any] = {"tool": name}
        if _debug_enabled():
            fields["traceback"] = "".join(
                traceback.format_exception(
                    type(payload), payload, payload.__traceback__
                )
            )
        raise _ToolCallError(
            "internal_error", f"{type(payload).__name__}: {payload}", **fields
        )

    result = payload if isinstance(payload, dict) else {"value": payload}
    if (
        result.get("error_kind") == "unknown_tool"
        and "called_via" not in result
        and "error" in result
    ):
        # An unknown tool name is a protocol error per the MCP spec.
        raise _InvalidParamsError(
            f"Unknown tool: {name!r}. tools/list (or statspai://functions) "
            "lists the callable names."
        )

    rid = result.get("result_id")
    if data_prov is not None:
        result.setdefault("data_provenance", data_prov)
        if isinstance(rid, str):
            from ._result_cache import RESULT_CACHE

            RESULT_CACHE.annotate(rid, {"_mcp_data_provenance": data_prov})
    replay = result.get("replay")
    if isinstance(replay, str) and replay_comment and "  # data = " not in replay:
        result["replay"] = replay + replay_comment
        if isinstance(rid, str):
            from ._result_cache import RESULT_CACHE

            RESULT_CACHE.set_replay(rid, result["replay"])
    if isinstance(result.get("replay"), str):
        from ._replay import replay_completeness

        # A replay line is not a script: say what re-running it needs.
        result["replay_completeness"] = replay_completeness(
            result["replay"], result.get("data_provenance"), result_id=result_id
        )
        if isinstance(rid, str):
            # The handle can be exported as a script a new process can run.
            result["replay_completeness"][
                "bundle_uri"
            ] = f"statspai://result/{rid}/bundle"

    # Image content: estimators can attach a PNG plot under ``_plot_png``
    # for the MCP layer to surface as an image content block. Claude
    # vision (and any MCP client supporting image content) will render
    # it inline; the bytes are stripped from the JSON payload.
    plot_bytes = result.get("_plot_png")
    if isinstance(plot_bytes, (bytes, bytearray)):
        result = {k: v for k, v in result.items() if k != "_plot_png"}
    else:
        plot_bytes = None

    structured = _normalise_tool_result(result)
    # Writes ``truncated`` / ``risk_summary`` / ``output_budget`` itself.
    structured, _ = apply_budget(structured, budget)

    # Structured tool output (MCP ``2025-06-18``+): ``structuredContent``
    # is the machine-readable result; the spec asks that the serialised
    # JSON also be returned as a ``text`` block for older clients, so the
    # same object is sent once more, compactly (no indentation).
    text = json.dumps(structured, separators=(",", ":"), allow_nan=False)
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    if plot_bytes is not None:
        import base64

        content.append(
            {
                "type": "image",
                "data": base64.b64encode(plot_bytes).decode("ascii"),
                "mimeType": "image/png",
            }
        )
    return {
        "content": content,
        "isError": bool(structured.get("error")),
        "structuredContent": structured,
    }


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


def _handle_ping(params: Dict[str, Any]) -> Dict[str, Any]:
    """MCP ``ping``: liveness check, answered with an empty result."""
    return {}


_METHODS = {
    "ping": _handle_ping,
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

    if not isinstance(msg, dict):
        return _jsonrpc_error(None, -32600, "Invalid Request: expected a JSON object")

    request_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params")
    if params is None:
        params = {}

    # JSON-RPC 2.0: a notification has no ``id`` field at all.
    if request_id is None and "id" not in msg:
        return None
    # MCP convention: ``notifications/<x>`` is a notification regardless
    # of whether the client erroneously included an ``id``. Silently
    # drop it instead of replying with -32601, which would generate
    # protocol noise on every session.
    if isinstance(method, str) and method.startswith("notifications/"):
        return None

    handler = _METHODS.get(method) if isinstance(method, str) else None
    if handler is None:
        return _jsonrpc_error(request_id, -32601, f"Method not found: {method!r}")
    if not isinstance(params, dict):
        return _jsonrpc_error(
            request_id, -32602, "Invalid params: `params` must be a JSON object"
        )

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
        if _debug_enabled():
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


class _LockedSink:
    """A ``write`` / ``flush`` pair serialised by one lock.

    Three writers share the stdio channel: the request loop (responses),
    the progress drain (``notifications/progress`` from the tool worker)
    and the sampling client (``sampling/createMessage``). Interleaved
    partial lines would corrupt the JSON-RPC stream, so every line goes
    through this object.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            n = self._stream.write(text)
            self._stream.flush()
            return n or 0

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()

    def write_line(self, line: str) -> None:
        self.write(line + "\n")


def _is_jsonrpc_reply(line: str) -> Optional[Dict[str, Any]]:
    """Parse ``line`` and return it when it is a JSON-RPC *reply*.

    A reply has an ``id`` and no ``method`` — the shape the client sends
    back for a server-initiated ``sampling/createMessage``. Anything else
    (requests, notifications, garbage) returns ``None`` and is handled by
    the request loop.
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(msg, dict) and "method" not in msg and "id" in msg:
        return msg
    return None


#: Env var: size of the ``tools/call`` worker pool (default 1).
WORKERS_ENV = "STATSPAI_MCP_WORKERS"


def _worker_count() -> int:
    raw = os.environ.get(WORKERS_ENV)
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


#: Env var: ``tools/call`` requests allowed to wait behind the running
#: ones (default 32; ``0`` = unlimited). Beyond it a call is answered at
#: once with ``error_kind='server_busy'`` instead of queueing.
MAX_QUEUED_CALLS_ENV = "STATSPAI_MCP_MAX_QUEUED_CALLS"
DEFAULT_MAX_QUEUED_CALLS = 32

#: Env var: timed-out / cancelled computations that may still be running
#: in the background before new calls are refused (default 4; ``0`` =
#: unlimited). A thread cannot be killed, so without this cap every
#: timeout adds one more estimator computing next to the following call.
MAX_ORPHANED_CALLS_ENV = "STATSPAI_MCP_MAX_ORPHANED_CALLS"
DEFAULT_MAX_ORPHANED_CALLS = 4

#: Env var: longest a ``tools/call`` may wait in the queue before it
#: starts, in seconds (default 900; ``0`` = unlimited). A call that waited
#: longer is answered ``server_busy`` instead of being run for a client
#: that has most likely given up on it.
MAX_QUEUE_SECONDS_ENV = "STATSPAI_MCP_MAX_QUEUE_SECONDS"
DEFAULT_MAX_QUEUE_SECONDS = 900

#: Env var: largest accepted request line, in bytes (default 64 MiB;
#: ``0`` = unlimited). Larger lines are answered with ``-32600`` and
#: never parsed or queued.
MAX_REQUEST_BYTES_ENV = "STATSPAI_MCP_MAX_REQUEST_BYTES"
DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024


def _env_limit(name: str, default: int) -> Optional[int]:
    """Non-negative integer limit from the environment; ``None`` = unlimited."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else None


def _request_key(request_id: Any) -> str:
    """Hashable, type-preserving key for a JSON-RPC id (``1`` ≠ ``"1"``)."""
    return json.dumps(request_id, sort_keys=True)


def serve_stdio(
    stdin: Optional[Iterable[str]] = None,
    stdout: Optional[TextIO] = None,
    *,
    workers: Optional[int] = None,
) -> None:
    """Run the JSON-RPC loop on stdio until stdin closes.

    Three kinds of thread cooperate:

    * a **reader** consumes stdin. JSON-RPC *replies* (the client's answer
      to a server-initiated ``sampling/createMessage``) are routed straight
      to :mod:`_sampling`, so a tool that asks the client's LLM a question
      (``interpret_result``) gets its answer while it is still running;
      everything else is queued for the main loop.
    * the **main loop** answers ``initialize`` / ``ping`` /
      ``tools/list`` / ``resources/*`` / ``prompts/*`` inline and hands
      each ``tools/call`` to the worker pool, so a long estimator never
      blocks liveness checks or catalogue reads. It also handles
      ``notifications/cancelled``: the named in-flight call's cancel event
      is set, the tool stops at its next progress checkpoint
      (:func:`statspai.agent._runner.progress`), and — per the MCP spec —
      no response is sent for the cancelled request.
    * a **worker pool** runs ``tools/call`` requests. It has one worker by
      default, so estimator execution stays serialised (estimators share
      process-global state such as warning filters and matplotlib);
      ``STATSPAI_MCP_WORKERS`` (or ``workers=``) raises it.

    The loop is bounded. A request line over
    ``STATSPAI_MCP_MAX_REQUEST_BYTES`` is refused unparsed (``-32600``);
    a ``tools/call`` arriving while ``STATSPAI_MCP_MAX_QUEUED_CALLS``
    others already wait, or while ``STATSPAI_MCP_MAX_ORPHANED_CALLS``
    timed-out computations are still running, is answered immediately
    with an ``isError`` result of kind ``server_busy``; one that waited in
    the queue longer than ``STATSPAI_MCP_MAX_QUEUE_SECONDS`` gets the same
    answer when its turn comes instead of being run; and a
    ``tools/call`` reusing the id of one still in flight is refused
    (``-32600``) rather than taking over its cancel handle.

    All output goes through one locked sink, so responses, progress
    notifications and sampling requests never interleave mid-line.
    Responses to ``tools/call`` may arrive out of order relative to
    requests handled inline, as JSON-RPC allows.

    Parameters
    ----------
    stdin, stdout : file-like, optional
        Defaults to ``sys.stdin`` / ``sys.stdout``. Tests can supply
        in-memory buffers instead.
    workers : int, optional
        Worker-pool size; defaults to ``STATSPAI_MCP_WORKERS`` or 1.
    """
    from concurrent.futures import ThreadPoolExecutor

    from . import _runner

    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout

    sink = _LockedSink(stdout)

    global _PROGRESS_SINK
    _PROGRESS_SINK = cast(TextIO, sink)

    # Register a writer for server-initiated ``sampling/createMessage``
    # requests. Helpers that need to invoke the client's LLM go through
    # ``_sampling.request_sampling`` which fails closed (raises
    # ``UnsupportedSamplingError``) when this writer isn't set OR the
    # client never advertised the capability — i.e. server-side
    # sampling is opt-in on both sides.
    from . import _sampling

    _sampling.set_writer(sink.write_line)

    inbox: "queue.Queue[Optional[str]]" = queue.Queue()
    n_workers = workers if workers else _worker_count()
    max_queued = _env_limit(MAX_QUEUED_CALLS_ENV, DEFAULT_MAX_QUEUED_CALLS)
    max_orphans = _env_limit(MAX_ORPHANED_CALLS_ENV, DEFAULT_MAX_ORPHANED_CALLS)
    max_request = _env_limit(MAX_REQUEST_BYTES_ENV, DEFAULT_MAX_REQUEST_BYTES)
    max_wait = _env_limit(MAX_QUEUE_SECONDS_ENV, DEFAULT_MAX_QUEUE_SECONDS)

    def _emit(line: str) -> None:
        try:
            sink.write_line(line)
        except (OSError, ValueError) as exc:  # stdout closed by the client
            print(f"statspai-mcp: could not write response: {exc}", file=sys.stderr)

    def _reader() -> None:
        try:
            for raw in stdin:
                line = raw.strip()
                if not line:
                    continue
                if max_request is not None and len(line) > max_request:
                    # Not parsed, so the id is unknown (JSON-RPC: null).
                    _emit(
                        _jsonrpc_error(
                            None,
                            -32600,
                            f"Invalid Request: {len(line)} characters exceeds "
                            f"the {max_request}-byte limit "
                            f"({MAX_REQUEST_BYTES_ENV}); pass data by "
                            "data_path or a data_id handle instead of inline.",
                        )
                    )
                    continue
                reply = _is_jsonrpc_reply(line)
                if reply is not None and _sampling.route_response(reply):
                    continue
                inbox.put(line)
        finally:
            inbox.put(None)

    inflight: Dict[str, threading.Event] = {}
    inflight_lock = threading.Lock()

    def _busy(in_flight: int) -> Optional[Dict[str, Any]]:
        """``server_busy`` result when a new call must not be admitted."""
        orphans = _runner.orphaned_threads()
        if max_orphans is not None and orphans >= max_orphans:
            return _tool_error_result(
                "server_busy",
                f"{orphans} timed-out or cancelled computations are still "
                "running in the background; no new call is started until "
                "they finish.",
                hint=(
                    "Wait and retry, or restart the server to stop them. "
                    f"{MAX_ORPHANED_CALLS_ENV} sets the limit."
                ),
                orphaned_tool_threads=orphans,
                retryable=True,
            )
        if max_queued is not None and in_flight >= n_workers + max_queued:
            return _tool_error_result(
                "server_busy",
                f"{in_flight} tools/call requests are already running or "
                "queued; this one was not queued.",
                hint=(
                    "Wait for earlier calls to answer (or cancel them) and "
                    f"retry. {MAX_QUEUED_CALLS_ENV} sets the limit."
                ),
                calls_in_flight=in_flight,
                retryable=True,
            )
        return None

    def _run_call(
        line: str, key: str, cancel: threading.Event, request_id: Any, queued_at: float
    ) -> None:
        try:
            if cancel.is_set():
                return  # cancelled while queued
            waited = time.monotonic() - queued_at
            if max_wait is not None and waited > max_wait:
                _emit(
                    _jsonrpc_result(
                        request_id,
                        _tool_error_result(
                            "server_busy",
                            f"This call waited {waited:.0f}s in the queue, past "
                            f"the {max_wait}s limit; it was not started.",
                            hint=(
                                "Retry now that earlier calls have finished. "
                                f"{MAX_QUEUE_SECONDS_ENV} sets the limit."
                            ),
                            queued_seconds=round(waited, 1),
                            retryable=True,
                        ),
                    )
                )
                return
            _runner.set_cancel_event(cancel)
            try:
                response = handle_request(line)
            finally:
                _runner.set_cancel_event(None)
            if response is not None and not cancel.is_set():
                _emit(response)
        finally:
            with inflight_lock:
                inflight.pop(key, None)

    def _cancel(params: Any) -> None:
        if not isinstance(params, dict) or "requestId" not in params:
            return
        with inflight_lock:
            ev = inflight.get(_request_key(params["requestId"]))
        if ev is not None:
            ev.set()

    executor = ThreadPoolExecutor(
        max_workers=n_workers,
        thread_name_prefix="statspai-mcp-call",
    )
    reader = threading.Thread(target=_reader, name="statspai-mcp-stdin", daemon=True)
    reader.start()
    try:
        while True:
            line = inbox.get()
            if line is None:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                msg = None
            if isinstance(msg, dict):
                method = msg.get("method")
                if method == "notifications/cancelled":
                    _cancel(msg.get("params"))
                    continue
                if method == "tools/call" and "id" in msg:
                    key = _request_key(msg["id"])
                    cancel = threading.Event()
                    with inflight_lock:
                        duplicate = key in inflight
                        busy = None if duplicate else _busy(len(inflight))
                        if not duplicate and busy is None:
                            inflight[key] = cancel
                    if duplicate:
                        _emit(
                            _jsonrpc_error(
                                msg["id"],
                                -32600,
                                "Invalid Request: a tools/call with this id "
                                "is still in flight.",
                            )
                        )
                    elif busy is not None:
                        _emit(_jsonrpc_result(msg["id"], busy))
                    else:
                        executor.submit(
                            _run_call, line, key, cancel, msg["id"], time.monotonic()
                        )
                    continue
            response = handle_request(line)
            if response is None:
                continue
            _emit(response)
    finally:
        # Let queued / running tool calls finish and write their responses
        # (a timed-out call returns at its deadline; an orphaned thread
        # does not hold the pool).
        executor.shutdown(wait=True)
        _PROGRESS_SINK = None
        _sampling.set_writer(None)
        _sampling.set_capability(False)


def _profile_help() -> str:
    """``--profile`` help text; sizes are described, not hard-coded."""
    return (
        "tools/list profile: 'core' (smallest useful set: discovery "
        "meta-tools, flagship estimators, result tools), 'curated' "
        "(hand-written estimators + workflow + pipelines + discovery "
        "meta-tools; default), or 'full' (every registered function as its "
        "own tool — several hundred entries, too large for most client "
        "context windows). Every function stays callable under every "
        "profile via call_function or by name."
    )


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    """Entry point for ``statspai-mcp`` / ``python -m statspai.agent.mcp_server``.

    ``--profile`` picks the ``tools/list`` shape (see :data:`_PROFILES`);
    the default is :data:`DEFAULT_PROFILE` (``curated``) because the full
    catalogue does not fit a client context window. Every function stays
    callable through ``call_function`` or by name.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="statspai-mcp",
        description="StatsPAI MCP server (JSON-RPC over stdio).",
        epilog=(
            "Environment: STATSPAI_MCP_DATA_ROOTS (readable directories), "
            "STATSPAI_MCP_ALLOW_REMOTE=1 (network data URLs), "
            "STATSPAI_MCP_MAX_DATA_BYTES, STATSPAI_MCP_MAX_OUTPUT_BYTES, "
            "STATSPAI_MCP_TOOL_TIMEOUT_SECONDS, STATSPAI_MCP_WORKERS, "
            "STATSPAI_MCP_MAX_QUEUED_CALLS, STATSPAI_MCP_MAX_ORPHANED_CALLS, "
            "STATSPAI_MCP_MAX_QUEUE_SECONDS, STATSPAI_MCP_MAX_REQUEST_BYTES, "
            "STATSPAI_MCP_ISOLATION=process (killable workers), "
            "STATSPAI_MCP_DATA_CACHE_SIZE / _BYTES, STATSPAI_MCP_DEBUG."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=list(_PROFILES),
        default=os.environ.get("STATSPAI_MCP_PROFILE", DEFAULT_PROFILE),
        help=_profile_help(),
    )
    args = parser.parse_args(argv)
    set_tool_profile(args.profile)
    serve_stdio()


__all__ = [
    "serve_stdio",
    "handle_request",
    "set_tool_profile",
    "DEFAULT_PROFILE",
    "tool_manifest",
    "MCP_PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "RESULT_SCHEMA_URI",
    "SERVER_NAME",
    "SERVER_VERSION",
]


if __name__ == "__main__":  # pragma: no cover
    main()
