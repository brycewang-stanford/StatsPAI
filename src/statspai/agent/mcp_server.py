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
    """Best-effort JSON encoder for numpy / pandas scalars."""
    try:
        import numpy as _np
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as _pd
        if isinstance(o, (_pd.Series, _pd.DataFrame)):
            return o.to_dict()
        if isinstance(o, _pd.Timestamp):
            return o.isoformat()
    except ImportError:  # pragma: no cover
        pass
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
#: the estimator never reads. Keep this set in sync with the
#: ``TOOL_REGISTRY`` in ``agent/tools.py``.
_DATALESS_TOOLS = frozenset({"honest_did", "sensitivity"})


def _build_mcp_tools() -> List[Dict[str, Any]]:
    """Convert the StatsPAI agent-tool manifest into MCP tool specs.

    We inject two server-handled arguments into every tool's schema
    so the LLM can supply them via the standard ``tools/call``
    arguments object:

    * ``data_path`` (required) — absolute path to a CSV/Parquet/etc.
      file the server loads into a DataFrame before dispatching.
    * ``detail`` (optional, default ``"agent"``) — payload depth,
      forwarded to ``result.to_dict(detail=...)`` so the LLM can pick
      ``"minimal"`` for cheap sub-step calls or ``"standard"`` /
      ``"agent"`` when violations + next-step hints should ride along.
    """
    manifest = tool_manifest()
    out: List[Dict[str, Any]] = []
    for t in manifest:
        schema = dict(t.get("input_schema") or {})
        props = dict(schema.get("properties") or {})
        required = list(schema.get("required") or [])
        if "data_path" not in props:
            props["data_path"] = {
                "type": "string",
                "description": "Absolute path to a CSV file on the "
                               "local filesystem; the server will "
                               "load it into a DataFrame.",
            }
            # Mark required ONLY for tools whose underlying function
            # actually takes a DataFrame; dataless tools (honest_did,
            # sensitivity, …) leave ``data_path`` optional so strict-
            # schema MCP clients don't refuse to dispatch them.
            if t["name"] not in _DATALESS_TOOLS:
                required.append("data_path")
        if "detail" not in props:
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


def _load_dataframe(path: str):
    if not os.path.isabs(path):
        raise ValueError(f"data_path must be absolute, got {path!r}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
    import pandas as pd
    lower = path.lower()
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if lower.endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith(".json"):
        return pd.read_json(path)
    raise ValueError(
        f"Unsupported file extension: {path!r}. Use .csv/.tsv/.parquet/.xlsx/.json."
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
    # estimator's signature has no ``data_path`` / ``detail`` and would
    # crash with a "got an unexpected keyword argument" error.
    data_path = arguments.pop("data_path", None)
    df = None
    if data_path:
        df = _load_dataframe(data_path)

    detail = arguments.pop("detail", "agent")
    if detail not in _DETAIL_LEVELS:
        raise _InvalidParamsError(
            "detail must be one of "
            f"{', '.join(repr(v) for v in _DETAIL_LEVELS)}; "
            f"got {detail!r}"
        )

    result = execute_tool(name, arguments, data=df, detail=detail)
    # MCP content format: text + optional structured data
    text = json.dumps(result, indent=2, default=_json_default)
    return {
        "content": [
            {"type": "text", "text": text},
        ],
        "isError": bool(result.get("error")),
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
        "description": "Run a DID estimator on a CSV, surface the "
                       "estimate, and walk through every "
                       "reviewer-checklist gap (parallel-trends test, "
                       "honest-DID sensitivity, Bacon decomposition, "
                       "placebo).",
        "arguments": [
            {"name": "data_path",
             "description": "Absolute path to a CSV file.",
             "required": True},
            {"name": "y", "description": "Outcome column.",
             "required": True},
            {"name": "treat",
             "description": "Binary 0/1 treatment column.",
             "required": True},
            {"name": "time", "description": "Time column.",
             "required": True},
        ],
        "_template": (
            "1. Call ``preflight`` with method='did' and the supplied "
            "columns; if the verdict is FAIL, stop and report the "
            "failed checks.\n"
            "2. Call the ``did`` tool with data_path={data_path}, "
            "y={y}, treat={treat}, time={time}.\n"
            "3. Call ``audit`` on the result and list every check "
            "with status='missing' AND importance='high'. For each, "
            "call the function in ``suggest_function``.\n"
            "4. Summarise: estimate, 95% CI, the violations table, "
            "and the BibTeX citation."
        ),
    },
    {
        "name": "design_then_estimate",
        "description": "Given an unfamiliar CSV, auto-detect the "
                       "study design, choose an appropriate estimator, "
                       "and run it with reviewer-grade diagnostics.",
        "arguments": [
            {"name": "data_path",
             "description": "Absolute path to the CSV file.",
             "required": True},
            {"name": "outcome",
             "description": "Outcome column the analyst cares about.",
             "required": True},
            {"name": "treatment",
             "description": "Treatment / exposure column.",
             "required": False},
        ],
        "_template": (
            "1. Read {data_path} and call ``detect_design`` to "
            "identify the study shape (panel / cross-section / RD).\n"
            "2. Call ``recommend`` with outcome={outcome} (and "
            "treatment={treatment} when supplied) to pick an "
            "estimator. Read the top recommendation's reasoning.\n"
            "3. Call ``preflight`` for the chosen estimator on this "
            "data; only proceed if verdict is PASS or WARN.\n"
            "4. Run the estimator. Then call ``audit`` and surface "
            "any high-importance missing robustness checks."
        ),
    },
    {
        "name": "robustness_followup",
        "description": "Take an existing fitted result and run the "
                       "high-importance follow-up sensitivities the "
                       "audit identifies as missing.",
        "arguments": [
            {"name": "result_summary",
             "description": "Pasted output of ``result.summary()`` "
                            "or the JSON ``to_dict(detail='agent')``.",
             "required": True},
        ],
        "_template": (
            "Given the result described below:\n\n"
            "{result_summary}\n\n"
            "1. Inspect the violations / next_steps fields.\n"
            "2. For each violation with severity='error', run the "
            "``suggest_function`` with the same data.\n"
            "3. For each missing high-importance audit check, run "
            "the suggested function.\n"
            "4. Rebuild a one-line ``brief`` for each follow-up "
            "result and report which (if any) overturn the original "
            "conclusion."
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
        return _jsonrpc_error(
            request_id, -32000, f"{type(exc).__name__}: {exc}",
            data={"traceback": traceback.format_exc()},
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
