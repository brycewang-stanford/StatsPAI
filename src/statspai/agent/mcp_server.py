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

def _build_mcp_tools() -> List[Dict[str, Any]]:
    """Convert the StatsPAI agent-tool manifest into MCP tool specs.

    We add a mandatory ``data_path`` property so remote clients can
    point the estimator at a CSV on disk (the server can't receive
    DataFrames over JSON).
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
            required.append("data_path")
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
#  Resource: statspai://catalog
# ═══════════════════════════════════════════════════════════════════════

def _catalog_text() -> str:
    """Return a Markdown catalog of every StatsPAI tool."""
    manifest = tool_manifest()
    lines = [
        "# StatsPAI tool catalog",
        "",
        f"Version: {SERVER_VERSION}. {len(manifest)} tools registered.",
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


# ═══════════════════════════════════════════════════════════════════════
#  JSON-RPC handlers
# ═══════════════════════════════════════════════════════════════════════

def _handle_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
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
        raise ValueError("`name` is required")

    data_path = arguments.pop("data_path", None)
    df = None
    if data_path:
        df = _load_dataframe(data_path)

    result = execute_tool(name, arguments, data=df)
    # MCP content format: text + optional structured data
    text = json.dumps(result, indent=2, default=_json_default)
    return {
        "content": [
            {"type": "text", "text": text},
        ],
        "isError": bool(result.get("error")),
    }


def _handle_resources_list(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "resources": [
            {
                "uri": "statspai://catalog",
                "name": "StatsPAI estimator catalog",
                "mimeType": "text/markdown",
                "description": "Markdown list of every registered "
                               "StatsPAI estimator with its "
                               "description.",
            },
        ],
    }


def _handle_resources_read(params: Dict[str, Any]) -> Dict[str, Any]:
    uri = params.get("uri")
    if uri != "statspai://catalog":
        raise ValueError(f"Unknown resource: {uri!r}")
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": "text/markdown",
                "text": _catalog_text(),
            },
        ],
    }


_METHODS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
}


def handle_request(line: str) -> Optional[str]:
    """Process a single JSON-RPC request line; return the response line.

    Returns ``None`` for notifications (requests with no ``id``).
    """
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        return _jsonrpc_error(None, -32700, f"Parse error: {e}")

    request_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}

    # JSON-RPC notifications (no id) must not receive a response
    if request_id is None and "id" not in msg:
        return None

    handler = _METHODS.get(method)
    if handler is None:
        return _jsonrpc_error(request_id, -32601, f"Method not found: {method!r}")

    try:
        result = handler(params)
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
