"""JSON-RPC error taxonomy for the MCP server.

Lives in its own module so `_resources` / `_prompts` / future split
files can raise the same typed errors without forming an import cycle
through `mcp_server`.
"""
from __future__ import annotations


class RpcError(Exception):
    """Internal exception carrying an explicit JSON-RPC error code.

    JSON-RPC 2.0 reserves ``-32xxx`` codes; MCP 2024-11-05 names
    ``-32002`` for resource-not-found. Using untyped ValueError + a
    blanket ``-32000`` would force MCP clients to regex the message to
    decide whether to retry, prompt the user, or surface a friendly
    error — typing the exception keeps the protocol semantically rich.
    """

    code: int = -32000  # generic server-defined error


class InvalidParamsError(RpcError):
    """``-32602`` per JSON-RPC 2.0 — caller-supplied params are wrong."""

    code = -32602


class ResourceNotFoundError(RpcError):
    """``-32002`` per MCP 2024-11-05 — URI does not resolve to a resource."""

    code = -32002


__all__ = ["RpcError", "InvalidParamsError", "ResourceNotFoundError"]
