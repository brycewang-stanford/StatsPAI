"""LLM agent-native tool-definition surface.

StatsPAI's agent-native branding delivers a concrete API here:

1. ``tool_manifest()`` — returns a list of JSON-schema tool specs in
   the OpenAI / Anthropic tool-use format; drop straight into an
   agent's ``tools=`` parameter and the model can call StatsPAI
   estimators directly.

2. ``execute_tool(name, arguments, data)`` — single entry point that
   dispatches a tool call back to the right StatsPAI function,
   serialises the result in a JSON-friendly dict.

3. ``remediate(error, context)`` — error → actionable-fix registry.
   Given a Python exception from a failed tool call, returns a
   structured suggestion the agent can use to repair its next call.

Why this matters
----------------
An LLM that wants to call StatsPAI programmatically needs three
things docstrings don't provide:

- Machine-readable schemas for registered StatsPAI functions.
- A round-trippable result representation.
- A mapping from errors to concrete next steps.

This module provides all three as a coherent layer so an agent built
on top of StatsPAI can loop "try → fail → remediate" deterministically.

Usage
-----
With Anthropic's Claude tool-use API::

    import anthropic, statspai as sp
    client = anthropic.Anthropic()
    response = client.messages.create(
        model='claude-opus-4-7',
        tools=sp.agent.tool_manifest(),
        messages=[{'role': 'user', 'content': 'Run a DID on ...'}],
    )
    for block in response.content:
        if block.type == 'tool_use':
            out = sp.agent.execute_tool(
                block.name, block.input, data=my_df,
            )

"""

from typing import Any

from .remediation import REMEDIATIONS, remediate
from .tools import TOOL_REGISTRY, execute_tool, tool_manifest

#: MCP server names re-exported lazily. Importing ``mcp_server`` here made
#: every ``import statspai`` load the whole server, and made
#: ``python -m statspai.agent.mcp_server`` warn that the module was already
#: in ``sys.modules`` before it ran as ``__main__``.
_MCP_EXPORTS = {
    "MCP_PROTOCOL_VERSION": "MCP_PROTOCOL_VERSION",
    "MCP_SERVER_NAME": "SERVER_NAME",
    "MCP_SERVER_VERSION": "SERVER_VERSION",
    "mcp_handle_request": "handle_request",
    "mcp_serve_http": "serve_http",
    "mcp_serve_stdio": "serve_stdio",
}


def __getattr__(name: str) -> Any:
    target = _MCP_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import mcp_server

    value = getattr(mcp_server, target)
    globals()[name] = value
    return value


__all__ = [
    "tool_manifest",
    "execute_tool",
    "TOOL_REGISTRY",
    "remediate",
    "REMEDIATIONS",
    "mcp_serve_stdio",
    "mcp_serve_http",
    "mcp_handle_request",
    "MCP_SERVER_NAME",
    "MCP_SERVER_VERSION",
    "MCP_PROTOCOL_VERSION",
]
