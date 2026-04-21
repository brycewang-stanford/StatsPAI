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

- A machine-readable schema of every estimator's parameters.
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
from .tools import tool_manifest, execute_tool, TOOL_REGISTRY
from .remediation import remediate, REMEDIATIONS
from .mcp_server import (
    serve_stdio as mcp_serve_stdio,
    handle_request as mcp_handle_request,
    SERVER_NAME as MCP_SERVER_NAME,
    SERVER_VERSION as MCP_SERVER_VERSION,
    MCP_PROTOCOL_VERSION,
)

__all__ = [
    'tool_manifest',
    'execute_tool',
    'TOOL_REGISTRY',
    'remediate',
    'REMEDIATIONS',
    'mcp_serve_stdio',
    'mcp_handle_request',
    'MCP_SERVER_NAME',
    'MCP_SERVER_VERSION',
    'MCP_PROTOCOL_VERSION',
]
