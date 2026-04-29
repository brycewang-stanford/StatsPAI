"""JSON-schema tool definitions + dispatch for StatsPAI estimators.

Each tool specification follows the Anthropic / OpenAI tool-use
format: ``{'name': ..., 'description': ..., 'input_schema': {...}}``.
Agents built on top of Claude, GPT-4, etc. can use these directly
without wrapping.

Layout (sub-package since v1.11):

* ``_helpers``           — shared serializers (private; re-exported below
                            for backwards compatibility with existing
                            ``from .tools import _default_serializer``
                            imports in workflow_tools / auto_dispatch /
                            pipeline_tools / mcp_server).
* ``_specs/``            — declarative TOOL_REGISTRY entries split by
                            causal-inference family.
* ``_dispatch``          — ``tool_manifest`` + ``execute_tool``.

Public API (unchanged from v1.10):
"""
from __future__ import annotations

# Public surface — these three names are what every external caller
# imports. Keep the surface stable; rearranging internals must not
# break ``from statspai.agent.tools import tool_manifest``.
from ._dispatch import (  # noqa: F401
    tool_manifest,
    execute_tool,
    _resolve_fn,
)
from ._specs import TOOL_REGISTRY  # noqa: F401

# Legacy private re-exports. ``workflow_tools`` / ``auto_dispatch`` /
# ``pipeline_tools`` / ``mcp_server`` historically did
# ``from .tools import _default_serializer`` — preserve that path so
# the v1.11 split is purely internal reorganisation.
from ._helpers import (  # noqa: F401
    _scalar_or_none,
    _default_serializer,
    _identification_serializer,
)


__all__ = [
    "tool_manifest",
    "execute_tool",
    "TOOL_REGISTRY",
]
