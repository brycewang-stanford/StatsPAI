"""Every argument an MCP tool advertises must reach its function.

The server refuses a call whose argument the target cannot bind
(``unknown_argument``), so a schema property the function does not accept
is an argument the server invites and then rejects. Checked for the
curated tools and every auto-registered tool in the full profile. The
hand-written workflow and pipeline tools validate against their own
schemas; their consumption is covered by their own tests.
"""

from statspai.agent import mcp_server
from statspai.agent.auto_dispatch import _allowed_kwargs
from statspai.agent.pipeline_tools import PIPELINE_TOOL_NAMES
from statspai.agent.tools._dispatch import (
    _SERVER_HANDLED_ARGS,
    _bindable_names,
    _resolve_fn,
)
from statspai.agent.tools._specs import TOOL_REGISTRY
from statspai.agent.workflow_tools import WORKFLOW_TOOL_NAMES


def _unbound(props, names):
    return sorted(
        p
        for p in props
        if p not in names and p not in _SERVER_HANDLED_ARGS and p != "data"
    )


def test_curated_tool_properties_bind_to_their_function():
    bad = {}
    for spec in TOOL_REGISTRY:
        names = _bindable_names(_resolve_fn(spec["statspai_fn"]))
        if names is None:  # **kwargs: binds anything
            continue
        miss = _unbound(spec["input_schema"].get("properties", {}), names)
        if miss:
            bad[spec["name"]] = miss
    assert bad == {}


def test_auto_tool_properties_bind_to_their_function():
    import statspai as sp

    hand_written = (
        {t["name"] for t in TOOL_REGISTRY}
        | set(WORKFLOW_TOOL_NAMES)
        | set(PIPELINE_TOOL_NAMES)
    )
    bad = {}
    checked = 0
    for tool in mcp_server._build_mcp_tools("full"):
        name = tool["name"]
        if name in hand_written:
            continue
        fn = getattr(sp, name, None)
        assert callable(fn), f"advertised tool {name!r} has no sp.{name}"
        allowed = _allowed_kwargs(name, fn)
        if allowed is None:
            continue
        checked += 1
        miss = _unbound(tool["inputSchema"].get("properties", {}), allowed)
        if miss:
            bad[name] = miss
    assert bad == {}
    assert checked > 100  # the full profile advertises several hundred
