"""Contract: every advertised workflow tool has a real dispatch branch.

``WORKFLOW_TOOL_SPECS`` is the single source of truth for the handle-based /
citation / smart-wrapper tools the MCP server exposes. ``WORKFLOW_TOOL_NAMES``
derives from it and ``execute_workflow_tool`` routes by name. The gap this guards:

``execute_workflow_tool`` falls through to a *silent error dict*
(``{'error': 'workflow_tool dispatch missed name ...'}``) — not an exception —
when a spec has no matching branch. The existing
``test_every_advertised_tool_is_executable`` only checks ``name in
WORKFLOW_TOOL_NAMES`` and therefore CANNOT catch a spec that was added without a
dispatch branch: it is "executable" by membership yet 500s for a real agent.

This module exercises the dispatcher for every spec name and asserts it routes to
a real handler, so adding a ``WORKFLOW_TOOL_SPECS`` entry without wiring
``execute_workflow_tool`` fails CI immediately.

Why not register these in the function registry instead? They are MCP-only tools
(no ``sp.<name>`` callable; they operate on result handles), so the single source
of truth correctly lives in ``WORKFLOW_TOOL_SPECS`` — putting them in the function
registry would pollute ``sp.list_functions()`` and inflate the public-symbol
count. The fix belongs at the manifest/dispatch layer, which is what this locks.
"""

import pytest

from statspai.agent.workflow_tools import (
    WORKFLOW_TOOL_SPECS,
    WORKFLOW_TOOL_NAMES,
    execute_workflow_tool,
)

_SPEC_NAMES = sorted(t["name"] for t in WORKFLOW_TOOL_SPECS)


def test_names_set_matches_specs():
    """``WORKFLOW_TOOL_NAMES`` must stay derived from the spec list."""
    assert WORKFLOW_TOOL_NAMES == frozenset(_SPEC_NAMES)


@pytest.mark.parametrize("name", _SPEC_NAMES)
def test_workflow_tool_has_dispatch_branch(name):
    """Routing must reach a real handler, never the 'dispatch missed' sentinel.

    We call with empty arguments and ``data=None``: a handler may legitimately
    raise or return its own validation error for missing inputs — that still
    proves the *branch exists*. The only failure mode we reject is the
    fall-through sentinel, which means the name has no wiring at all.
    """
    try:
        result = execute_workflow_tool(name, {}, data=None)
    except Exception:
        # An exception from the handler still means the branch was reached.
        return
    if isinstance(result, dict):
        err = result.get("error")
        assert not (isinstance(err, str) and "dispatch missed" in err), (
            f"workflow tool {name!r} is advertised in WORKFLOW_TOOL_SPECS but "
            f"execute_workflow_tool has no branch for it — an agent call would "
            f"500. Add a dispatch branch in workflow_tools.execute_workflow_tool."
        )


@pytest.mark.parametrize("spec", WORKFLOW_TOOL_SPECS, ids=_SPEC_NAMES)
def test_workflow_tool_spec_shape(spec):
    """Each spec carries a name, a description, and a JSON-object input_schema."""
    assert isinstance(spec.get("name"), str) and spec["name"]
    assert isinstance(spec.get("description"), str) and spec["description"]
    schema = spec.get("input_schema")
    assert isinstance(schema, dict), f"{spec['name']}: input_schema must be a dict"
    assert (
        schema.get("type") == "object"
    ), f"{spec['name']}: input_schema.type must be 'object'"
    # properties must be a dict when present; required ⊆ properties
    props = schema.get("properties")
    assert props is None or isinstance(props, dict)
    required = set(schema.get("required") or [])
    assert required.issubset(set((props or {}).keys())), (
        f"{spec['name']}: required params absent from properties: "
        f"{sorted(required - set((props or {}).keys()))}"
    )
