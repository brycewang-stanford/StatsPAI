"""``tools/list`` size budget and the shared-argument contract.

The arguments the server handles for every tool (data source, handles,
output controls) are repeated in each advertised schema. With long
descriptions they were two thirds of the list: 146 KiB under the
``curated`` profile, 82 KiB under ``core``. They now carry one clause
each and the long form is served once as ``statspai://arguments``.

The budgets below are ratchets with headroom for a few new tools, not
targets: a jump past them means a per-tool cost crept back in.
"""

from __future__ import annotations

import json

import pytest

from statspai.agent import mcp_server as m

# Measured 2026-10-14 after compaction: curated 71 KiB, core 39 KiB.
# 2026-10-16: core gained load_data / describe_data / transform_data (the
# session instructions start with them); shorter shared clauses paid for
# it. core 46,802 B with the committed snapshot, 46,991 B live
# (STATSPAI_MCP_SCHEMA_SNAPSHOT=0); curated 73,523 / 73,758 B.
# 2026-10-19: curated gained feols / iv / match / sun_abraham / sdid and
# dropped nbreg / xtnbreg (still callable by name): 81,574 -> 87,014 B live
# (42 -> 45 tools); the budget moves with it, the per-tool cost does not.
BUDGET_BYTES = {"curated": 88 * 1024, "core": 46 * 1024}


def _rpc(method, params=None):
    line = m.handle_request(
        json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        )
    )
    return json.loads(line)


@pytest.mark.parametrize("profile", sorted(BUDGET_BYTES))
def test_tool_list_fits_its_budget(profile):
    size = len(json.dumps(m._build_mcp_tools(profile)))
    assert (
        size <= BUDGET_BYTES[profile]
    ), f"{profile} tools/list is {size} bytes; budget {BUDGET_BYTES[profile]}"


def test_shared_arguments_cost_under_a_kilobyte_per_tool():
    shared = m._common_argument_schemas(remote_ok=False)
    assert set(shared) == set(m._COMMON_ARGUMENT_ORDER)
    on_every_tool = {k: v for k, v in shared.items() if k not in m._LOAD_ONLY_ARGUMENTS}
    assert len(json.dumps(on_every_tool)) < 1024
    for name, schema in shared.items():
        assert 0 < len(schema["description"]) <= 64, name


def test_every_tool_carries_the_shared_arguments():
    always = set(m._COMMON_ARGUMENT_ORDER) - m._LOAD_ONLY_ARGUMENTS
    for tool in m._build_mcp_tools("curated"):
        props = tool["inputSchema"]["properties"]
        assert always <= set(props), tool["name"]
        listed = m._LOAD_ONLY_ARGUMENTS <= set(props)
        assert listed == (tool["name"] in m._LOAD_ARGUMENT_TOOLS), tool["name"]


def test_load_only_arguments_still_work_on_a_tool_that_does_not_list_them(tmp_path):
    path = tmp_path / "d.csv"
    rows = ["y,x,junk"] + [f"{2 * i + (i % 3)},{i},{i * i}" for i in range(40)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert (
        "data_columns"
        not in next(t for t in m._build_mcp_tools("curated") if t["name"] == "regress")[
            "inputSchema"
        ]["properties"]
    )
    out = _rpc(
        "tools/call",
        {
            "name": "regress",
            "arguments": {
                "data_path": str(path),
                "formula": "y ~ x",
                "data_columns": ["y", "x"],
                "data_sample_n": 30,
            },
        },
    )["result"]
    assert not out.get("isError"), out
    payload = out["structuredContent"]
    assert payload["n_obs"] == 30
    assert payload["data_provenance"]["columns_requested"] == ["y", "x"]


def test_long_form_reference_covers_every_shared_argument():
    listed = {r["uri"] for r in _rpc("resources/list")["result"]["resources"]}
    assert m.COMMON_ARGUMENTS_URI in listed
    text = _rpc("resources/read", {"uri": m.COMMON_ARGUMENTS_URI})["result"][
        "contents"
    ][0]["text"]
    for name in m._COMMON_ARGUMENT_ORDER:
        assert f"`{name}`" in text, name
    # What the one-clause schemas no longer say must be findable here.
    for fact in (".parquet", "seed=0", "truncated", "risk_summary", "'minimal'"):
        assert fact in text, fact


def test_session_instructions_point_at_the_reference():
    text = _rpc("initialize", {"protocolVersion": m.MCP_PROTOCOL_VERSION})["result"][
        "instructions"
    ]
    assert m.COMMON_ARGUMENTS_URI in text
    assert "data_sample_n" in text
