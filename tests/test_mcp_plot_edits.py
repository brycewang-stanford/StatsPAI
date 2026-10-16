"""``plot_from_result(edits=...)``: an agent restyles a result's figure.

The edits are ``sp.interactive`` actions, so the figure editor's rules
apply (data locked, limits widened with a notice). The reply carries the
reproducible edit code and the pre-submission check next to the image.
"""

from __future__ import annotations

import json

import pytest

import statspai as sp
from statspai.agent import execute_tool, mcp_handle_request


@pytest.fixture(scope="module")
def rid():
    fit = execute_tool(
        "callaway_santanna",
        {"y": "lemp", "t": "year", "i": "countyreal", "g": "first_treat"},
        data=sp.datasets.mpdta(),
        as_handle=True,
    )
    return fit["result_id"]


def test_edits_are_applied_and_reported(rid):
    out = execute_tool(
        "plot_from_result",
        {
            "result_id": rid,
            "edits": [
                {"action": "set_title", "text": "Minimum wage"},
                {"action": "apply_theme", "theme": "aea"},
                {"action": "set_ylim", "bottom": -0.01, "top": 0.01},
            ],
        },
    )
    assert out["mime_type"] == "image/png" and out["image_bytes"] > 1000
    assert "ax.title.set_text('Minimum wage')" in out["edit_code"]
    assert "# theme 'aea'" in out["edit_code"]
    assert isinstance(out["check"], list)
    # the limits would have hidden the intervals: widened, and said so
    assert any("would hide data" in n for n in out["notices"])


def test_bad_edit_returns_an_error_and_no_image(rid):
    out = execute_tool(
        "plot_from_result",
        {"result_id": rid, "edits": [{"action": "delete_data"}]},
    )
    assert "edit not applied" in out["error"] and "set_title" in out["error"]
    assert "_plot_png" not in out
    out = execute_tool("plot_from_result", {"result_id": rid, "edits": "aea"})
    assert "list of action objects" in out["error"]


def test_without_edits_the_reply_is_unchanged(rid):
    out = execute_tool("plot_from_result", {"result_id": rid})
    assert {"edit_code", "check", "notices"}.isdisjoint(out)


def test_tools_list_and_call_over_mcp(rid):
    listed = json.loads(
        mcp_handle_request(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        )
    )
    tool = next(t for t in listed["result"]["tools"] if t["name"] == "plot_from_result")
    assert tool["inputSchema"]["properties"]["edits"]["items"] == {"type": "object"}
    reply = json.loads(
        mcp_handle_request(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "plot_from_result",
                        "arguments": {
                            "result_id": rid,
                            "edits": [
                                {"action": "apply_palette", "palette": "okabe_ito"}
                            ],
                        },
                    },
                }
            )
        )
    )
    content = reply["result"]["content"]
    assert any(c["type"] == "image" for c in content)
    text = next(c["text"] for c in content if c["type"] == "text")
    assert "set_color" in text or "set_facecolor" in text  # the recolour code


# ----------------------------------------------------------------------
# transform_data's mice op (found while checking this file's mypy output:
# it read a result attribute that does not exist and passed vars=, which
# sp.mice does not take, so both forms failed)
# ----------------------------------------------------------------------


def _with_gaps():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    df.loc[::7, "a"] = np.nan
    df.loc[::5, "b"] = np.nan
    return df


def test_transform_mice_fills_every_gap_reproducibly():
    import statspai.agent.workflow_tools as wt

    df = _with_gaps()
    filled = wt._apply_transform(df, {"op": "mice", "m": 2})
    assert filled.notna().all().all() and filled.shape == df.shape
    observed = df["a"].notna()
    assert (filled.loc[observed, "a"] == df.loc[observed, "a"]).all()
    assert filled.equals(wt._apply_transform(df, {"op": "mice", "m": 2}))
    out = execute_tool(
        "transform_data", {"operations": [{"op": "mice", "m": 2}]}, data=df
    )
    assert "error" not in out, out


def test_transform_mice_columns_fills_only_those():
    import statspai.agent.workflow_tools as wt

    df = _with_gaps()
    filled = wt._apply_transform(df, {"op": "mice", "columns": ["a"], "m": 2})
    assert filled["a"].notna().all()
    assert filled["b"].isna().sum() == df["b"].isna().sum()  # untouched
    observed = df["a"].notna()
    assert (filled.loc[observed, "a"] == df.loc[observed, "a"]).all()
    again = wt._apply_transform(df, {"op": "mice", "columns": ["a"], "m": 2})
    assert filled.equals(again)  # default seed: the same imputation
