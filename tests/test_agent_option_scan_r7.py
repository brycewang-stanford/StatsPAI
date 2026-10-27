"""Option scan round 7: the agent tool layer (``statspai.agent``).

Each test pins an identity between a tool call and the library call it
stands for, or a hand-checkable answer on a small frame. Every defect
here was silent: the tool returned numbers, just not the ones asked for.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import execute_tool
from statspai.agent._data_loader import _uniform_sample
from statspai.agent.mcp_server import handle_request
from statspai.agent.tools._specs import TOOL_REGISTRY
from statspai.agent.workflow_tools import _apply_transform


def _call(tool: str, **arguments) -> dict:
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
    )
    msg = json.loads(handle_request(raw))
    assert "result" in msg, msg
    return msg["result"]["structuredContent"]


@pytest.fixture
def hetero():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.integers(0, 10, n)})
    df["y"] = 1 + 2 * df["x"] + rng.normal(size=n) * (1 + df["x"].abs())
    return df


# ---------------------------------------------------------------------------
# Advertised defaults must be the defaults the call runs with
# ---------------------------------------------------------------------------


def test_curated_schema_defaults_equal_library_defaults():
    """A schema ``default`` the library does not share makes the answer
    depend on the client: one that fills schema defaults gets HC1 SEs from
    ``regress``, one that does not gets classical SEs (0.135 vs 0.181 on
    the fixture below)."""
    mismatched = []
    for spec in TOOL_REGISTRY:
        fn = getattr(sp, spec["statspai_fn"])
        params = inspect.signature(fn).parameters
        aliases = getattr(fn, "__statspai_aliases__", {}) or {}
        for key, schema in spec["input_schema"].get("properties", {}).items():
            if "default" not in schema:
                continue
            p = params.get(key) or params.get(aliases.get(key, ""))
            if p is None:
                continue
            if p.default is inspect.Parameter.empty or p.default != schema["default"]:
                mismatched.append((spec["name"], key, schema["default"], p.default))
    assert mismatched == []


def test_regress_tool_without_robust_matches_library_default(hetero):
    out = execute_tool("regress", {"formula": "y ~ x"}, data=hetero)
    direct = sp.regress("y ~ x", data=hetero)
    assert out["coefficients"]["x"]["std_error"] == direct.std_errors["x"]
    hc1 = sp.regress("y ~ x", data=hetero, robust="hc1")
    out_hc1 = execute_tool(
        "regress", {"formula": "y ~ x", "robust": "hc1"}, data=hetero
    )
    assert out_hc1["coefficients"]["x"]["std_error"] == hc1.std_errors["x"]


# ---------------------------------------------------------------------------
# data_sample_n / data_columns apply to every data source
# ---------------------------------------------------------------------------


def test_data_sample_n_applies_to_handles_and_inline_tables(hetero):
    frame = hetero[["x", "y"]]
    records = frame.to_dict("records")
    sub = _uniform_sample(pd.DataFrame(records), 50)
    expect = sp.regress("y ~ x", data=sub).params["x"]

    inline = _call("regress", formula="y ~ x", data_records=records, data_sample_n=50)
    assert inline["n_obs"] == 50
    assert inline["coefficients"]["x"]["estimate"] == pytest.approx(expect, rel=1e-12)
    assert inline["data_provenance"]["sample_n"] == 50

    data_id = _call("load_data", data_records=records)["data_id"]
    via_handle = _call("regress", formula="y ~ x", data_id=data_id, data_sample_n=50)
    assert via_handle["n_obs"] == 50
    assert via_handle["coefficients"]["x"]["estimate"] == pytest.approx(
        expect, rel=1e-12
    )
    assert via_handle["data_provenance"]["sample_n"] == 50


def test_data_columns_applies_to_inline_tables(hetero):
    records = hetero[["x", "y"]].to_dict("records")
    out = _call("describe_data", data_records=records, data_columns=["x"])
    assert [c["name"] if isinstance(c, dict) else c for c in out["columns"]] == ["x"]


def test_load_data_on_a_sampled_handle_registers_the_sample(hetero):
    records = hetero[["x", "y"]].to_dict("records")
    data_id = _call("load_data", data_records=records)["data_id"]
    sampled = _call("load_data", data_id=data_id, data_sample_n=30)
    assert sampled["data_id"] != data_id
    assert sampled["n_rows"] == 30


def test_sampling_arguments_without_a_data_source_are_refused():
    out = _call("search_functions", query="did", data_sample_n=10)
    assert out.get("error_kind") == "invalid_arguments", out


# ---------------------------------------------------------------------------
# transform_data steps
# ---------------------------------------------------------------------------


@pytest.fixture
def grouped():
    return pd.DataFrame(
        {
            "g": [0] * 5 + [1] * 5,
            "w": [1, 2, 3, 4, 100, 10, 20, 30, 40, 1000.0],
        }
    )


def test_winsor_step_honours_by(grouped):
    got = _apply_transform(
        grouped, {"op": "winsor", "columns": ["w"], "cuts": [20, 80], "by": "g"}
    )
    want = sp.winsor(grouped, vars=["w"], cuts=(20, 80), by="g", replace=True)
    # by group: g=0 -> [1.5, 2, 3, 4, 52]; pooled cutoffs gave [2.5, ..., 70]
    assert got["w"].tolist() == want["w"].tolist()
    assert got["w"].tolist()[:5] == [1.5, 2.0, 3.0, 4.0, 52.0]


def test_winsor_step_honours_trim(grouped):
    got = _apply_transform(
        grouped, {"op": "winsor", "columns": ["w"], "cuts": [20, 80], "trim": True}
    )
    want = sp.winsor(grouped, vars=["w"], cuts=(20, 80), trim=True, replace=True)
    pd.testing.assert_series_equal(got["w"], want["w"])
    assert got["w"].isna().sum() == 4


def test_unknown_step_key_is_refused(grouped):
    from statspai.exceptions import MethodIncompatibility

    # ``colums`` (typo) used to be ignored, so dropna ran on every column.
    with pytest.raises(MethodIncompatibility, match="colums"):
        _apply_transform(grouped, {"op": "dropna", "colums": ["w"]})


def test_wide_to_long_with_text_suffix():
    wide = pd.DataFrame({"id": [1, 2], "y_a": [1, 2], "y_b": [3, 4]})
    got = _apply_transform(
        wide,
        {
            "op": "wide_to_long",
            "stubnames": ["y"],
            "i": "id",
            "j": "k",
            "sep": "_",
            "suffix": r"\w+",
        },
    )
    assert sorted(got["y"].tolist()) == [1, 2, 3, 4]


def test_wide_to_long_that_matches_nothing_fails_loudly():
    from statspai.exceptions import MethodIncompatibility

    wide = pd.DataFrame({"id": [1, 2], "y_a": [1, 2], "y_b": [3, 4]})
    # The default numeric suffix matches no column: this used to return an
    # empty frame (0 rows) as a valid new handle.
    with pytest.raises(MethodIncompatibility, match="suffix"):
        _apply_transform(
            wide,
            {"op": "wide_to_long", "stubnames": ["y"], "i": "id", "j": "k", "sep": "_"},
        )


# ---------------------------------------------------------------------------
# Composite and workflow tools refuse arguments they cannot honour
# ---------------------------------------------------------------------------


@pytest.fixture
def rd_frame():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"x": rng.uniform(-1, 1, n), "g": rng.integers(0, 20, n)})
    df["y"] = 0.5 * df["x"] + 1.0 * (df["x"] >= 0) + rng.normal(size=n)
    return df


def test_pipeline_rd_refuses_options_it_does_not_apply(rd_frame):
    # cluster= and p= used to be dropped: the report showed the default fit
    # (0.638, SE 0.552), not the requested one (0.699, SE 0.568).
    out = execute_tool(
        "pipeline_rd", {"y": "y", "x": "x", "cluster": "g", "p": 2}, data=rd_frame
    )
    assert out.get("error_kind") == "unknown_argument", out
    assert set(out["unknown_arguments"]) == {"cluster", "p"}


def test_workflow_tool_refuses_unknown_argument(hetero):
    fit = execute_tool("regress", {"formula": "y ~ x"}, data=hetero, as_handle=True)
    out = execute_tool("brief_result", {"result_id": fit["result_id"], "alpah": 0.1})
    assert out.get("error_kind") == "unknown_argument", out


@pytest.fixture(scope="module")
def cs_handle():
    df = sp.datasets.mpdta()
    fit = execute_tool(
        "callaway_santanna",
        {"y": "lemp", "g": "first_treat", "t": "year", "i": "countyreal"},
        data=df,
        as_handle=True,
    )
    direct = sp.callaway_santanna(
        df, y="lemp", g="first_treat", t="year", i="countyreal"
    )
    return fit["result_id"], direct


def test_honest_did_from_result_honours_alpha(cs_handle):
    rid, direct = cs_handle
    out = execute_tool(
        "honest_did_from_result",
        {"result_id": rid, "e": 0, "m_bar": 0.5, "alpha": 0.10},
    )
    want = sp.honest_did(direct, e=0, m_grid=[0.5], method="smoothness", alpha=0.10)
    assert out["rows"][0]["ci_lower"] == pytest.approx(want["ci_lower"].iloc[0])
    assert out["rows"][0]["ci_upper"] == pytest.approx(want["ci_upper"].iloc[0])


def test_honest_did_breakdown_is_none_when_nothing_rejects(cs_handle):
    rid, _ = cs_handle
    out = execute_tool("honest_did_from_result", {"result_id": rid, "e": 0, "m_bar": 5})
    assert not any(r["rejects_zero"] for r in out["rows"])
    # 0.0 claimed robustness up to M = 0, which was never evaluated.
    assert out["max_rejecting_M"] is None


def test_pipeline_did_cohort_without_id_is_refused():
    df = sp.datasets.mpdta()
    out = execute_tool(
        "pipeline_did",
        {"y": "lemp", "treat": "treat", "time": "year", "cohort": "first_treat"},
        data=df,
    )
    assert out.get("expected_argument") == "id", out


def test_deterministic_interpretation_reads_result_fields_and_level():
    from statspai.agent.workflow_tools import _deterministic_interpretation

    fields = {
        "estimate": 1.0,
        "se": 0.5,
        "ci": [0.2, 1.8],
        "pvalue": 0.04,
        "alpha": 0.1,
    }
    text = _deterministic_interpretation(object(), {"fields": fields})
    assert "standard error 0.5" in text
    # the interval is a 90% one; it used to be dropped (or labelled 95%)
    assert "90% confidence interval: [0.2, 1.8]" in text
    assert "p-value: 0.04" in text
