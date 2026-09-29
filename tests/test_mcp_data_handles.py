"""Data handles, inline data and transform chains over MCP (roadmap W2).

Before these, the only way to hand a tool data was an absolute file path;
a frame derived in one step could not reach the next. Now:

* ``load_data`` returns a ``data_id`` every tool accepts in place of
  ``data_path``;
* ``data_records`` / ``data_csv`` carry a small table inline;
* ``transform_data`` derives a new handle and records the lineage, which
  rides in ``data_provenance`` on every result fitted from it;
* ``statspai://data/{id}`` reads a handle back.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.agent._data_cache import DATA_CACHE
from statspai.agent.mcp_server import handle_request


def _rpc(method: str, params: dict, request_id: int = 1) -> dict:
    raw = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    )
    response = handle_request(raw)
    assert response is not None
    return json.loads(response)


def _call(tool: str, **arguments) -> dict:
    msg = _rpc("tools/call", {"name": tool, "arguments": arguments})
    assert "result" in msg, msg
    return msg["result"]["structuredContent"]


@pytest.fixture
def panel_csv(tmp_path):
    rng = np.random.default_rng(0)
    n_units, n_periods = 40, 6
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), n_periods),
            "year": np.tile(np.arange(2000, 2000 + n_periods), n_units),
        }
    )
    df["g"] = np.where(df["id"] < 20, 2003, 0)
    df["post"] = ((df["g"] > 0) & (df["year"] >= df["g"])).astype(int)
    df["wage"] = 10 + 0.8 * df["post"] + 0.05 * df["id"] + rng.normal(size=len(df))
    df["wage"] = df["wage"].round(3)
    df.loc[df.sample(8, random_state=1).index, "wage"] = np.nan
    path = tmp_path / "panel.csv"
    df.to_csv(path, index=False)
    return str(path), df


class TestLoadAndDescribe:
    def test_load_returns_handle_and_profile(self, panel_csv):
        path, df = panel_csv
        out = _call("load_data", data_path=path, name="panel")
        assert out["data_id"].startswith("d_")
        assert out["n_rows"] == len(df) and out["n_cols"] == df.shape[1]
        assert out["missing"] == {"wage": 8}
        assert out["dtypes"]["year"].startswith("int")
        assert len(out["head"]) == 5
        assert out["numeric_summary"]["wage"]["mean"] == pytest.approx(
            float(df["wage"].mean()), rel=1e-9
        )
        assert out["provenance"]["source_type"] == "local"
        assert out["provenance"]["label"] == "panel"
        assert DATA_CACHE.get(out["data_id"]) is not None

    def test_describe_handle_includes_lineage(self, panel_csv):
        path, _ = panel_csv
        did = _call("load_data", data_path=path)["data_id"]
        out = _call("describe_data", data_id=did, head=2)
        assert out["data_id"] == did
        assert len(out["head"]) == 2
        assert out["lineage"][0]["data_id"] == did
        assert out["lineage"][0]["source_provenance"]["source_type"] == "local"

    def test_load_without_any_source_is_actionable(self):
        out = _call("load_data")
        assert (
            out["error"] == "load_data: expected argument `data_id`, got no arguments"
        )
        assert out["expected_argument"] == "data_id"
        assert out["got_arguments"] == []
        assert out["try"].startswith("load_data(data_id='d_")
        assert "data_records=[...]" in out["try"] and "data_csv=" in out["try"]


class TestInlineData:
    def test_records(self):
        rows = [{"y": 1.0 + 2 * i + (i % 3) * 0.1, "x": float(i)} for i in range(30)]
        out = _call("regress", formula="y ~ x", data_records=rows)
        assert out["coefficients"]["x"]["estimate"] == pytest.approx(2.0, abs=0.05)
        prov = out["data_provenance"]
        assert prov["source_type"] == "inline" and prov["format"] == "records"
        assert prov["n_rows"] == 30 and len(prov["sha256"]) == 64

    def test_csv_text(self):
        csv = "y,x\n" + "\n".join(f"{1 + 2 * i},{i}" for i in range(20))
        out = _call("regress", formula="y ~ x", data_csv=csv)
        assert out["coefficients"]["x"]["estimate"] == pytest.approx(2.0, abs=1e-9)
        assert out["data_provenance"]["format"] == "csv"

    def test_two_sources_rejected(self, panel_csv):
        path, _ = panel_csv
        msg = _rpc(
            "tools/call",
            {
                "name": "regress",
                "arguments": {
                    "formula": "wage ~ post",
                    "data_path": path,
                    "data_csv": "a,b\n1,2",
                },
            },
        )
        # A tool-execution failure: isError result, not a JSON-RPC error.
        assert msg["result"]["isError"] is True
        sc = msg["result"]["structuredContent"]
        assert sc["error_kind"] == "invalid_arguments"
        assert "exactly one data source" in sc["message"]

    def test_bad_records_rejected(self):
        msg = _rpc(
            "tools/call",
            {
                "name": "regress",
                "arguments": {"formula": "y ~ x", "data_records": [1, 2]},
            },
        )
        assert msg["result"]["isError"] is True
        assert "row objects" in msg["result"]["structuredContent"]["message"]


class TestTransformChain:
    def test_chain_and_fit_from_handle(self, panel_csv):
        path, df = panel_csv
        base = _call("load_data", data_path=path)["data_id"]
        derived = _call(
            "transform_data",
            data_id=base,
            name="analysis sample",
            operations=[
                {"op": "dropna", "columns": ["wage"]},
                {"op": "query", "expr": "year >= 2001"},
                {"op": "assign", "column": "lwage", "expr": "log(wage)"},
                {"op": "winsor", "columns": ["wage"], "cuts": [1, 99]},
            ],
        )
        assert derived["data_id"].startswith("d_") and derived["data_id"] != base
        assert derived["parent_id"] == base
        expected = df.dropna(subset=["wage"]).query("year >= 2001")
        assert derived["n_rows"] == len(expected)
        assert "lwage" in derived["columns"]
        ops = derived["operations"]
        assert [o["op"] for o in ops] == ["dropna", "query", "assign", "winsor"]
        assert ops[0]["n_rows_before"] == len(df)
        assert ops[0]["n_rows_after"] == len(df) - 8
        # The derived frame really is winsorised at the 1/99 percentiles.
        frame = DATA_CACHE.get(derived["data_id"])
        # sp.winsor uses Stata's percentile definition (winsor2 / _pctile).
        lo, hi = np.percentile(
            expected["wage"], [1, 99], method="averaged_inverted_cdf"
        )
        assert frame["wage"].min() >= lo - 1e-9 and frame["wage"].max() <= hi + 1e-9

        # Fit from the derived handle: provenance carries the lineage.
        fit = _call(
            "callaway_santanna",
            data_id=derived["data_id"],
            y="lwage",
            g="g",
            t="year",
            i="id",
            as_handle=True,
        )
        assert "error" not in fit, fit
        prov = fit["data_provenance"]
        assert prov["source_type"] == "handle"
        assert prov["data_id"] == derived["data_id"]
        assert [x["data_id"] for x in prov["lineage"]] == [derived["data_id"], base]
        assert prov["lineage"][0]["operations"][1]["op"] == "query"
        assert prov["root"]["source_type"] == "local"
        # And the same fit from a file with the same rows gives the same ATT.
        direct = _call(
            "callaway_santanna",
            data_records=frame.to_dict(orient="records"),
            y="lwage",
            g="g",
            t="year",
            i="id",
        )
        assert fit["estimate"] == pytest.approx(direct["estimate"], rel=1e-9)

    def test_failed_step_aborts_with_position(self, panel_csv):
        path, _ = panel_csv
        base = _call("load_data", data_path=path)["data_id"]
        out = _call(
            "transform_data",
            data_id=base,
            operations=[
                {"op": "query", "expr": "year >= 2001"},
                {"op": "select", "columns": ["wage", "nope"]},
            ],
        )
        assert "error" in out
        assert out["failed_step"] == 1
        assert "nope" in out["error"]
        assert out["applied"][0]["op"] == "query"
        assert out["degradation"]["section"].startswith("transform_data step 1")
        assert out["error_kind"] == "method_incompatibility"
        # Nothing was registered for the failed chain.
        assert not any(
            DATA_CACHE.get_entry(k).arguments.get("parent_id") == base
            for k in DATA_CACHE.keys()
            if DATA_CACHE.get_entry(k) is not None
        )

    def test_reshape_round_trip(self):
        rows = [
            {"id": 1, "y_2000": 1.0, "y_2001": 2.0},
            {"id": 2, "y_2000": 3.0, "y_2001": 4.0},
        ]
        base = _call("load_data", data_records=rows)["data_id"]
        long = _call(
            "transform_data",
            data_id=base,
            operations=[
                {
                    "op": "wide_to_long",
                    "stubnames": ["y"],
                    "i": "id",
                    "j": "year",
                    "sep": "_",
                }
            ],
        )
        assert long["n_rows"] == 4 and set(long["columns"]) == {"id", "year", "y"}
        wide = _call(
            "transform_data",
            data_id=long["data_id"],
            operations=[
                {"op": "long_to_wide", "index": "id", "columns": "year", "values": "y"}
            ],
        )
        assert wide["n_rows"] == 2 and "y_2000" in wide["columns"]
        assert wide["parent_id"] == long["data_id"]

    def test_function_op_requires_dataframe_result(self, panel_csv):
        path, _ = panel_csv
        base = _call("load_data", data_path=path)["data_id"]
        out = _call(
            "transform_data",
            data_id=base,
            operations=[
                {
                    "op": "function",
                    "name": "regress",
                    "arguments": {"formula": "wage ~ post"},
                }
            ],
        )
        assert "error" in out
        assert "not a DataFrame" in out["error"]


class TestHandleResource:
    def test_read_resource(self, panel_csv):
        path, _ = panel_csv
        did = _call("load_data", data_path=path)["data_id"]
        msg = _rpc("resources/read", {"uri": f"statspai://data/{did}"})
        body = json.loads(msg["result"]["contents"][0]["text"])
        assert body["data_id"] == did
        assert body["n_rows"] == 240
        assert body["lineage"][0]["data_id"] == did
        assert body["provenance"]["tool"] == "load_data"

    def test_template_listed(self):
        msg = _rpc("resources/templates/list", {})
        uris = {t["uriTemplate"] for t in msg["result"]["resourceTemplates"]}
        assert "statspai://data/{id}" in uris

    def test_missing_handle(self):
        msg = _rpc("resources/read", {"uri": "statspai://data/d_deadbeef"})
        assert msg["error"]["code"] == -32002
        call = _rpc(
            "tools/call",
            {
                "name": "regress",
                "arguments": {"formula": "y ~ x", "data_id": "d_deadbeef"},
            },
        )
        assert call["result"]["isError"] is True
        sc = call["result"]["structuredContent"]
        assert sc["error_kind"] == "missing_data_handle"
        assert sc["miss_reason"] == "unknown"
        assert "load_data" in sc["hint"]
        assert json.loads(call["result"]["content"][0]["text"]) == sc

    def test_evicted_handle_explains_itself(self, panel_csv):
        path, _ = panel_csv
        first = _call("load_data", data_path=path)["data_id"]
        for _ in range(DATA_CACHE._max_size):
            _call("load_data", data_records=[{"a": 1}])
        assert DATA_CACHE.get(first) is None
        msg = _rpc(
            "tools/call", {"name": "describe_data", "arguments": {"data_id": first}}
        )
        assert msg["result"]["isError"] is True
        sc = msg["result"]["structuredContent"]
        assert sc["error_kind"] == "missing_data_handle"
        assert sc["miss_reason"] == "lru"
        assert "evicted" in sc["hint"]
