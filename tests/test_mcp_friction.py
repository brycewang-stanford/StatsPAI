"""MCP entry points accept what agents actually send."""

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import mcp_server as M


@pytest.fixture()
def iv_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    z, u = rng.normal(size=n), rng.normal(size=n)
    t = 0.6 * z + u + rng.normal(size=n)
    df = pd.DataFrame({"y": 0.5 * t + u, "t": t, "z": z, "x1": rng.normal(size=n)})
    path = tmp_path / "iv.csv"
    df.to_csv(path, index=False)
    return str(path), df


def _call(name, args):
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args},
    }
    out = json.loads(M.handle_request(json.dumps(req)))["result"]
    text = "\n".join(c["text"] for c in out["content"] if c.get("type") == "text")
    return json.loads(text)


@pytest.mark.parametrize(
    "extra",
    [
        {"data_id": "None"},
        {"data_id": ""},
        {"data_id": "DATA_PATH"},
        {"data_records": []},
        {"data_csv": "placeholder"},
    ],
)
def test_placeholder_sources_do_not_conflict_with_a_real_one(iv_csv, extra):
    path, df = iv_csv
    out = _call("regress", {"formula": "y ~ t + x1", "data_path": path, **extra})
    assert "error" not in out
    ref = sp.regress("y ~ t + x1", data=df)
    assert out["coefficients"]["t"]["estimate"] == pytest.approx(float(ref.params["t"]))


def test_two_real_sources_still_conflict(iv_csv):
    path, _ = iv_csv
    out = _call(
        "regress",
        {"formula": "y ~ t", "data_path": path, "data_csv": "y,t\n1,2\n2,3\n"},
    )
    assert "exactly one data source" in out["error"]


def test_iv_preflight_accepts_named_parts_instead_of_a_formula(iv_csv):
    _, df = iv_csv
    out = sp.preflight(
        df, "iv", y="y", treatment="t", instrument="z", covariates=["x1"]
    )
    checks = {c["name"]: c["status"] for c in out["checks"]}
    assert checks["formula_columns_exist"] == "passed"
    bad = sp.preflight(df, "iv", y="y", treatment="t", instrument="nope")
    assert {c["name"]: c["status"] for c in bad["checks"]}[
        "formula_columns_exist"
    ] == "failed"
