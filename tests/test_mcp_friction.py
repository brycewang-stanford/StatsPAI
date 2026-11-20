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


def test_explicit_cs_reads_a_treatment_indicator_like_auto_does():
    rng = np.random.default_rng(0)
    n, T, g = 200, 6, 3
    tr = np.arange(n) < n // 2
    rows = [
        pd.DataFrame(
            {
                "unit": np.arange(n),
                "time": t,
                "treated": (tr & (t >= g)).astype(int),
                "first_treat": np.where(tr, g, 0),
                "y": rng.normal(size=n) + 1.0 * (tr & (t >= g)),
            }
        )
        for t in range(T)
    ]
    df = pd.concat(rows, ignore_index=True)
    ref = sp.did(df, y="y", treat="first_treat", time="time", id="unit", method="cs")
    got = sp.did(df, y="y", treat="treated", time="time", id="unit", method="cs")
    assert got.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert got.se == pytest.approx(ref.se, rel=1e-12)


def _dirty(n=3000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(50, 10, n),
            "t": rng.integers(0, 2, n),
            "x": rng.normal(size=n),
        }
    )
    clean = df.copy()
    df.loc[:99, "y"] = -99.0
    return pd.concat([df, df.iloc[200:260]], ignore_index=True), clean


def test_preflight_flags_missing_value_codes_and_duplicate_records():
    dirty, clean = _dirty()
    out = sp.preflight(dirty, "regress", formula="y ~ t + x")
    warns = {w["check"]: w for w in out["data_warnings"]}
    assert out["verdict"] == "WARN"
    assert warns["sentinel_codes"]["evidence"]["columns"]["y"] == {
        "value": -99.0,
        "n_rows": 100,
    }
    assert warns["duplicate_rows"]["evidence"]["n_duplicates"] == 60
    ok = sp.preflight(clean, "regress", formula="y ~ t + x")
    assert ok["data_warnings"] == [] and ok["verdict"] == "PASS"


def test_a_real_99_inside_the_scale_is_not_a_code():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"score": rng.integers(0, 101, 5000).astype(float)})
    from statspai.smart.preflight import data_hygiene_warnings

    assert data_hygiene_warnings(df) == []


def test_estimator_reply_leads_with_the_data_warnings(tmp_path):
    dirty, _ = _dirty()
    path = tmp_path / "d.csv"
    dirty.to_csv(path, index=False)
    out = _call("regress", {"formula": "y ~ t + x", "data_path": str(path)})
    assert next(iter(out)) == "data_warnings"
    assert {w["check"] for w in out["data_warnings"]} == {
        "sentinel_codes",
        "duplicate_rows",
    }
