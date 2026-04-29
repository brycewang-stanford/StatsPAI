"""Tests for Module F — composite pipeline tools (pipeline_did / iv / rd).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statspai.agent import execute_tool, mcp_handle_request


def _rpc(method, params=None, request_id=1):
    msg = {"jsonrpc": "2.0", "id": request_id, "method": method,
           "params": params or {}}
    line = mcp_handle_request(json.dumps(msg))
    return json.loads(line)


def _toy_panel():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        treat = i % 2
        for t in (0, 1):
            y = 1.0 + 0.5 * t + 0.4 * treat * t + rng.normal(scale=0.1)
            rows.append({"id": i, "time": t, "treat": treat, "y": y})
    return pd.DataFrame(rows)


def _toy_iv():
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.6 * z + 0.4 * u + rng.normal(size=n) * 0.5
    y = 1.0 + 0.7 * d + 0.5 * u + rng.normal(size=n) * 0.5
    return pd.DataFrame({"y": y, "d": d, "z": z})


def _toy_rd():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-1, 1, n)
    treat = (x >= 0).astype(int)
    y = 1.0 + 0.4 * treat + 0.3 * x + rng.normal(size=n) * 0.5
    return pd.DataFrame({"y": y, "x": x})


# ----------------------------------------------------------------------
# Manifest entries
# ----------------------------------------------------------------------

class TestPipelineManifest:
    def test_pipeline_tools_listed(self):
        msg = _rpc("tools/list", {})
        names = {t["name"] for t in msg["result"]["tools"]}
        assert "pipeline_did" in names
        assert "pipeline_iv" in names
        assert "pipeline_rd" in names


# ----------------------------------------------------------------------
# pipeline_did
# ----------------------------------------------------------------------

class TestPipelineDID:
    def test_basic_workflow(self):
        df = _toy_panel()
        out = execute_tool(
            "pipeline_did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
        )
        assert out.get("pipeline") == "pipeline_did"
        assert "result_id" in out
        assert "stages" in out
        # estimate stage should succeed
        names = [s["name"] for s in out["stages"]]
        assert "estimate" in names
        # narrative is markdown with a header
        assert out["narrative"].startswith("# DID workflow")

    def test_missing_required_args(self):
        df = _toy_panel()
        out = execute_tool("pipeline_did", {"y": "y"}, data=df)
        assert "error" in out

    def test_no_data_returns_error(self):
        out = execute_tool("pipeline_did",
                            {"y": "y", "treat": "treat", "time": "time"})
        assert "error" in out

    def test_next_calls_carry_result_id(self):
        df = _toy_panel()
        out = execute_tool(
            "pipeline_did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
        )
        rid = out["result_id"]
        assert all(c["arguments"].get("result_id") == rid
                   or "result_id" not in c["arguments"]
                   for c in out["next_calls"])


# ----------------------------------------------------------------------
# pipeline_iv
# ----------------------------------------------------------------------

class TestPipelineIV:
    def test_basic_workflow(self):
        df = _toy_iv()
        out = execute_tool(
            "pipeline_iv",
            {"formula": "y ~ (d ~ z)"},
            data=df,
        )
        if "error" in out and "estimator" in out["error"]:
            pytest.skip("ivreg unavailable in this build")
        assert out["pipeline"] == "pipeline_iv"
        assert "result_id" in out
        assert "stages" in out


# ----------------------------------------------------------------------
# pipeline_rd
# ----------------------------------------------------------------------

class TestPipelineRD:
    def test_basic_workflow(self):
        df = _toy_rd()
        out = execute_tool(
            "pipeline_rd",
            {"y": "y", "x": "x", "c": 0.0},
            data=df,
        )
        if "error" in out and "rdrobust" in out["error"]:
            pytest.skip("rdrobust unavailable in this build")
        assert out["pipeline"] == "pipeline_rd"
        assert "result_id" in out
