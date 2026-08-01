"""Tests for Module F — composite pipeline tools (pipeline_did / iv / rd)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import execute_tool, mcp_handle_request


def _rpc(method, params=None, request_id=1):
    msg = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
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


def _staggered_panel(seed: int = 3):
    """Staggered adoption — the branch that reaches honest_did and bacon."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(80):
        cohort = int(rng.choice([0, 2005, 2008]))
        for year in range(2000, 2011):
            treated = 1 if cohort > 0 and year >= cohort else 0
            rows.append(
                (
                    unit,
                    year,
                    cohort,
                    1.0
                    + 0.2 * (year - 2000)
                    + 0.03 * unit
                    + (1.2 if treated else 0.0)
                    + rng.normal(),
                )
            )
    df = pd.DataFrame(rows, columns=["unit", "year", "cohort", "y"])
    df["treat"] = ((df.cohort > 0) & (df.year >= df.cohort)).astype(int)
    return df


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
        # "stages" existing says nothing about whether they ran. A stage
        # that failed still appears in the list.
        assert not [s for s in out["stages"] if s["status"] == "failed"], [
            s for s in out["stages"] if s["status"] == "failed"
        ]
        est_stage = next(s for s in out["stages"] if s["name"] == "estimate")
        assert est_stage["status"] == "ok"
        # The reported estimate must be the one sp.did computes.
        direct = sp.did(df, y="y", treat="treat", time="time")
        assert f"{float(direct.estimate):.4g}" in est_stage["summary"]
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
        out = execute_tool("pipeline_did", {"y": "y", "treat": "treat", "time": "time"})
        assert "error" in out

    def test_staggered_branch_runs_honest_did_and_bacon(self):
        """The cohort branch is what reaches the two conditional stages.

        Only the 2x2 path was covered, so honest_did and
        bacon_decomposition — the stages that make this a *reviewer-grade*
        DID pipeline rather than a wrapper around sp.did — were never
        exercised at all.
        """
        df = _staggered_panel()
        out = execute_tool(
            "pipeline_did",
            {
                "y": "y",
                "treat": "treat",
                "time": "year",
                "id": "unit",
                "cohort": "cohort",
            },
            data=df,
        )
        by_name = {s["name"]: s for s in out["stages"]}
        assert not [s for s in out["stages"] if s["status"] == "failed"], [
            s for s in out["stages"] if s["status"] == "failed"
        ]
        # A cohort column must dispatch Callaway-Sant'Anna, not 2x2 did.
        assert by_name["estimate"]["status"] == "ok"
        assert "callaway_santanna" in by_name["estimate"]["summary"]

        # Both conditional stages must actually run on this branch.
        assert by_name["honest_did"]["status"] == "ok", by_name["honest_did"]
        assert by_name["bacon_decomposition"]["status"] == "ok"

        # And report their headline numbers rather than the word
        # "computed", which told a reader nothing.
        assert "computed" not in by_name["honest_did"]["summary"]
        assert "M=" in by_name["honest_did"]["summary"]

        bacon_summary = by_name["bacon_decomposition"]["summary"]
        direct = sp.bacon_decomposition(
            df, y="y", treat="treat", time="year", id="unit"
        )
        assert f"{float(direct['negative_weight_share']):.3g}" in bacon_summary
        assert f"{int(direct['n_comparisons'])} 2x2" in bacon_summary
        assert f"{float(direct['beta_twfe']):.4g}" in bacon_summary

        # The narrative is what an agent pastes verbatim, so the numbers
        # have to reach it too.
        assert by_name["honest_did"]["summary"] in out["narrative"]
        assert bacon_summary in out["narrative"]

    def test_next_calls_carry_result_id(self):
        df = _toy_panel()
        out = execute_tool(
            "pipeline_did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
        )
        rid = out["result_id"]
        assert all(
            c["arguments"].get("result_id") == rid or "result_id" not in c["arguments"]
            for c in out["next_calls"]
        )


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
        # Three of these four stages used to fail on every run — the
        # diagnostics were handed the fitted result where they wanted
        # (data, column names) — while the assertions above stayed green.
        by_name = {s["name"]: s for s in out["stages"]}
        assert by_name["estimate"]["status"] == "ok"
        assert by_name["effective_f_test"]["status"] == "ok", by_name[
            "effective_f_test"
        ]
        assert by_name["anderson_rubin_test"]["status"] == "ok", by_name[
            "anderson_rubin_test"
        ]
        assert not [s for s in out["stages"] if s["status"] == "failed"], [
            s for s in out["stages"] if s["status"] == "failed"
        ]

        # The estimate stage must name the endogenous coefficient, not be
        # the empty "ivreg: " it used to print.
        beta = float(sp.ivreg("y ~ (d ~ z)", data=df).params["d"])
        assert f"{beta:.4g}" in by_name["estimate"]["summary"]

        # And the first-stage F must be the Olea-Pflueger effective F,
        # not the "F=nan" produced by reading attributes off a dict.
        f_eff = sp.effective_f_test(df, endog="d", instruments=["z"])["F_eff"]
        assert f"{f_eff:.2f}" in by_name["effective_f_test"]["summary"]


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
        by_name = {s["name"]: s for s in out["stages"]}
        assert not [s for s in out["stages"] if s["status"] == "failed"], [
            s for s in out["stages"] if s["status"] == "failed"
        ]
        for expected in ("estimate", "rddensity", "rdbwsensitivity", "rdplot"):
            assert expected in by_name, sorted(by_name)
            assert by_name[expected]["status"] == "ok"
        # rdplot advertises a PNG; make sure bytes actually came back.
        assert "PNG" in by_name["rdplot"]["summary"]
