"""Regression tests for the 2026-09 agent-native audit fixes.

Each test pins one gap that made StatsPAI hard (or silently wrong) to
drive from an LLM agent:

* the MCP ``tools/list`` payload (~2 MB under the historical full
  profile) now has ``core`` / ``curated`` profiles plus three discovery
  meta-tools, so a client that sees ~35 tools can still reach every
  registered function;
* the auto-dispatch path no longer drops unknown arguments silently;
* estimator warnings and nested diagnostics reach the agent payload, and
  a crashing diagnostic is recorded under ``degradations`` instead of
  reading as "clean";
* ``sp.did`` / ``sp.rd`` accept the spellings agents guess
  (``outcome=`` / ``treatment=`` / ``cutoff=``), ``sp.did(method='auto')``
  handles a 0/1 indicator on a multi-period panel, and unknown keywords
  get a did-you-mean message instead of "missing positional arguments";
* ``sp.search_functions`` answers natural-language task queries;
* remediation hints only reference functions that exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import mcp_server
from statspai.agent.mcp_server import handle_request
from statspai.exceptions import MethodIncompatibility
from statspai.workflow._degradation import WorkflowDegradedWarning


def _rpc(method: str, params: dict, request_id: int = 1) -> dict:
    raw = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    )
    response = handle_request(raw)
    assert response is not None
    return json.loads(response)


@pytest.fixture
def profile_reset():
    yield
    mcp_server.set_tool_profile(None)


@pytest.fixture
def panel_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_units, n_periods = 50, 8
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), n_periods),
            "t": np.tile(np.arange(n_periods), n_units),
        }
    )
    df["treat"] = ((df["id"] < 25) & (df["t"] >= 4)).astype(int)
    df["y"] = rng.normal(size=len(df)) + 0.5 * df["treat"] + 0.01 * df["id"]
    return df


# ---------------------------------------------------------------------------
# MCP: tool-list profiles + discovery meta-tools
# ---------------------------------------------------------------------------


class TestToolProfiles:
    META = {"search_functions", "describe_function", "call_function"}

    def test_curated_profile_fits_a_context_window(self, profile_reset):
        mcp_server.set_tool_profile("curated")
        msg = _rpc("tools/list", {})
        tools = msg["result"]["tools"]
        names = {t["name"] for t in tools}
        assert self.META <= names
        assert {"did", "callaway_santanna", "rdrobust", "ivreg", "regress"} <= names
        payload = json.dumps(msg)
        # ~100 KB, versus ~2 MB for the full catalogue.
        assert len(payload) < 300_000, len(payload)
        assert len(tools) < 80

    def test_core_profile_is_a_subset_of_curated(self, profile_reset):
        mcp_server.set_tool_profile("core")
        core = {t["name"] for t in _rpc("tools/list", {})["result"]["tools"]}
        mcp_server.set_tool_profile("curated")
        curated = {t["name"] for t in _rpc("tools/list", {})["result"]["tools"]}
        assert self.META <= core <= curated
        assert len(core) < len(curated)

    def test_full_profile_still_advertises_meta_tools(self, profile_reset):
        mcp_server.set_tool_profile("full")
        names = {t["name"] for t in _rpc("tools/list", {})["result"]["tools"]}
        assert self.META <= names
        assert len(names) > 400

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError, match="Unknown MCP tool profile"):
            mcp_server.set_tool_profile("everything")

    def test_tools_call_reaches_unlisted_tool_under_curated(self, profile_reset):
        # The profile shapes the advertised list only; dispatch stays open.
        mcp_server.set_tool_profile("curated")
        listed = {t["name"] for t in _rpc("tools/list", {})["result"]["tools"]}
        assert "adjust_pvalues" not in listed
        msg = _rpc(
            "tools/call",
            {"name": "adjust_pvalues", "arguments": {"pvalues": [0.01, 0.04]}},
        )
        assert "result" in msg
        assert msg["result"]["isError"] is False


class TestDiscoveryMetaTools:
    def test_search_then_describe_then_call(self, tmp_path):
        hits = _rpc(
            "tools/call",
            {
                "name": "search_functions",
                "arguments": {
                    "query": "multiple testing p-value adjustment",
                    "limit": 5,
                },
            },
        )["result"]["structuredContent"]
        assert hits["n_matches"] >= 1
        names = [h["name"] for h in hits["matches"]]
        assert "adjust_pvalues" in names

        desc = _rpc(
            "tools/call",
            {"name": "describe_function", "arguments": {"name": "sp.adjust_pvalues"}},
        )["result"]["structuredContent"]
        assert desc["name"] == "adjust_pvalues"
        assert desc["schema"]["parameters"]["type"] == "object"
        assert desc["call_with"]["tool"] == "call_function"

        out = _rpc(
            "tools/call",
            {
                "name": "call_function",
                "arguments": {
                    "function": "adjust_pvalues",
                    "arguments": {"pvalues": [0.01, 0.04], "method": "holm"},
                },
            },
        )["result"]["structuredContent"]
        assert "error" not in out
        assert out["called_via"] == "call_function"

    def test_describe_unknown_name_suggests(self):
        out = _rpc(
            "tools/call",
            {"name": "describe_function", "arguments": {"name": "callaway_santana"}},
        )["result"]["structuredContent"]
        assert "error" in out
        assert "callaway_santanna" in out["did_you_mean"]

    def test_call_function_with_data(self, tmp_path, panel_df):
        path = tmp_path / "panel.csv"
        panel_df.to_csv(path, index=False)
        out = _rpc(
            "tools/call",
            {
                "name": "call_function",
                "arguments": {
                    "function": "regress",
                    "arguments": {"formula": "y ~ treat"},
                    "data_path": str(path),
                },
            },
        )["result"]["structuredContent"]
        assert "error" not in out, out
        assert out["called_via"] == "call_function"


# ---------------------------------------------------------------------------
# MCP: nothing silent — unsupported args, runtime warnings, stdout
# ---------------------------------------------------------------------------


class TestNothingSilent:
    def test_auto_dispatch_reports_unsupported_args(self):
        from statspai.agent.tools import execute_tool

        out = execute_tool(
            "adjust_pvalues",
            {"pvalues": [0.01, 0.04], "clusterr": "id"},
        )
        assert out.get("_unsupported_args") == ["clusterr"]

    def test_auto_dispatch_accepts_signature_params_missing_from_registry(self):
        # Registry entries can lag the live signature; the live signature
        # must win so a legitimate argument is never reported as unknown.
        from statspai.agent.auto_dispatch import _allowed_kwargs

        allowed = _allowed_kwargs("did", sp.did)
        assert allowed is None or {"covariates", "cluster", "outcome"} <= allowed

    def test_runtime_warnings_reach_the_payload(self, tmp_path, panel_df):
        path = tmp_path / "panel.csv"
        panel_df.to_csv(path, index=False)
        # method='twfe' on 8 periods collapses to pre/post and warns.
        msg = _rpc(
            "tools/call",
            {
                "name": "did",
                "arguments": {
                    "y": "y",
                    "treat": "treat",
                    "time": "t",
                    "method": "twfe",
                    "data_path": str(path),
                },
            },
        )
        payload = msg["result"]["structuredContent"]
        cats = {w["category"] for w in payload.get("runtime_warnings", [])}
        assert "AssumptionWarning" in cats, payload.get("runtime_warnings")

    def test_estimator_prints_do_not_reach_stdout(self, monkeypatch, capsys):
        def _noisy(name, arguments, **kw):
            print("THIS WOULD CORRUPT JSON-RPC")
            return {"ok": True}

        monkeypatch.setattr(mcp_server, "execute_tool", _noisy)
        msg = _rpc("tools/call", {"name": "bibtex", "arguments": {"keys": ["x"]}})
        assert msg["result"]["structuredContent"] == {"ok": True}
        captured = capsys.readouterr()
        assert "CORRUPT" not in captured.out
        assert "CORRUPT" in captured.err

    def test_file_writing_tools_are_not_read_only(self, profile_reset):
        mcp_server.set_tool_profile("full")
        by_name = {t["name"]: t for t in mcp_server._build_mcp_tools()}
        assert by_name["cs_report"]["annotations"]["readOnlyHint"] is False
        assert by_name["did"]["annotations"]["readOnlyHint"] is True


# ---------------------------------------------------------------------------
# Result layer
# ---------------------------------------------------------------------------


class TestResultPayload:
    @pytest.fixture
    def rd_result(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-1, 1, 800)
        y = 1 + 0.5 * (x >= 0) + x + rng.normal(0, 0.3, 800)
        return sp.rd(pd.DataFrame({"y": y, "x": x}), y="y", x="x", cutoff=0)

    def test_nested_diagnostics_survive(self, rd_result):
        d = rd_result.to_dict(detail="agent")
        assert isinstance(d["diagnostics"].get("mccrary"), dict)
        assert "pvalue" in d["diagnostics"]["mccrary"]
        json.dumps(d)  # still JSON-safe

    def test_degradations_recorded_not_swallowed(self, rd_result, monkeypatch):
        def _boom():
            raise RuntimeError("violation detector bug")

        monkeypatch.setattr(rd_result, "violations", _boom)
        with pytest.warns(WorkflowDegradedWarning, match="violation detector bug"):
            d = rd_result.to_dict(detail="agent")
        assert d["violations"] == []
        assert d["degradations"][0]["section"] == "violations"
        assert d["degradations"][0]["error_type"] == "RuntimeError"

    def test_clean_result_has_empty_degradations(self, rd_result):
        assert rd_result.to_dict(detail="agent")["degradations"] == []

    def test_to_json_accepts_detail(self, rd_result):
        agent = json.loads(rd_result.to_json(detail="agent"))
        assert "violations" in agent and "degradations" in agent
        standard = json.loads(rd_result.to_json())
        assert "violations" not in standard

    def test_mixin_results_have_to_json(self):
        from statspai._result_serialize import ResultProtocolMixin

        assert callable(getattr(ResultProtocolMixin, "to_json", None))


# ---------------------------------------------------------------------------
# Entry-point consistency
# ---------------------------------------------------------------------------


class TestEntryPoints:
    def test_rd_accepts_cutoff_alias(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(-1, 1, 500)
        df = pd.DataFrame(
            {"y": 1 + 0.4 * (x >= 0.1) + x + rng.normal(0, 0.2, 500), "x": x}
        )
        a = sp.rd(df, y="y", x="x", cutoff=0.1)
        b = sp.rd(df, y="y", x="x", c=0.1)
        assert a.estimate == pytest.approx(b.estimate)

    def test_did_accepts_outcome_treatment_unit(self, panel_df):
        a = sp.did(panel_df, outcome="y", treatment="treat", unit="id", time="t")
        b = sp.did(panel_df, y="y", treat="treat", id="id", time="t")
        assert a.estimate == pytest.approx(b.estimate)

    def test_did_unknown_keyword_gets_did_you_mean(self, panel_df):
        with pytest.raises(
            TypeError, match=r"unexpected keyword.*did you mean 'outcome'"
        ):
            sp.did(panel_df, outcom="y", treat="treat", time="t")

    def test_did_auto_derives_cohort_from_indicator(self, panel_df):
        r = sp.did(panel_df, y="y", treat="treat", id="id", time="t")
        assert "Callaway" in r.method
        panel_df["g"] = np.where(panel_df["id"] < 25, 4, 0)
        explicit = sp.did(panel_df, y="y", treat="g", id="id", time="t")
        assert r.estimate == pytest.approx(explicit.estimate)
        assert r.se == pytest.approx(explicit.se)

    def test_did_auto_rejects_non_absorbing_indicator(self, panel_df):
        df = panel_df.copy()
        df.loc[(df["id"] == 0) & (df["t"] == 6), "treat"] = 0
        with pytest.raises(MethodIncompatibility, match="switches back"):
            sp.did(df, y="y", treat="treat", id="id", time="t")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("staggered treatment timing with covariates", "staggered_cs"),
            ("two-way fixed effects bias", "bacon_decomposition"),
            ("regression discontinuity", "rdrobust"),
            ("effect of a policy with panel data", "did"),
            ("diff in diff", "did"),
            ("synthetic control", "synth"),
            ("weak instrument robust confidence interval", "anderson_rubin_ci"),
        ],
    )
    def test_natural_language_queries_hit(self, query, expected):
        names = [h["name"] for h in sp.search_functions(query)[:10]]
        assert expected in names, names

    def test_classes_are_not_listed(self):
        names = [h["name"] for h in sp.search_functions("result")]
        assert not any(n[0].isupper() for n in names)

    def test_backward_compatible_keyword_hits(self):
        assert sp.search_functions("treatment")
        assert "proximal" in [f["name"] for f in sp.search_functions("bridge")]


# ---------------------------------------------------------------------------
# Remediation hints must name real functions
# ---------------------------------------------------------------------------


def test_remediation_hints_reference_registered_functions():
    src = Path(sp.__file__).parent / "agent" / "remediation.py"
    text = src.read_text(encoding="utf-8")
    registered = set(sp.list_functions())
    referenced = set(re.findall(r"sp\.([a-z_][a-z0-9_]*)\(", text))
    # ``sp.<submodule>.<fn>`` style references are attribute chains, not
    # function calls; only direct ``sp.<name>(`` calls are checked.
    missing = sorted(
        n for n in referenced if n not in registered and not hasattr(sp, n)
    )
    assert not missing, f"remediation.py references unknown functions: {missing}"
