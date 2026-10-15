"""Tests for Module D — output enrichment (next_calls / citations / narrative)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.agent import execute_tool
from statspai.agent._enrichment import (
    _FOLLOWUP_BY_TOOL,
    _required_args_by_tool,
    build_citations,
    build_narrative,
    build_next_calls,
    enrich_payload,
    fetch_bibtex,
)


def _toy_panel():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        treat = i % 2
        for t in (0, 1):
            y = 1.0 + 0.5 * t + 0.4 * treat * t + rng.normal(scale=0.1)
            rows.append({"id": i, "time": t, "treat": treat, "y": y})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# next_calls
# ----------------------------------------------------------------------


class TestBuildNextCalls:
    def test_did_has_followups(self):
        out = build_next_calls(
            "did", result_id="r_abc", base_args={"y": "wage", "data_path": "/tmp/x.csv"}
        )
        assert len(out) >= 1
        first = out[0]
        assert first["tool"] == "audit_result"
        assert first["arguments"]["result_id"] == "r_abc"
        assert first["ready"] is True
        # audit_result works from the handle alone: forwarding the
        # fit's column arguments would hand it keys its schema lacks.
        assert first["arguments"] == {"result_id": "r_abc"}

    def test_forwarded_arguments_are_in_the_target_schema(self):
        from statspai.agent._enrichment import _accepted_args

        base_args = {
            "data_path": "/tmp/x.csv",
            "y": "wage",
            "time": "year",
            "g": "first_treat",
            "id": "unit",
            "running_var": "score",
        }
        n_checked = 0
        for source in sorted(_FOLLOWUP_BY_TOOL):
            for call in build_next_calls(source, result_id="r_q", base_args=base_args):
                accepted = _accepted_args(call["tool"])
                if accepted is None:
                    continue
                extra = set(call["arguments"]) - accepted
                assert not extra, (source, call["tool"], sorted(extra))
                n_checked += 1
        assert n_checked > 20
        # ...and a tool that takes the column still receives it.
        rd = {c["tool"]: c for c in build_next_calls("rdrobust", base_args=base_args)}
        assert rd["rdplot"]["arguments"]["y"] == "wage"

    def test_iv_followups_have_weak_iv(self):
        out = build_next_calls("ivreg")
        names = [c["tool"] for c in out]
        assert "effective_f_test" in names
        assert "anderson_rubin_test" in names
        weak_iv = next(c for c in out if c["tool"] == "effective_f_test")
        assert weak_iv["ready"] is False
        assert set(weak_iv["missing_arguments"]) >= {"endog", "instruments"}

    def test_unknown_tool_returns_empty(self):
        assert build_next_calls("not_a_real_tool") == []

    def test_template_arguments_preserved_over_base(self):
        out = build_next_calls(
            "did", result_id="r_z", base_args={"method": "OVERWRITE_ME"}
        )
        # honest_did_from_result template carries method=SD; base_args
        # should NOT overwrite it.
        for c in out:
            if c["tool"] == "honest_did_from_result":
                assert c["arguments"]["method"] == "SD"

    def test_rd_aliases_make_followups_ready(self):
        out = build_next_calls(
            "rdrobust",
            result_id="r_rd",
            base_args={"y": "y", "running_var": "score", "data_path": "/tmp/rd.csv"},
        )
        by_tool = {c["tool"]: c for c in out}
        assert by_tool["rdplot"]["arguments"]["x"] == "score"
        assert by_tool["rddensity"]["arguments"]["x"] == "score"
        assert by_tool["rdsensitivity"]["arguments"]["x"] == "score"
        assert by_tool["rdplot"]["ready"] is True
        assert by_tool["rddensity"]["ready"] is True
        assert by_tool["rdsensitivity"]["ready"] is True

    def test_result_object_followups_expose_missing_argument(self):
        # The server resolves result_id into a ``result`` parameter, so the
        # handle makes a result-reading follow-up ready; without it the
        # follow-up says what is missing.
        out = build_next_calls("causal_forest", result_id="r_cf")
        cate = next(c for c in out if c["tool"] == "cate_summary")
        assert cate["ready"] is True
        assert cate["arguments"] == {"result_id": "r_cf"}
        out = build_next_calls("causal_forest")
        cate = next(c for c in out if c["tool"] == "cate_summary")
        assert cate["ready"] is False
        assert "result" in cate["missing_arguments"]
        assert "hint" in cate

    def test_data_bound_followups_need_a_dataset(self):
        from statspai.agent._enrichment import attach_data_source

        out = build_next_calls(
            "rdrobust", result_id="r_rd", base_args={"y": "y", "running_var": "score"}
        )
        rdplot = next(c for c in out if c["tool"] == "rdplot")
        # rdplot reads data, not a fitted result: no handle, not ready yet.
        assert "result_id" not in rdplot["arguments"]
        assert rdplot["ready"] is False and rdplot["missing_arguments"] == ["data_id"]
        payload = {"next_calls": out}
        attach_data_source(payload, {"data_id": "d_123"})
        assert rdplot["arguments"]["data_id"] == "d_123"
        assert rdplot["ready"] is True and "missing_arguments" not in rdplot

    def test_all_followups_carry_schema_consistent_readiness(self):
        from statspai.agent._enrichment import _accepted_args

        required_by_tool = _required_args_by_tool()
        assert required_by_tool
        base_args = {
            "data_id": "d_panel",
            "y": "y",
            "treat": "treat",
            "time": "time",
            "id": "id",
            "running_var": "score",
            "instrument": "z",
            "endog": "d",
            "covariates": ["x1", "x2"],
        }
        n_ready = 0
        for source in sorted(_FOLLOWUP_BY_TOOL):
            for call in build_next_calls(
                source,
                result_id="r_followup",
                base_args=base_args,
            ):
                tool = call["tool"]
                args = call.get("arguments") or {}
                accepted = _accepted_args(tool)
                if accepted is not None:
                    assert set(args) <= accepted, (source, call)
                required = set(required_by_tool.get(tool, set()))
                if "result_id" in args:
                    required.discard("result")
                if call.get("ready"):
                    n_ready += 1
                    assert required <= set(args), (source, call)
                    assert "missing_arguments" not in call
                else:
                    assert call["missing_arguments"], (source, call)
                    assert call.get("hint")
        assert n_ready > 10


# ----------------------------------------------------------------------
# citations
# ----------------------------------------------------------------------


class TestBuildCitations:
    def test_callaway_santanna(self):
        keys = build_citations("callaway_santanna")
        assert "callaway2021difference" in keys

    def test_unknown_tool_returns_empty(self):
        assert build_citations("not_a_real_tool") == []

    def test_fetch_bibtex_round_trips(self):
        # Pull a key that exists in paper.bib (skip if the build doesn't
        # ship one).
        keys = build_citations("synth")
        if not keys:
            pytest.skip("synth has no citation keys in this build")
        bib = fetch_bibtex(keys)
        # At least one key should resolve to a non-empty BibTeX body
        assert any(v for v in bib.values()), bib


# ----------------------------------------------------------------------
# narrative
# ----------------------------------------------------------------------


class TestBuildNarrative:
    def test_basic_payload(self):
        text = build_narrative(
            "did",
            {
                "method": "did",
                "estimate": 0.243,
                "std_error": 0.041,
                "conf_low": 0.162,
                "conf_high": 0.324,
                "n_obs": 5234,
                "estimand": "ATT",
            },
        )
        assert "did" in text.lower()
        assert "0.243" in text or "0.24" in text
        assert "ATT" in text
        assert "5,234" in text

    def test_estimator_payload_shape(self):
        # The estimator serializer emits ``se`` / ``pvalue`` / ``ci`` /
        # ``alpha`` and next steps as dicts; a zero p-value is a value.
        text = build_narrative(
            "callaway_santanna",
            {
                "method": "CS",
                "estimate": -0.0330,
                "se": 0.00776,
                "pvalue": 0.0,
                "ci": [-0.0482, -0.0178],
                "alpha": 0.1,
                "next_steps": [
                    {"action": "sp.pretrends_test(result)", "reason": "joint pre-test"}
                ],
            },
        )
        assert "90% CI [-0.0482, -0.0178]" in text
        assert "p=0" in text
        assert "`sp.pretrends_test(result)`" in text
        assert "{'action'" not in text

    def test_violations_surfaced(self):
        text = build_narrative(
            "did", {"method": "did", "violations": ["pre-trend rejected"]}
        )
        assert "violations" in text.lower()

    def test_empty_payload_returns_empty(self):
        text = build_narrative("did", {})
        # Header line with the method name is fine even on empty body
        assert isinstance(text, str)


# ----------------------------------------------------------------------
# enrich_payload integration
# ----------------------------------------------------------------------


class TestEnrichPayload:
    def test_inplace_mutation(self):
        p = {"estimate": 1.0, "method": "callaway_santanna"}
        out = enrich_payload(p, tool_name="callaway_santanna", result_id="r_x")
        assert out is p  # in-place
        assert "next_calls" in p
        assert "citations" in p
        assert "callaway2021difference" in p["citations"]["keys"]

    def test_existing_keys_not_overwritten(self):
        p = {"narrative": "user-provided", "citations": {"keys": ["custom"]}}
        enrich_payload(p, tool_name="callaway_santanna")
        assert p["narrative"] == "user-provided"
        assert p["citations"]["keys"] == ["custom"]


# ----------------------------------------------------------------------
# End-to-end: execute_tool surfaces next_calls + citations
# ----------------------------------------------------------------------


class TestExecuteToolEnrichment:
    def test_did_emits_next_calls(self):
        df = _toy_panel()
        out = execute_tool(
            "did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
            as_handle=True,
        )
        assert "next_calls" in out
        # First follow-up should be audit_result, fully wired.
        first = out["next_calls"][0]
        assert first["tool"] == "audit_result"
        assert first["arguments"]["result_id"] == out["result_id"]

    def test_did_emits_citations_or_skips_silently(self):
        df = _toy_panel()
        out = execute_tool(
            "did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
        )
        # `did` has no explicit citation key (empty list in the map)
        # so we should NOT emit a placeholder. This guards CLAUDE.md §10.
        assert "citations" not in out or out["citations"]["keys"]


def test_server_common_arguments_match_the_server():
    from statspai.agent._enrichment import _SERVER_COMMON_ARGS
    from statspai.agent.mcp_server import _COMMON_ARGUMENT_ORDER

    assert _SERVER_COMMON_ARGS == frozenset(_COMMON_ARGUMENT_ORDER)
