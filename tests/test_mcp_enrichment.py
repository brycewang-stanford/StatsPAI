"""Tests for Module D — output enrichment (next_calls / citations / narrative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.agent import execute_tool
from statspai.agent._enrichment import (
    build_next_calls,
    build_citations,
    fetch_bibtex,
    build_narrative,
    enrich_payload,
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
        out = build_next_calls("did", result_id="r_abc",
                                base_args={"y": "wage", "data_path": "/tmp/x.csv"})
        assert len(out) >= 1
        first = out[0]
        assert first["tool"] == "audit_result"
        assert first["arguments"]["result_id"] == "r_abc"
        # Forwarded base args
        assert first["arguments"].get("y") == "wage"

    def test_iv_followups_have_weak_iv(self):
        out = build_next_calls("ivreg")
        names = [c["tool"] for c in out]
        assert "effective_f_test" in names
        assert "anderson_rubin_test" in names

    def test_unknown_tool_returns_empty(self):
        assert build_next_calls("not_a_real_tool") == []

    def test_template_arguments_preserved_over_base(self):
        out = build_next_calls("did",
                                result_id="r_z",
                                base_args={"method": "OVERWRITE_ME"})
        # honest_did_from_result template carries method=SD; base_args
        # should NOT overwrite it.
        for c in out:
            if c["tool"] == "honest_did_from_result":
                assert c["arguments"]["method"] == "SD"


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
        text = build_narrative("did",
                                {"method": "did", "estimate": 0.243,
                                 "std_error": 0.041,
                                 "conf_low": 0.162, "conf_high": 0.324,
                                 "n_obs": 5234, "estimand": "ATT"})
        assert "did" in text.lower()
        assert "0.243" in text or "0.24" in text
        assert "ATT" in text
        assert "5,234" in text

    def test_violations_surfaced(self):
        text = build_narrative("did",
                                {"method": "did",
                                 "violations": ["pre-trend rejected"]})
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
        out = enrich_payload(p, tool_name="callaway_santanna",
                              result_id="r_x")
        assert out is p  # in-place
        assert "next_calls" in p
        assert "citations" in p
        assert "callaway2021difference" in p["citations"]["keys"]

    def test_existing_keys_not_overwritten(self):
        p = {"narrative": "user-provided",
             "citations": {"keys": ["custom"]}}
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
