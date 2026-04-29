"""Tests for Module G — expanded prompt-template surface."""
from __future__ import annotations

import json

from statspai.agent import mcp_handle_request


def _rpc(method, params=None, request_id=1):
    msg = {"jsonrpc": "2.0", "id": request_id, "method": method,
           "params": params or {}}
    return json.loads(mcp_handle_request(json.dumps(msg)))


REQUIRED_PROMPTS = {
    "audit_did_result",
    "audit_iv_result",
    "audit_rd_result",
    "design_then_estimate",
    "robustness_followup",
    "paper_render",
    "compare_methods",
    "policy_evaluation",
    "synth_full",
    "decompose_inequality",
}


class TestPromptListExpansion:
    def test_all_required_prompts_present(self):
        msg = _rpc("prompts/list", {})
        names = {p["name"] for p in msg["result"]["prompts"]}
        missing = REQUIRED_PROMPTS - names
        assert not missing, f"missing prompts: {missing}"

    def test_each_prompt_has_args_metadata(self):
        msg = _rpc("prompts/list", {})
        for p in msg["result"]["prompts"]:
            for arg in p["arguments"]:
                # Every arg must declare its required field
                assert "required" in arg, p["name"]

    def test_audit_iv_renders_template(self):
        msg = _rpc("prompts/get", {
            "name": "audit_iv_result",
            "arguments": {"data_path": "/d.csv", "formula": "y ~ (d ~ z)"},
        })
        text = msg["result"]["messages"][0]["content"]["text"]
        assert "pipeline_iv" in text
        assert "/d.csv" in text
        assert "y ~ (d ~ z)" in text

    def test_paper_render_uses_result_id(self):
        msg = _rpc("prompts/get", {
            "name": "paper_render",
            "arguments": {"result_id": "r_xyz"},
        })
        text = msg["result"]["messages"][0]["content"]["text"]
        assert "r_xyz" in text
        assert "brief_result" in text
        assert "audit_result" in text
        assert "plot_from_result" in text
        assert "bibtex" in text

    def test_compare_methods_template(self):
        msg = _rpc("prompts/get", {
            "name": "compare_methods",
            "arguments": {"data_path": "/d.csv", "y": "wage",
                          "treat": "treated"},
        })
        text = msg["result"]["messages"][0]["content"]["text"]
        assert "callaway_santanna" in text
        assert "did_imputation" in text
