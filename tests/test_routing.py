"""Machine-readable estimator routing (``sp.route`` / ``sp.decision_guide``).

The tables in ``statspai._routing`` restate the decision logic of
``docs/guides/choosing_*_estimator.md``. These tests keep them honest:
every route calls a registered function, every ``read_more`` is a heading
in the packaged guide, the packaged guides are byte-identical to
``docs/guides``, and the MCP tool / resource expose the same content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import statspai as sp
from statspai import _routing
from statspai.exceptions import MethodIncompatibility

ROOT = Path(sp.__file__).resolve().parents[2]
PKG_GUIDES = Path(sp.__file__).resolve().parent / "agent" / "_guides"
DOC_GUIDES = ROOT / "docs" / "guides"
REGISTERED = set(sp.list_functions())


def _headings(text: str) -> set[str]:
    return {
        re.sub(r"^#+\s*", "", ln).strip()
        for ln in text.splitlines()
        if ln.startswith("#")
    }


@pytest.mark.parametrize("family", sorted(_routing.FAMILIES))
def test_routes_call_registered_functions(family):
    fam = _routing.FAMILIES[family]
    for r in fam.routes:
        assert r.call in REGISTERED, (family, r.call)
        for extra in r.also:
            assert extra in REGISTERED, (family, extra)
        assert f"sp.{r.call}(" in r.example, (family, r.example)


@pytest.mark.parametrize("family", sorted(_routing.FAMILIES))
def test_route_conditions_use_declared_questions(family):
    fam = _routing.FAMILIES[family]
    questions = {q.key: q for q in fam.questions}
    for r in fam.routes:
        for key, accepted in r.when.items():
            assert key in questions, (family, key)
            values = [accepted] if isinstance(accepted, str) else list(accepted)
            for v in values:
                assert v in questions[key].options, (family, key, v)


@pytest.mark.parametrize("family", sorted(_routing.FAMILIES))
def test_read_more_anchors_exist_in_guide(family):
    fam = _routing.FAMILIES[family]
    text = (PKG_GUIDES / fam.guide).read_text(encoding="utf-8")
    heads = _headings(text)
    for r in fam.routes:
        if r.read_more:
            assert r.read_more in heads, (family, r.read_more)


def test_packaged_guides_match_docs():
    for src in sorted(DOC_GUIDES.glob("choosing_*_estimator.md")):
        dst = PKG_GUIDES / src.name
        assert dst.exists(), f"{src.name} not packaged; run scripts/sync_guides.py"
        assert (
            dst.read_bytes() == src.read_bytes()
        ), f"{src.name} drifted; run scripts/sync_guides.py"
    assert {f.guide for f in _routing.FAMILIES.values()} <= {
        p.name for p in PKG_GUIDES.glob("*.md")
    }


class TestRoute:
    def test_staggered_did_routes_to_callaway_santanna(self):
        r = sp.route("did", design="staggered", timing_random="no", covariates="none")
        assert r["routes"][0]["call"] == "callaway_santanna"
        assert "aggte" in r["routes"][0]["also"]
        assert "target" in r["unanswered"]

    def test_random_timing_routes_to_design_based(self):
        r = sp.route("did", design="staggered", timing_random="yes")
        assert r["routes"][0]["call"] == "staggered_rollout"

    def test_two_by_two_with_covariates_is_drdid(self):
        r = sp.route("did", design="2x2", covariates="yes")
        assert r["routes"][0]["call"] == "drdid"

    def test_weak_iv(self):
        r = sp.route("iv", strength="very_weak")
        assert r["routes"][0]["call"] == "anderson_rubin_ci"

    def test_rd_sharp_default(self):
        r = sp.route(
            "rd", assignment="sharp", running="continuous", inference="local_polynomial"
        )
        assert r["routes"][0]["call"] == "rdrobust"
        assert "rddensity" in r["routes"][0]["also"]

    def test_matching_att(self):
        assert (
            sp.route("matching", estimand="att", covariates="few")["routes"][0]["call"]
            == "ebalance"
        )

    def test_qte_three_period(self):
        assert (
            sp.route("qte", design="three_period_panel")["routes"][0]["call"]
            == "panel_qtet"
        )

    def test_dynamic_panel_persistent(self):
        assert (
            sp.route("dynamic_panel", persistence="near_unit_root")["routes"][0]["call"]
            == "xtdpdsys"
        )

    def test_no_answers_lists_everything_pending(self):
        r = sp.route("did")
        assert r["routes"] == []
        assert set(r["pending_routes"]) == {
            x.call for x in _routing.FAMILIES["did"].routes
        }
        assert r["next_question"]["key"] == "design"

    def test_unknown_family_question_answer(self):
        with pytest.raises(MethodIncompatibility, match="Unknown routing family"):
            sp.route("synthetic")
        with pytest.raises(MethodIncompatibility, match="Unknown question"):
            sp.route("did", colour="blue")
        with pytest.raises(MethodIncompatibility, match="Unknown answer"):
            sp.route("did", design="quasi")

    def test_decision_guide_shapes(self):
        g = sp.decision_guide("iv")
        assert g["family"] == "iv" and g["guide"].endswith(".md")
        assert {q["key"] for q in g["questions"]} >= {"strength", "n_instruments"}
        assert all({"when", "call", "example", "why"} <= set(r) for r in g["routes"])
        assert set(sp.decision_guide()) == set(_routing.FAMILIES)


class TestMCP:
    @staticmethod
    def _rpc(method, params):
        from statspai.agent.mcp_server import handle_request

        msg = json.loads(
            handle_request(
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                )
            )
        )
        return msg

    def test_route_estimator_tool(self):
        msg = self._rpc(
            "tools/call",
            {
                "name": "route_estimator",
                "arguments": {
                    "family": "did",
                    "answers": {
                        "design": "staggered",
                        "timing_random": "no",
                        "covariates": "yes",
                    },
                },
            },
        )
        out = msg["result"]["structuredContent"]
        assert out["routes"][0]["call"] == "callaway_santanna"
        assert "estimator='dr'" in out["routes"][0]["example"]
        assert out["guide_uri"] == "statspai://guide/did"

    def test_route_estimator_without_answers_returns_questions(self):
        out = self._rpc(
            "tools/call", {"name": "route_estimator", "arguments": {"family": "rd"}}
        )["result"]["structuredContent"]
        assert {q["key"] for q in out["questions"]} >= {"assignment", "running"}
        assert len(out["routes"]) == len(_routing.FAMILIES["rd"].routes)

    def test_route_estimator_bad_answer_is_structured(self):
        out = self._rpc(
            "tools/call",
            {
                "name": "route_estimator",
                "arguments": {"family": "rd", "answers": {"running": "wavy"}},
            },
        )["result"]["structuredContent"]
        assert out["error_kind"] == "method_incompatibility"
        assert "Unknown answer" in out["error"]

    def test_guide_resource(self):
        msg = self._rpc("resources/read", {"uri": "statspai://guide/matching"})
        text = msg["result"]["contents"][0]["text"]
        assert text.startswith("# Choosing a matching / weighting estimator")
        assert msg["result"]["contents"][0]["mimeType"] == "text/markdown"
        tmpl = self._rpc("resources/templates/list", {})
        assert "statspai://guide/{family}" in {
            t["uriTemplate"] for t in tmpl["result"]["resourceTemplates"]
        }

    def test_unknown_guide(self):
        msg = self._rpc("resources/read", {"uri": "statspai://guide/synth"})
        assert msg["error"]["code"] == -32002
