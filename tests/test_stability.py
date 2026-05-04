"""Tests for the v1.13 stability-tier surface.

Two distinct facets are exercised:

* function-level ``stability`` (``stable`` / ``experimental`` / ``deprecated``)
* per-function ``limitations`` (variant-level gaps inside otherwise
  stable functions, e.g. ``hal_tmle(variant='projection')``)

Both must surface through the registry, the OpenAI tool-schema
serializer, the help layer, and the CLI list filter.
"""
from __future__ import annotations

import pytest


class TestStabilitySchema:
    def test_tiers_constant_exposed(self):
        import statspai as sp

        assert "stable" in sp.STABILITY_TIERS
        assert "experimental" in sp.STABILITY_TIERS
        assert "deprecated" in sp.STABILITY_TIERS

    def test_default_stability_is_stable(self):
        from statspai.registry import FunctionSpec

        spec = FunctionSpec(name="x", category="utils", description="x")
        assert spec.stability == "stable"
        assert spec.limitations == []

    def test_invalid_stability_raises_at_construction(self):
        from statspai.registry import FunctionSpec

        with pytest.raises(ValueError, match="stability="):
            FunctionSpec(
                name="x", category="utils", description="x",
                stability="bogus_tier",
            )


class TestStabilityFlowsThroughRegistry:
    """to_dict / agent_card / OpenAI schema must all carry the new fields."""

    def test_to_dict_includes_stability_and_limitations(self):
        from statspai import describe_function

        d = describe_function("hal_tmle")
        assert d["stability"] == "stable"
        assert d["limitations"], (
            "hal_tmle should advertise its projection-variant gap in "
            "limitations so agents can route around it without "
            "burning a tool-call on NotImplementedError"
        )
        # The flagship example from the user's framing — must be visible
        # in the schema.
        assert any("projection" in lim for lim in d["limitations"])

    def test_principal_strat_advertises_instrument_gap(self):
        from statspai import describe_function

        d = describe_function("principal_strat")
        assert d["stability"] == "stable"
        assert any(
            "instrument" in lim.lower() for lim in d["limitations"]
        ), (
            "principal_strat must advertise that the instrument= "
            "two-layer setup is not yet implemented"
        )

    def test_agent_card_includes_stability_and_limitations(self):
        from statspai import agent_card

        card = agent_card("hal_tmle")
        assert card["stability"] == "stable"
        assert "limitations" in card
        assert card["limitations"]

    def test_openai_schema_prefixes_experimental_in_description(self):
        from statspai import function_schema

        schema = function_schema("text_treatment_effect")
        assert schema["description"].startswith("[experimental]"), (
            "An LLM tool-caller reads only the schema description; the "
            "experimental tag must travel with it"
        )

    def test_openai_schema_appends_limitations_to_description(self):
        from statspai import function_schema

        schema = function_schema("hal_tmle")
        assert "Known limitations:" in schema["description"]
        assert "projection" in schema["description"]


class TestStabilityFilters:
    def test_list_functions_stability_filter_stable(self):
        from statspai import list_functions

        stable = list_functions(stability="stable")
        # Most of the catalogue is stable, so the filtered set should
        # be much larger than the experimental bucket but still strictly
        # smaller than the unfiltered set (since at least one function
        # is experimental).
        all_fns = list_functions()
        assert len(stable) >= 100
        assert len(stable) < len(all_fns)

    def test_list_functions_stability_filter_experimental(self):
        from statspai import list_functions

        exp = list_functions(stability="experimental")
        # The three v1.13 experimental flagship entries.
        for name in ("text_treatment_effect", "llm_annotator_correct",
                     "did_multiplegt_dyn"):
            assert name in exp, (
                f"{name} should be visible to "
                "list_functions(stability='experimental')"
            )

    def test_list_functions_invalid_stability_raises(self):
        from statspai import list_functions

        with pytest.raises(ValueError, match="stability="):
            list_functions(stability="bogus")

    def test_category_and_stability_compose(self):
        from statspai import list_functions

        causal_stable = set(list_functions(category="causal", stability="stable"))
        causal_all = set(list_functions(category="causal"))
        # Intersection equals the causal-stable set; experimental causal
        # entries must be excluded by the filter.
        assert causal_stable.issubset(causal_all)
        # did_multiplegt_dyn is causal AND experimental — must drop out.
        assert "did_multiplegt_dyn" in causal_all
        assert "did_multiplegt_dyn" not in causal_stable

    def test_search_results_include_stability(self):
        from statspai import search_functions

        hits = search_functions("treatment")
        assert hits, "expected non-empty search hits for 'treatment'"
        for h in hits:
            assert "stability" in h, (
                "search results must carry stability so callers can "
                "post-filter without a second round-trip"
            )

    def test_agent_cards_stability_filter(self):
        from statspai import agent_cards

        stable_cards = agent_cards(stability="stable")
        for c in stable_cards:
            assert c["stability"] == "stable"


class TestStabilityInHelpLayer:
    def test_overview_mentions_stability_block(self):
        import statspai as sp

        text = str(sp.help())
        assert "STABILITY" in text
        assert "[stable]" in text
        # Either no experimental entries (then the tier is omitted) OR
        # the badge shows up.
        from statspai.registry import _REGISTRY
        if any(s.stability == "experimental" for s in _REGISTRY.values()):
            assert "[experimental]" in text

    def test_function_detail_shows_stability_line(self):
        import statspai as sp

        text = str(sp.help("hal_tmle"))
        assert "Stability :" in text
        assert "Known limitations" in text
        assert "projection" in text

    def test_experimental_function_detail_shows_tier(self):
        import statspai as sp

        text = str(sp.help("text_treatment_effect"))
        assert "experimental" in text.lower()

    def test_category_listing_marks_experimental(self):
        import statspai as sp

        text = str(sp.help("causal_text"))
        # text_treatment_effect is experimental in the causal_text
        # category — it must carry the badge in the listing so a human
        # scanning the column doesn't have to drill in to find out.
        assert "[experimental]" in text


class TestStabilityInCLI:
    def test_cli_list_stability_filter(self, capsys):
        from statspai.cli import main

        rc = main(["list", "--stability", "experimental"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "did_multiplegt_dyn" in out
        assert "text_treatment_effect" in out
        # Stable-only entry must not show up.
        assert "regress" not in out.split("\n")

    def test_cli_list_invalid_stability_rejected(self, capsys):
        from statspai.cli import main

        # argparse rejects bogus choices with SystemExit(2)
        with pytest.raises(SystemExit):
            main(["list", "--stability", "bogus"])
