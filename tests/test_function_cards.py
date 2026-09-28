"""Per-function agent cards, batch 2 (``statspai._function_cards``).

The family cards are the floor; these cards refine one entry point each
for the DiD / IV / RD / synthetic-control / DML families. The tests pin
the contract: every card names a registered function, every pointer
resolves, the card reaches ``describe_function`` / ``agent_card``, a
per-function card overrides a family template field by field (and keeps
the fields it does not state), and a hand-written registry entry keeps
its own content first.
"""

from __future__ import annotations

import builtins

import pytest

import statspai as sp
from statspai import exceptions as sp_exceptions
from statspai import registry
from statspai._agent_cards_extra import EXTRA_AGENT_CARDS
from statspai._causal_family_seeds import CAUSAL_FAMILY_SEEDS
from statspai._family_cards import expand_family_cards
from statspai._function_cards import FUNCTION_CARDS

registry._ensure_full_registry()
_REG = registry._REGISTRY

_FAMILIES = {
    "did": ("did", "sun_abraham", "honest_did", "etwfe", "event_study"),
    "iv": ("ivreg", "liml", "jive", "bartik", "anderson_rubin_ci"),
    "rd": ("rdrobust", "rddensity", "rdrandinf", "rkd", "rd_honest"),
    "synth": ("sdid", "augsynth", "scpi", "synth_compare", "matrix_completion"),
    "dml": ("dml", "dml_panel", "dynamic_dml", "dml_sensitivity"),
}


def _resolves(alt: str) -> bool:
    name = alt.replace("sp.", "").split("(")[0].strip()
    return name in _REG or callable(getattr(sp, name, None))


def test_batch_covers_the_five_families():
    for family, names in _FAMILIES.items():
        missing = [n for n in names if n not in FUNCTION_CARDS]
        assert not missing, (family, missing)
    assert len(FUNCTION_CARDS) >= 80


@pytest.mark.parametrize("name", sorted(FUNCTION_CARDS))
def test_card_is_registered_and_pointers_resolve(name):
    assert name in _REG, f"{name}: card for an unregistered function"
    card = FUNCTION_CARDS[name]
    allowed = {
        "assumptions",
        "pre_conditions",
        "failure_modes",
        "alternatives",
        "not_recommended_when",
        "cost_profile",
        "typical_n_min",
        "inherits_from",
    }
    assert set(card) <= allowed, (name, set(card) - allowed)
    assert card, name
    for alt in card.get("alternatives", []):
        assert _resolves(alt), (name, alt)
        assert alt.replace("sp.", "") != name, f"{name} lists itself"
    for fm in card.get("failure_modes", []):
        assert set(fm) >= {"symptom", "exception", "remedy"}, (name, fm)
        if fm.get("alternative"):
            assert _resolves(fm["alternative"]), (name, fm["alternative"])
        exc = fm["exception"]
        if not exc.startswith("(none"):
            leaf = exc.split(".")[-1]
            assert hasattr(sp_exceptions, leaf) or hasattr(builtins, leaf), (name, exc)
    for field in ("assumptions", "pre_conditions", "not_recommended_when"):
        for item in card.get(field, []):
            assert isinstance(item, str) and len(item) >= 20, (name, field, item)


@pytest.mark.parametrize("name", sorted(FUNCTION_CARDS))
def test_card_reaches_every_discovery_view(name):
    card = FUNCTION_CARDS[name]
    described = sp.describe_function(name)
    agent = sp.agent_card(name)
    for field in ("assumptions", "pre_conditions", "not_recommended_when"):
        for item in card.get(field, []):
            assert item in described[field], (name, field, item[:50])
            assert item in agent[field], (name, field, item[:50])
    for fm in card.get("failure_modes", []):
        assert any(m["symptom"] == fm["symptom"] for m in described["failure_modes"]), (
            name,
            fm["symptom"][:50],
        )
    if card.get("cost_profile"):
        assert described["cost_profile"], name
    # No card leaves its function without the negative guidance batch 2 exists for.
    assert described["not_recommended_when"] or described["failure_modes"], name


class TestPrecedence:
    """Family floor < per-function card < hand-written registry entry."""

    def test_function_card_overrides_template_field_by_field(self):
        # ``rdrandinf`` is seeded from the RD family template (continuity at
        # the cutoff) but is a local-randomisation estimator: its card states
        # its own assumptions, which must *replace* the template's ...
        template_names = {n for _, names in CAUSAL_FAMILY_SEEDS for n in names}
        assert "rdrandinf" in template_names
        described = sp.describe_function("rdrandinf")
        assert described["assumptions"] == FUNCTION_CARDS["rdrandinf"]["assumptions"]
        assert not any(
            "continuous at the cutoff" in a for a in described["assumptions"]
        )
        # ... while a field the card does not state keeps the template's.
        assert "sdid" in template_names
        assert "assumptions" not in FUNCTION_CARDS["sdid"]
        sdid = sp.describe_function("sdid")
        assert sdid["assumptions"], "sdid lost the template floor"
        own = FUNCTION_CARDS["sdid"]["not_recommended_when"]
        assert sdid["not_recommended_when"][: len(own)] == own

    def test_function_card_overrides_family_card_field_by_field(self):
        family = expand_family_cards()
        overlap = sorted(set(family) & set(FUNCTION_CARDS))
        assert overlap, "no family/function overlap to test"
        for name in overlap:
            card, floor = FUNCTION_CARDS[name], family[name]
            described = sp.describe_function(name)
            for field in ("assumptions", "pre_conditions", "not_recommended_when"):
                if field in card:
                    assert described[field][: len(card[field])] == card[field], (
                        name,
                        field,
                    )
                elif floor.get(field):
                    assert floor[field][0] in described[field], (name, field)

    def test_hand_written_entry_keeps_its_own_first(self):
        overlap = sorted(set(EXTRA_AGENT_CARDS) & set(FUNCTION_CARDS))
        assert overlap, "no hand-written/function overlap to test"
        for name in overlap:
            hand = EXTRA_AGENT_CARDS[name]
            described = sp.describe_function(name)
            for field in ("assumptions", "not_recommended_when"):
                if hand.get(field):
                    assert described[field][0] == hand[field][0], (name, field)
                for item in FUNCTION_CARDS[name].get(field, []):
                    assert item in described[field], (name, field, item[:40])

    def test_registry_specs_stay_first(self):
        # ``did`` and ``rdrobust`` have hand-written FunctionSpec cards; the
        # per-function card appends and never displaces their first item.
        for name in ("did", "rdrobust"):
            spec_first = _REG[name].assumptions[0]
            assert spec_first not in FUNCTION_CARDS[name].get("assumptions", [])
            assert sp.describe_function(name)["assumptions"][0] == spec_first

    def test_no_alternative_points_at_itself(self):
        for name, spec in _REG.items():
            for alt in spec.alternatives:
                assert alt.replace("sp.", "").split("(")[0].strip() != name, name


def test_agent_card_schema_carries_batch_2_guidance():
    schema = sp.function_schema("rdrandinf", agent_native=True)
    x = schema["x_statspai"]
    assert (
        FUNCTION_CARDS["rdrandinf"]["not_recommended_when"][0]
        in x["not_recommended_when"]
    )
