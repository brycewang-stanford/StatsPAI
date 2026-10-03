"""Per-family agent cards (``statspai._family_cards``) stay wired and honest."""

from __future__ import annotations

import builtins

import pytest

import statspai as sp
from statspai import exceptions as sp_exceptions
from statspai import registry
from statspai._family_cards import (
    FAMILY_CARDS,
    apply_variant_overrides,
    expand_family_cards,
)

registry._ensure_full_registry()
_REG = registry._REGISTRY


def _resolves(alt: str) -> bool:
    name = alt.replace("sp.", "").split("(")[0].strip()
    return name in _REG or callable(getattr(sp, name, None))


@pytest.mark.parametrize("family", sorted(FAMILY_CARDS))
def test_members_are_registered(family):
    missing = [m for m in FAMILY_CARDS[family]["members"] if m not in _REG]
    assert not missing, f"{family}: unregistered members {missing}"


@pytest.mark.parametrize("family", sorted(FAMILY_CARDS))
def test_card_pointers_resolve(family):
    card = FAMILY_CARDS[family]
    for alt in card.get("alternatives", []):
        assert _resolves(alt), (family, alt)
    for fm in card.get("failure_modes", []):
        assert set(fm) >= {"symptom", "exception", "remedy"}, (family, fm)
        if fm.get("alternative"):
            assert _resolves(fm["alternative"]), (family, fm["alternative"])
        exc = fm["exception"]
        if not exc.startswith("(none"):
            leaf = exc.split(".")[-1]
            assert hasattr(sp_exceptions, leaf) or hasattr(builtins, leaf), (
                family,
                exc,
            )
    parent = card.get("inherits_from")
    if parent:
        assert parent in _REG and _REG[parent].assumptions, (family, parent)


def test_every_family_states_assumptions_or_preconditions():
    for family, card in FAMILY_CARDS.items():
        assert card.get("assumptions") or card.get("pre_conditions"), family


def test_no_member_in_two_families():
    seen = {}
    for family, card in FAMILY_CARDS.items():
        for m in card["members"]:
            assert m not in seen, f"{m} in both {seen[m]} and {family}"
            seen[m] = family


def test_cards_reach_the_registry():
    cards = expand_family_cards()
    for name, card in cards.items():
        spec = _REG[name]
        merged = spec.agent_card()
        # A family statement reaches the members it is about, and only
        # those: ``apply_variant_overrides`` holds the scoping.
        kept, _ = apply_variant_overrides(name, list(card.get("assumptions", [])), [])
        dropped = set(card.get("assumptions", [])) - set(kept)
        for a in kept:
            assert a in merged["assumptions"], (name, a[:40])
        for a in dropped:
            assert a not in merged["assumptions"], (name, a[:40])
        if card.get("inherits_from"):
            assert spec.inherits_from  # family parent or a declared one
        described = sp.describe_function(name)
        if kept:
            assert described["assumptions"], name


def test_hand_written_specs_keep_their_own_first():
    # ``rdrobust`` has its own card; the family card must not push its
    # assumptions in front of the hand-written ones.
    own = registry._REGISTRY["rdrobust"].assumptions
    assert own[0].lower().startswith(("continuity", "no manipulation", "units", "the"))
