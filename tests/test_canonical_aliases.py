"""Tests for the D6 family-canonical declaration and the §3 contract audit."""

from __future__ import annotations

import statspai
from statspai._canonical_aliases import CANONICAL_ALIASES, canonical_of, is_legacy_alias


def test_bjs_canonical_and_aliases() -> None:
    assert canonical_of("bjs") == "bjs"
    assert canonical_of("borusyak_jaravel_spiess") == "bjs"
    assert canonical_of("did_imputation") == "bjs"
    assert not is_legacy_alias("bjs")
    assert is_legacy_alias("borusyak_jaravel_spiess")
    assert is_legacy_alias("did_imputation")


def test_gardner_canonical_and_aliases() -> None:
    assert canonical_of("gardner_did") == "gardner_did"
    assert canonical_of("did_2stage") == "gardner_did"
    assert not is_legacy_alias("gardner_did")
    assert is_legacy_alias("did_2stage")


def test_unknown_name_is_its_own_canonical() -> None:
    for n in ("feols", "ivreg", "xx", ""):
        assert canonical_of(n) == n
        assert not is_legacy_alias(n)


def test_legacy_aliases_still_resolve_and_match_canonical() -> None:
    """Additive guarantee: every legacy alias is a real, callable function
    whose behaviour matches the canonical one (no removals, no renames)."""
    bjs_aliases = CANONICAL_ALIASES["bjs"]
    can_bjs = statspai.bjs
    for alias in bjs_aliases:
        fn = getattr(statspai, alias)
        assert fn is can_bjs
    can_g = statspai.gardner_did
    for alias in CANONICAL_ALIASES["gardner_did"]:
        assert getattr(statspai, alias) is can_g
