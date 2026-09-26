"""sp.support_tier: the maintenance tier is derived, not stored."""

import pytest

import statspai as sp
from statspai.registry import _FRONTIER_METHOD_LIMITATIONS


def test_certified_stable_estimator_is_core():
    assert sp.support_tier("callaway_santanna") == "core"


def test_frontier_method_cards_are_research_regardless_of_stability():
    for name in _FRONTIER_METHOD_LIMITATIONS:
        if name in sp.list_functions():
            assert sp.support_tier(name) == "research", name


def test_tier_follows_the_two_registry_axes():
    for name in sp.list_functions()[:200]:
        spec = sp.describe_function(name)
        tier = sp.support_tier(name)
        assert tier in {"core", "extension", "research"}
        if spec.get("stability") in ("experimental", "deprecated"):
            assert tier == "research", name


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        sp.support_tier("definitely_not_a_function")
