"""Registry metadata coverage ratchet (roadmap W3).

The 2026-09-28 agent-native audit found that 982 of 1,259 registry
entries were auto-generated from signatures with ``returns=""`` /
``example=""`` / ``reference=""`` and no agent-native card, and that a
third of parameter descriptions were the placeholder "``<name>
parameter.``". Search, ``describe_function`` and the MCP cards are only
as good as this metadata, so its coverage is pinned here and may only
move up. Each floor sits a little under the measured value at the time
it was set (measured values in the comments) so ordinary churn cannot
red a PR; raise a floor when a curation batch lands.

Auto-generated entries harvest their ``Returns`` / first ``Examples``
statement / ``References`` from the docstring; family variants without a
card inherit their dispatcher's (``rd_* → rdrobust``, ``did_* → did``, …).
"""

from __future__ import annotations

import re

import pytest

import statspai as sp
from statspai import registry

registry._ensure_full_registry()
_SPECS = list(registry._REGISTRY.values())
_CALLABLES = [s for s in _SPECS if getattr(s, "_kind", None) != "class"]

_PLACEHOLDER = re.compile(r"^\w+ parameter( \(.*\))?\.$")


def _share(items, pred) -> float:
    return sum(1 for s in items if pred(s)) / len(items)


# --------------------------------------------------------------------------
# Floors. Measured 2026-09-28 after the docstring harvest:
#   example 0.959 · returns 0.834 (callables) · reference 0.334 ·
#   card 0.463 (callables) · placeholder params 0.527; family cards lift card to 0.75
# Re-measured after the W10 discovery pass (docstring descriptions for
# hand-written / class / alias parameters, estimator cards batch 3):
#   card 0.779 · placeholder params 0.388 · option-param enums 0.673 ·
#   result_class 0.981 (callables) · curated assumptions 265 callables
# --------------------------------------------------------------------------
EXAMPLE_FLOOR = 0.95
RETURNS_FLOOR = 0.80
REFERENCE_FLOOR = 0.30
CARD_FLOOR = 0.77
PLACEHOLDER_PARAM_CEILING = 0.39
OPTION_ENUM_FLOOR = 0.66
RESULT_CLASS_FLOOR = 0.97
CURATED_ASSUMPTIONS_FLOOR = 260


def test_example_coverage():
    assert _share(_SPECS, lambda s: bool(s.example)) >= EXAMPLE_FLOOR


def test_returns_coverage_on_callables():
    assert _share(_CALLABLES, lambda s: bool(s.returns)) >= RETURNS_FLOOR


def test_reference_coverage():
    assert _share(_SPECS, lambda s: bool(s.reference)) >= REFERENCE_FLOOR


def test_agent_card_coverage_on_callables():
    assert (
        _share(_CALLABLES, lambda s: bool(s.assumptions or s.inherits_from))
        >= CARD_FLOOR
    )


def test_placeholder_parameter_descriptions_do_not_grow():
    total = placeholder = 0
    for s in _SPECS:
        for p in s.params:
            total += 1
            desc = (p.description or "").strip()
            if not desc or _PLACEHOLDER.match(desc):
                placeholder += 1
    assert placeholder / total <= PLACEHOLDER_PARAM_CEILING


def test_option_parameters_carry_enums():
    """Share of string-typed option parameters (``method`` / ``kernel`` /
    ``vce`` …) whose schema states the accepted values."""
    from statspai._schema_enrich import OPTION_PARAM_NAMES

    total = with_enum = 0
    for s in _CALLABLES:
        props = s.to_openai_schema()["parameters"]["properties"]
        for pname, prop in props.items():
            typ = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
            if pname in OPTION_PARAM_NAMES and "string" in typ:
                total += 1
                with_enum += bool(prop.get("enum"))
    assert with_enum / total >= OPTION_ENUM_FLOOR


def test_result_class_coverage_on_callables():
    assert _share(_CALLABLES, lambda s: bool(s.result_class)) >= RESULT_CLASS_FLOOR


def test_curated_assumption_cards_do_not_shrink():
    curated = sum(
        1
        for s in _CALLABLES
        if s.agent_card()["provenance"].get("assumptions")
        in {"curated", "curated+family"}
    )
    assert curated >= CURATED_ASSUMPTIONS_FLOOR


def test_classes_are_marked_and_not_tools():
    classes = [s.name for s in _SPECS if getattr(s, "_kind", None) == "class"]
    assert len(classes) > 250
    for name in classes[:50]:
        assert sp.describe_function(name)["kind"] == "class"
    from statspai.agent.auto_tools import _is_agent_safe

    assert not any(_is_agent_safe(n, registry._REGISTRY[n]) for n in classes[:50])


def test_schema_parameters_are_internally_consistent():
    """No enum on a mismatched type, no default outside its enum, no default
    of a type the parameter does not declare."""
    json_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": (list, tuple),
        "object": dict,
    }
    bad = []
    for schema in sp.all_schemas():
        for pname, prop in schema["parameters"]["properties"].items():
            typ = prop.get("type")
            types = typ if isinstance(typ, list) else [typ]
            enum = prop.get("enum")
            default = prop.get("default")
            if enum is not None:
                for v in enum:
                    if not any(
                        isinstance(v, json_types[t])
                        and not (t != "boolean" and isinstance(v, bool))
                        for t in types
                        if t in json_types
                    ):
                        bad.append((schema["name"], pname, "enum-type", v, typ))
                        break
                if default is not None and default not in enum:
                    bad.append((schema["name"], pname, "default-not-in-enum", default))
            if default is not None:
                ok = any(
                    isinstance(default, json_types[t])
                    and not (t != "boolean" and isinstance(default, bool))
                    for t in types
                    if t in json_types
                )
                if not ok:
                    bad.append((schema["name"], pname, "default-type", default, typ))
    assert not bad, bad[:20]


def test_describe_function_merges_inherited_card():
    d = sp.describe_function("rd_honest")
    assert d["inherited_from"] == "rdrobust"
    assert d["inheritance"] in {"declared", "family (derived from the name)"}
    assert len(d["assumptions"]) >= len(registry._REGISTRY["rdrobust"].assumptions)
    assert isinstance(d["auto_generated"], bool)
    # A family-derived link on an auto entry says so.
    derived = [
        n for n, s in registry._REGISTRY.items() if getattr(s, "_inherited_auto", False)
    ]
    assert derived, "family inheritance derived nothing"
    dd = sp.describe_function(derived[0])
    assert dd["inheritance"] == "family (derived from the name)"
    assert dd["auto_generated"] is True and dd["assumptions"]


def test_harvested_fields_are_bounded_and_bind():
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from registry_example_audit import audit_one  # noqa: E402

    for s in _SPECS:
        if not getattr(s, "_auto", False):
            continue  # hand-written entries may be as long as their authors like
        assert len(s.returns) <= 260, s.name
        assert len(s.reference) <= 260, s.name
        if s.example:
            assert not audit_one(s.name, s.example), (s.name, s.example)


@pytest.mark.parametrize("name", ["aggte", "pretrends_test", "did_2x2", "rd_honest"])
def test_harvest_samples(name):
    s = registry._REGISTRY[name]
    assert s.returns and s.example and f"sp.{name}(" in s.example


class TestStructuredEvidence:
    """``evidence`` is the parity index as data, in every discovery view."""

    def test_track_a_module_reports_recorded_t2(self):
        e = sp.describe_function("regress")["evidence"]
        assert e["status"] == "bit-exact" and e["grade"] == "T2"
        assert e["grade_basis"] == "recorded"
        assert {"R", "Stata"} <= set(e["sides"])
        assert e["reference"] and e["tests"]

    def test_stochastic_forest_is_t3_not_parity(self):
        e = sp.describe_function("causal_forest")["evidence"]
        assert e["grade"] == "T3" and e["status"] == "aligned"

    def test_known_truth_only_is_t1(self):
        e = sp.describe_function("DoubleML")["evidence"]
        assert e["status"] == "analytical-only" and e["grade"] == "T1"
        assert e["sides"] == ["py"]

    def test_unverified_is_explicit(self):
        e = sp.describe_function("route")["evidence"]
        assert e == {
            "status": "unverified",
            "grade": None,
            "grade_basis": "no numerical evidence",
            "sides": [],
            "reference": "",
            "reference_versions": {},
            "tolerance": "",
            "tests": [],
            "module_id": None,
            "source": None,
        }

    def test_evidence_present_in_card_and_agent_schema(self):
        card = sp.agent_card("callaway_santanna")
        assert card["evidence"]["grade"] == "T2"
        schema = sp.function_schema("callaway_santanna", agent_native=True)
        assert schema["x_statspai"]["evidence"]["status"] == "bit-exact"

    def test_grade_never_overstates_status(self):
        from statspai.registry import evidence_record

        for name in registry._REGISTRY:
            e = evidence_record(name)
            if e["grade"] == "T2":
                assert e["status"] in {"bit-exact", "aligned"}, name
                assert set(e["sides"]) & {"R", "Stata"}, name
            if e["status"] == "unverified":
                assert e["grade"] is None, name
