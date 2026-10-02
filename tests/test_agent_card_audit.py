"""The agent cards of the 30 most-used entry points say true things.

``scripts/agent_card_audit.py`` checks each card against a real call
(review item A1). The enum sweep takes about ten minutes, so it runs
offline and its result is committed as ``docs/dev/agent_card_audit.json``;
this file runs the fast layer on every test run and pins the committed
result to the schemas it was computed from.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

import statspai as sp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agent_card_audit.py"
REPORT = ROOT / "docs" / "dev" / "agent_card_audit.json"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="source checkout only (scripts/ not installed)"
)


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("agent_card_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_thirty_functions_each_with_a_call(audit):
    assert len(audit.TOP_30) == len(set(audit.TOP_30)) == 30
    assert set(audit.TOP_30) <= set(audit.CALLS)
    registered = set(sp.list_functions())
    assert set(audit.TOP_30) <= registered


def test_fast_layer_finds_no_defect(audit):
    """Required arguments, result class and alternatives, by real calls."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = audit.build(enums=False)
    defects = {f["function"]: f["defects"] for f in fresh["functions"] if f["defects"]}
    assert not defects, defects
    for f in fresh["functions"]:
        assert f["base_call"] == "ok"
        assert f["result_class"]["matches"] is True


def test_committed_enum_sweep_has_no_rejected_value(report):
    assert report["n_functions"] == 30
    assert report["n_with_defects"] == 0
    assert report["enum_values"]["rejected"] == 0
    assert report["enum_values"]["ok"] > 200


def test_enum_values_added_since_the_sweep_are_accepted(audit, report):
    """The committed sweep may lag the schemas; a new value is tried here.

    Re-running the whole sweep takes about ten minutes, so a commit that
    adds an enum value to one of the thirty is not asked to. Instead every
    value the committed report has not seen is called for real, now: it may
    need a precondition, it may not be refused. Regenerate the report
    (``python scripts/agent_card_audit.py``) when convenient.
    """
    committed = {
        (f["function"], arg, value)
        for f in report["functions"]
        for arg, per_value in (f.get("enums") or {}).items()
        for value in per_value
    }
    fresh = []
    for name in audit.TOP_30:
        props = sp.function_schema(name)["parameters"]["properties"]
        for arg, spec in props.items():
            for value in spec.get("enum") or []:
                if (name, arg, str(value)) not in committed:
                    fresh.append((name, arg, value))
    assert len(fresh) <= 25, (
        f"{len(fresh)} enum values are newer than the committed sweep; "
        "run python scripts/agent_card_audit.py"
    )
    rejected = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, arg, value in fresh:
            outcome = audit.try_enum_value(name, arg, value)
            if outcome["status"] == "rejected":
                rejected.append((name, arg, value, outcome["error"]))
    assert not rejected, rejected


def test_every_precondition_carries_its_reason(report):
    for f in report["functions"]:
        for per_value in (f.get("enums") or {}).values():
            for outcome in per_value.values():
                if outcome["status"] == "precondition":
                    assert len(outcome["error"]) > 20


# ---------------------------------------------------------------------------
# What the cards say (read by hand on 2026-10-03; see VARIANT_OVERRIDES)
# ---------------------------------------------------------------------------


def _assumptions(name):
    return " | ".join(sp.describe_function(name)["assumptions"])


def _symptoms(name):
    return " | ".join(f["symptom"] for f in sp.describe_function(name)["failure_modes"])


def test_binary_models_do_not_list_ordered_or_multinomial_assumptions():
    for name in ("logit", "probit"):
        text = _assumptions(name)
        assert "independence of irrelevant alternatives" not in text
        assert "proportional odds" not in text
        assert "link function" in text  # the family statement that does apply
    # ...and the models those statements are about keep them.
    assert "independence of irrelevant alternatives" in _assumptions("mlogit")
    assert "proportional odds" in _assumptions("ologit")
    assert "proportional odds" not in _assumptions("mlogit")
    assert "independence of irrelevant alternatives" not in _assumptions("ologit")


def test_robust_did_estimators_are_not_told_to_use_themselves():
    for name in ("callaway_santanna", "sun_abraham", "did_imputation", "etwfe"):
        assert "use CS or SA" not in _assumptions(name), name
        assert "TWFE method" not in _symptoms(name), name
        assert "Parallel trends" in _assumptions(name)
        assert _assumptions(name).count("SUTVA") == 1, name
    # The advice stays where it belongs: on the dispatcher that can run TWFE.
    assert "use CS or SA" in _assumptions("did")
    assert "TWFE method" in _symptoms("did")


def test_rdrobust_card_covers_the_fuzzy_and_kink_designs():
    text = _assumptions("rdrobust")
    assert "Continuity of potential outcomes" in text
    assert "fuzzy=" in text and "monotonicity" in text and "compliers" in text
    assert "deriv=1 (kink)" in text
    # Continuity-based local polynomials, not the local-randomization framework.
    assert "Local randomization only" not in text


def test_point_treatment_ipw_is_not_described_as_a_longitudinal_method():
    text = _assumptions("ipw")
    assert "Sequential exchangeability" not in text and "given the past" not in text
    assert "point treatment" in text and "propensity model" in text


def test_dml_card_names_the_parameter_it_estimates():
    text = _assumptions("dml")
    assert "√n CATE" not in text
    assert "√n-consistent" in text


def test_overrides_only_name_real_functions_and_real_statements():
    """An override that matches nothing is a typo or a stale entry."""
    from statspai._family_cards import VARIANT_OVERRIDES
    from statspai.registry import _REGISTRY

    for name, override in VARIANT_OVERRIDES.items():
        assert name in _REGISTRY, name
        raw = _REGISTRY[name].agent_card()  # after overrides
        for field, texts in override.get("add", {}).items():
            for text in texts:
                assert text in raw[field], (name, text[:40])
        for field, prefixes in override.get("drop", {}).items():
            values = [v["symptom"] if isinstance(v, dict) else v for v in raw[field]]
            added = override.get("add", {}).get(field, [])
            for prefix in prefixes:
                left = [v for v in values if v.startswith(prefix) and v not in added]
                assert not left, (name, prefix)
