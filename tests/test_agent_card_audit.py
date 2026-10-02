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
