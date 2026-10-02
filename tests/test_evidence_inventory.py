"""``docs/evidence_inventory.{json,md}`` is generated, current and honest.

The inventory is the entry x configuration x output view of
``sp.validation_scope`` (2026-10-02 review, R1 / R3). It replaces a
hand-maintained point-in-time table, so the properties worth guarding are
that it cannot go stale and that its counts are what a direct query
returns.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

import statspai as sp
from statspai.validation_scope import SCOPE_FUNCTIONS, SCOPES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_evidence_inventory.py"
JSON_OUT = ROOT / "docs" / "evidence_inventory.json"
MD_OUT = ROOT / "docs" / "evidence_inventory.md"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="source checkout only (scripts/ not installed)"
)


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_evidence_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(builder):
    return builder.build()


def test_committed_inventory_is_current(builder, built):
    assert json.loads(JSON_OUT.read_text(encoding="utf-8")) == built, (
        "docs/evidence_inventory.json is stale: "
        "run python scripts/build_evidence_inventory.py"
    )
    assert MD_OUT.read_text(encoding="utf-8") == builder.render(built)


def test_every_scope_function_is_inventoried_over_its_whole_grid(built):
    assert set(built["functions"]) == set(SCOPE_FUNCTIONS)
    for name, inv in built["functions"].items():
        scope = SCOPES[name]
        cells = math.prod(len(scope.domains[d]) for d in scope.dimensions)
        assert inv["cells"] == cells
        assert sum(inv["overall"].values()) == cells
        for out, counts in inv["by_output"].items():
            assert sum(counts.values()) == cells, (name, out)
        assert len(inv["rows"]) == len(scope.rows)


def test_counts_agree_with_direct_queries(built):
    """Spot checks against ``sp.validation_scope`` itself, by hand."""
    reg = built["functions"]["regress"]
    # joint Wald F is pinned for classical / hc1 / cr1, unweighted.
    assert reg["by_output"]["joint_test"]["reference"] == 3
    direct = sp.validation_scope(function="regress", vce="hc3", weights="none")
    assert direct["outputs"]["joint_test"]["status"] == "not_covered"

    sa = built["functions"]["sun_abraham"]
    # fixed shares x two summary aggregations x never-treated controls.
    assert sa["by_output"]["vcov"]["reference"] == 2
    assert sa["cells"] == 8

    forest = built["functions"]["causal_forest"]
    # A forest is never a same-byte reference: T3 / S / B only.
    assert "reference" not in forest["by_output"]["estimate"]
    assert forest["overall"].get("covered", 0) == 0


def test_no_function_is_fully_covered_on_its_whole_grid(built):
    """The inventory exists because a function tier overstates this."""
    partial = [
        name
        for name, inv in built["functions"].items()
        if inv["overall"].get("covered", 0) < inv["cells"]
    ]
    assert len(partial) == len(built["functions"]) >= 14


def test_option_fixtures_are_read_and_hashed(built):
    fixtures = built["option_fixtures"]
    on_disk = sorted(
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "tests/stata_parity/option_parity").rglob("*_Stata.json")
    )
    assert [fx["fixture"] for fx in fixtures] == on_disk
    assert len(fixtures) >= 6
    for fx in fixtures:
        assert fx["consumers"], f"{fx['fixture']} is read by no test"
        digest = hashlib.sha256((ROOT / fx["fixture"]).read_bytes()).hexdigest()
        assert fx["sha256"] == digest
