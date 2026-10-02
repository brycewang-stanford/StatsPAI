"""Every evidence file is read by something, and Track A is fully locked.

``scripts/evidence_track_manifest.py`` is the one view over the six
places numerical evidence is stored (review item R3). It adds no lock of
its own; these tests pin the properties that should survive any new
fixture: a file nobody reads is dead evidence, and a Track A or
option-level file outside every hash lock can be edited unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evidence_track_manifest.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="source checkout only (scripts/ not installed)"
)


@pytest.fixture(scope="module")
def tracks():
    spec = importlib.util.spec_from_file_location("evidence_track_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build()["tracks"]


def test_all_six_tracks_are_inventoried(tracks):
    assert set(tracks) == {
        "r_parity",
        "stata_parity",
        "reference_parity",
        "orig_parity",
        "external_parity",
        "coverage_monte_carlo",
    }
    assert tracks["r_parity"]["files"] > 200
    assert tracks["reference_parity"]["files"] > 400


def test_no_evidence_file_is_unread(tracks):
    unnamed = {name: t["unnamed"] for name, t in tracks.items() if t["unnamed"]}
    assert not unnamed, (
        "evidence files that no test, harness or generator names "
        f"(delete them or wire them in): {unnamed}"
    )


def test_track_a_files_are_all_hash_locked(tracks):
    assert tracks["r_parity"]["unlocked"] == []
    assert tracks["r_parity"]["locked"] == tracks["r_parity"]["files"]


def test_stata_results_and_option_fixtures_are_all_hash_locked(tracks):
    """Track A results by the Tier A lock, option-level ones by the inventory."""
    stata = tracks["stata_parity"]
    assert stata["unlocked"] == [], stata["unlocked"]
    assert stata["by_lock"]["evidence_inventory"] >= 9  # 6 fixtures + 3 inputs


def test_what_the_manuscript_tabulates_is_frozen(tracks):
    """Original-data results and the coverage runs are in the JSS manifest."""
    orig = tracks["orig_parity"]
    results = [p for p in orig["unlocked"] if "/results/" in p]
    assert results == [], results
    assert tracks["coverage_monte_carlo"]["unlocked"] == []
