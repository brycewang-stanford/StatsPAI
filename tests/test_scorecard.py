"""``scripts/build_scorecard.py`` reports the same numbers as the parity API.

The scorecard is the page a sceptical reader is sent to, so its figures must
be the ones ``sp.parity_summary()`` / ``sp.parity_status()`` return, not a
second computation that can drift from them. The page itself is refreshed at
release (``release_gate --fix-census``) and is not compared here.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

import statspai as sp

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scorecard():
    spec = importlib.util.spec_from_file_location(
        "statspai_build_scorecard_for_tests",
        REPO_ROOT / "scripts" / "build_scorecard.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cell(page: str, label: str) -> int:
    match = re.search(rf"^\| {re.escape(label)} \| ([\d,]+) \|", page, re.MULTILINE)
    assert match, label
    return int(match.group(1).replace(",", ""))


def test_correctness_counts_equal_the_estimator_stratum(scorecard):
    page = scorecard.render()
    stratum = sp.parity_summary()["denominators"]["estimator"]
    assert _cell(page, "Registered estimators") == stratum["total"]
    assert _cell(page, "With numerical evidence of any kind") == stratum["verified"]
    assert (
        _cell(page, "Same data, matched to an R / Stata / author reference (T2)")
        == stratum["cross_language"]
    )


def test_unverified_list_is_every_unverified_estimator(scorecard):
    rows = scorecard._estimator_rows()
    listed = {r["name"] for r in rows if r["status"] == "unverified"}
    assert listed == {
        r["name"] for r in rows if sp.parity_status(r["name"])["status"] == "unverified"
    }
    total = sp.parity_summary()["denominators"]["estimator"]
    assert len(listed) == total["total"] - total["verified"]


def test_coverage_band_is_the_coverage_suites_band(scorecard):
    from tests.coverage_monte_carlo.test_coverage import _wilson_bounds

    for draws in (300, 1000):
        assert scorecard._wilson(0.95, draws) == pytest.approx(_wilson_bounds(draws))


def test_coverage_verdicts_follow_the_committed_rates(scorecard):
    rows = json.loads(
        (scorecard.TRACK_B / "coverage_b1000.json").read_text(encoding="utf-8")
    )
    lo, hi = scorecard._wilson(0.95, 1000)
    inside = sum(lo <= r["rate"] <= hi + scorecard.UPPER_SLACK for r in rows)
    assert f"**{inside} of {len(rows)} rows inside the band.**" in "\n".join(
        scorecard._calibration()
    )


def test_speed_rows_use_the_reference_compare_perf_declares(scorecard):
    # Read ESTIMATORS without importing compare_perf: importing it creates
    # Paper-JSS/manuscript/{tables,figures}, and in a worktree that empty
    # tree turns the JSS tests' designed skips into failures.
    import ast

    tree = ast.parse(
        (REPO_ROOT / "tests" / "perf" / "compare_perf.py").read_text(encoding="utf-8")
    )
    node = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "ESTIMATORS" for t in n.targets)
    )
    estimators = ast.literal_eval(node)
    declared = {stem: meta.get("ref_side", "R") for stem, meta in estimators.items()}
    for stem, _label, _ref, side in scorecard.PERF_MODULES:
        assert declared[stem] == side, stem
