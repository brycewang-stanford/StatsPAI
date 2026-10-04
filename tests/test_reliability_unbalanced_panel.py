"""The unbalanced-panel coverage study: reproducible, and what the docs rest on.

``tests/reliability/unbalanced_panel.py`` simulates the coverage of the
95% interval for the overall ATT of four staggered-adoption estimators
under four patterns of missing unit-period cells. This file checks the
stored design, that one cell reproduces its stored prefix, that the
statements in ``tests/reliability/README.md`` hold with their Monte
Carlo error, and that ``sp.callaway_santanna`` says what its
repeated-cross-section route assumes.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

import statspai as sp

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "unbalanced_panel.py"
RESULTS = ROOT / "reliability" / "unbalanced_panel_results.json"
WITHIN = ("cs", "bjs", "twfe")


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("unbalanced_panel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cell(results, pattern, N):
    (cell,) = [c for c in results["cells"] if (c["pattern"], c["N"]) == (pattern, N)]
    return cell


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 1000 and results["level"] == 0.95
    keys = {(c["pattern"], c["N"]) for c in results["cells"]}
    assert keys == {(p, n) for p in study.PATTERNS for n in study.N_VALUES}
    for cell in results["cells"]:
        for name in study.ESTIMATORS:
            assert cell[name]["n_fitted"] == 1000 and not cell[name]["refused"]


def test_a_cell_reproduces_its_stored_prefix(study, results):
    fresh = study.run_cell(100, "attrit_level", study.PREFIX)
    stored = _cell(results, "attrit_level", 100)
    for name in study.ESTIMATORS:
        assert fresh[name]["prefix_hits"] == stored[name]["prefix_hits"], name


def test_within_unit_estimators_survive_ignorable_missingness(results):
    """Balanced, missing at random, attrition on the unit's level."""
    for pattern in ("balanced", "mcar", "attrit_level"):
        for N in (100, 400):
            cell = _cell(results, pattern, N)
            for name in WITHIN:
                cov, se = cell[name]["coverage"], cell[name]["mc_se"]
                assert abs(cell[name]["bias"]) < 0.02, (pattern, N, name)
                assert 0.95 - 3 * se - 0.005 < cov < 0.975, (pattern, N, name, cov)


def test_group_mean_route_matches_within_route_when_cells_are_missing_at_random(
    results,
):
    for N in (100, 400):
        cell = _cell(results, "mcar", N)["cs_rcs"]
        assert abs(cell["bias"]) < 0.02
        assert abs(cell["coverage"] - 0.95) < 3 * cell["mc_se"]


def test_group_mean_route_fails_when_units_leave_by_level(results):
    small = _cell(results, "attrit_level", 100)["cs_rcs"]
    large = _cell(results, "attrit_level", 400)["cs_rcs"]
    assert -0.45 < small["bias"] < -0.35 and -0.45 < large["bias"] < -0.35
    assert small["coverage"] < 0.56 and large["coverage"] < 0.06


def test_nothing_survives_attrition_on_the_outcome(study, results):
    for N in (100, 400):
        cell = _cell(results, "attrit_outcome", N)
        for name in study.ESTIMATORS:
            assert cell[name]["bias"] > 0.35, (N, name)
            assert cell[name]["coverage"] < (0.5 if N == 100 else 0.05), (N, name)


# --------------------------------------------------------------------- #
# What sp.callaway_santanna says
# --------------------------------------------------------------------- #


def test_unbalanced_warning_names_the_composition_assumption(study):
    df = study.draw(100, "attrit_level", 1)
    with pytest.warns(UserWarning, match="composition"):
        sp.callaway_santanna(df, y="y", g="g", t="t", i="id")


def test_group_mean_route_records_its_assumption(study):
    df = study.draw(100, "attrit_level", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rcs = sp.callaway_santanna(
            df, y="y", g="g", t="t", i="id", allow_unbalanced_panel=True
        )
        within = sp.callaway_santanna(df, y="y", g="g", t="t", i="id")
        balanced = sp.callaway_santanna(
            study.draw(100, "balanced", 1),
            y="y",
            g="g",
            t="t",
            i="id",
            allow_unbalanced_panel=True,
        )
    assert "composition" in rcs.model_info["unbalanced_assumption"]
    assert "unbalanced_assumption" not in within.model_info
    assert "unbalanced_assumption" not in balanced.model_info
    # the two routes disagree on this draw by far more than their SEs
    a = float(sp.aggte(rcs, type="simple").estimate)
    b = float(sp.aggte(within, type="simple").estimate)
    assert b - a > 0.2
