"""The DML learner study: reproducible, and what its statements rest on.

``tests/reliability/dml_learners.py`` simulates the coverage of the 95%
interval of ``sp.dml(model='plr')`` under four first-stage learners and
of ``sp.dml_model_averaging``, with linear and nonlinear confounding.
This file checks the stored design, that one cell reproduces its stored
prefix, and that the statements in ``tests/reliability/README.md`` hold
with their Monte Carlo error.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "dml_learners.py"
RESULTS = ROOT / "reliability" / "dml_learners_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("dml_learners", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cell(results, shape, n, learner):
    (cell,) = [
        c
        for c in results["cells"]
        if (c["shape"], c["n"], c["learner"]) == (shape, n, learner)
    ]
    return cell


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 300 and results["level"] == 0.95
    keys = {(c["shape"], c["n"], c["learner"]) for c in results["cells"]}
    assert keys == {
        (s, n, k) for s in study.SHAPES for n in study.N_VALUES for k in study.METHODS
    }


def test_a_cell_reproduces_its_stored_prefix(study, results):
    fresh = study.run_cell(500, "nonlinear", "lasso", study.PREFIX)
    stored = _cell(results, "nonlinear", 500, "lasso")
    assert fresh["prefix_hits"] == stored["prefix_hits"]
    fresh = study.run_cell(500, "linear", "ols", study.PREFIX)
    stored = _cell(results, "linear", 500, "ols")
    assert fresh["prefix_hits"] == stored["prefix_hits"]


def test_linear_learners_are_right_only_when_the_confounding_is_linear(results):
    for n in (500, 2000):
        for learner in ("ols", "lasso"):
            good = _cell(results, "linear", n, learner)
            assert abs(good["bias"]) < 0.005
            assert abs(good["coverage"] - 0.95) < 3 * good["mc_se"]
            bad = _cell(results, "nonlinear", n, learner)
            # the bias does not shrink with n: it is not a small-sample matter
            assert 0.70 < bad["bias"] < 0.74 and bad["coverage"] == 0.0


def test_tree_learners_trade_a_small_bias_for_robustness(results):
    for n in (500, 2000):
        for learner in ("rf", "gbm"):
            lin = _cell(results, "linear", n, learner)
            assert -0.03 < lin["bias"] < 0 and 0.89 < lin["coverage"] < 0.95
    # nonlinear: the forest is the slower of the two to shed its bias
    rf = [_cell(results, "nonlinear", n, "rf") for n in (500, 2000)]
    gbm = [_cell(results, "nonlinear", n, "gbm") for n in (500, 2000)]
    assert rf[0]["bias"] > 0.12 and rf[0]["coverage"] < 0.5
    assert 0.03 < rf[1]["bias"] < 0.06 and rf[1]["coverage"] < 0.73
    assert 0 < gbm[1]["bias"] < gbm[0]["bias"] < 0.07
    assert all(0.85 < c["coverage"] < 0.94 for c in gbm)


def test_model_averaging_is_never_badly_wrong_and_not_a_cure(results):
    for n in (500, 2000):
        lin = _cell(results, "linear", n, "stacking")
        assert abs(lin["bias"]) < 0.005
        assert abs(lin["coverage"] - 0.95) < 3 * lin["mc_se"]
    small = _cell(results, "nonlinear", 500, "stacking")
    large = _cell(results, "nonlinear", 2000, "stacking")
    # far from the linear learners' +0.72, and shrinking with n ...
    assert 0.06 < small["bias"] < 0.11 and 0.01 < large["bias"] < 0.03
    # ... but below nominal, and no better than boosting alone at n = 500
    assert 0.65 < small["coverage"] < 0.80 and 0.82 < large["coverage"] < 0.92
    assert small["coverage"] < _cell(results, "nonlinear", 500, "gbm")["coverage"]


def test_reported_standard_error_tracks_the_sampling_spread(results):
    """The interval is wrong because of bias, not because the SE is."""
    for cell in results["cells"]:
        assert 0.8 < cell["mean_se"] / cell["sd"] < 1.2, (
            cell["shape"],
            cell["n"],
            cell["learner"],
        )
