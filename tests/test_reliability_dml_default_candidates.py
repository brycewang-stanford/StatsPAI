"""The study that kept the ``sp.dml(model='plr')`` default learner.

``tests/reliability/dml_default_candidates.py`` compared the
gradient-boosting default with four replacements on seven designs, under
a decision rule fixed before the first run, and short stacking, added
after it. This file checks the stored design, that two cells reproduce
their stored prefix, that the rule as fixed does not separate the
candidates (so the decision rests on the pooled evidence the study's
docstring states), and the numbers the CHANGELOG, MIGRATION and the guide
quote.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "dml_default_candidates.py"
RESULTS = ROOT / "reliability" / "dml_default_candidates_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("dml_default_candidates", SCRIPT)
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


def _score(results, learner, shapes=None):
    return min(
        c["coverage"]
        for c in results["cells"]
        if c["learner"] == learner and (shapes is None or c["shape"] in shapes)
    )


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 300 and results["level"] == 0.95
    keys = {(c["shape"], c["n"], c["learner"]) for c in results["cells"]}
    assert keys == {(s, n, k) for s, n in study.CELLS for k in study.METHODS}
    assert len(study.CELLS) == 7 and set(study.METHODS) == {
        "gbm",
        "gbm_es",
        "hgb_es",
        "lasso_splines",
        "stack",
        "short_stacking",
    }


@pytest.mark.parametrize("learner", ["gbm", "lasso_splines"])  # the cheap ones
def test_a_cell_reproduces_its_stored_prefix(study, results, learner):
    fresh = study._chunk(("trackb", 500, learner, 0, study.CHUNK))
    assert sum(fresh["hit"]) == _cell(results, "trackb", 500, learner)["prefix_hits"]


def test_the_rule_as_fixed_does_not_separate_the_candidates(results):
    """Rule 2 replaces only if the best worst-cell coverage beats the
    default's by more than 0.025. The interaction cells pull every score
    to near 0, so no candidate clears the margin."""
    learners = sorted({c["learner"] for c in results["cells"]})
    scores = {k: _score(results, k) for k in learners}
    assert max(scores.values()) == pytest.approx(0.023, abs=1e-3)  # hgb_es
    assert scores["gbm"] == 0.0 and scores["short_stacking"] == 0.0
    assert max(scores.values()) - scores["gbm"] <= 0.025


def test_short_stacking_against_the_default(results):
    """The trade the owner accepted: better with linear confounding and
    on the Track B design, worse with smooth additive confounding in ten
    covariates, no help with an interaction, several times slower."""
    for shape, ns in (("linear", (500, 2000)), ("trackb", (500,))):
        for n in ns:
            ss = _cell(results, shape, n, "short_stacking")
            gbm = _cell(results, shape, n, "gbm")
            assert ss["coverage"] > gbm["coverage"]
            assert abs(ss["coverage"] - 0.95) < 2.5 * ss["mc_se"] + 0.01
    for n, ss_cov, gbm_cov in ((500, 0.720, 0.863), (2000, 0.860, 0.893)):
        assert _cell(results, "nonlinear", n, "short_stacking")[
            "coverage"
        ] == pytest.approx(ss_cov, abs=1e-3)
        assert _cell(results, "nonlinear", n, "gbm")["coverage"] == pytest.approx(
            gbm_cov, abs=1e-3
        )
    for n in (500, 2000):
        ss = _cell(results, "interaction", n, "short_stacking")
        gbm = _cell(results, "interaction", n, "gbm")
        assert ss["coverage"] == 0.0 and gbm["coverage"] < 0.03
        assert ss["seconds_per_fit"] > 5 * gbm["seconds_per_fit"]


def test_the_decision_does_not_rest_on_the_interaction_designs(results):
    """Without the cells where every learner is near 0, nothing beats the
    default by the margin either."""
    smooth = {"linear", "nonlinear", "trackb"}
    learners = sorted({c["learner"] for c in results["cells"]})
    scores = {k: _score(results, k, smooth) for k in learners}
    best = max(scores, key=scores.get)
    assert best == "lasso_splines"
    assert scores["lasso_splines"] == pytest.approx(0.880, abs=1e-3)
    assert scores["gbm"] == pytest.approx(0.863, abs=1e-3)
    assert scores[best] - scores["gbm"] <= 0.025


def test_numbers_quoted_in_the_docs(results):
    # the summary() note: 0.86 to 0.92 with smooth confounding, < 0.03 with
    # an interaction (the boosting cells of this study)
    smooth = [
        c["coverage"]
        for c in results["cells"]
        if c["learner"] == "gbm" and c["shape"] != "interaction"
    ]
    assert 0.86 <= min(smooth) and max(smooth) < 0.925
    inter = [
        c["coverage"]
        for c in results["cells"]
        if c["shape"] == "interaction" and c["learner"] == "gbm"
    ]
    assert max(inter) < 0.03
    # the guide: every learner under 4% with an interaction
    assert all(
        c["coverage"] < 0.04 for c in results["cells"] if c["shape"] == "interaction"
    )
    # early stopping stops too soon on smooth nonlinear confounding
    for k, lo, hi in (("gbm_es", 0.68, 0.77), ("hgb_es", 0.21, 0.47)):
        cov = [_cell(results, "nonlinear", n, k)["coverage"] for n in (500, 2000)]
        assert lo < min(cov) and max(cov) < hi


def test_reported_standard_error_tracks_the_sampling_spread(results):
    """The intervals fail through bias in the nuisance fits, not the SE.
    The two exceptions are early-stopped boosting (alone and inside the
    Lasso stack) with an interaction at n = 2000, where the estimate
    varies half again as much as the SE says."""
    off = {
        (c["shape"], c["n"], c["learner"])
        for c in results["cells"]
        if not 0.85 < c["mean_se"] / c["sd"] < 1.15
    }
    assert off == {("interaction", 2000, "gbm_es"), ("interaction", 2000, "stack")}
    for shape, n, k in off:
        assert (
            0.6
            < _cell(results, shape, n, k)["mean_se"] / _cell(results, shape, n, k)["sd"]
            < 0.7
        )
