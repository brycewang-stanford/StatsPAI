"""The few-cluster size study: stored results are reproducible and say what we say.

``tests/reliability/few_clusters.py`` simulates the rejection rate of a
true null under four inference methods across 16 designs (2,000
replications each) and writes ``few_clusters_results.json``. This file
checks three things: the stored file has the declared shape; one cell
recomputed on its first 60 replications reproduces the stored counts
exactly (same seeds, deterministic); and the statements the package makes
about few clusters (``FEW_CLUSTERS_HINT``) are the ones the numbers
support, each with its Monte Carlo error taken into account.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from statspai.core._agent_summary import FEW_CLUSTERS_HINT

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "few_clusters.py"
RESULTS = ROOT / "reliability" / "few_clusters_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("few_clusters", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cell(results, G, treated, sizes):
    (cell,) = [
        c
        for c in results["cells"]
        if (c["G"], c["treated"], c["sizes"]) == (G, treated, sizes)
    ]
    return cell


def _rate(results, G, treated, sizes, method):
    c = _cell(results, G, treated, sizes)[method]
    return c["rejection_rate"], c["mc_se"]


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 2000 and results["alpha"] == 0.05
    keys = {(c["G"], c["treated"], c["sizes"]) for c in results["cells"]}
    assert keys == {
        (g, t, s) for g in study.G_VALUES for t in study.TREATED for s in study.SIZES
    }
    assert len(results["cells"]) == 16
    for c in results["cells"]:
        for m in study.METHODS:
            assert 0.0 <= c[m]["rejection_rate"] <= 1.0


def test_a_cell_reproduces_its_stored_prefix(study, results):
    """Same seeds, same counts: the stored file came from this script."""
    fresh = study.run_cell(6, "two", "unbalanced", study.PREFIX)
    stored = _cell(results, 6, "two", "unbalanced")
    for m in study.METHODS:
        assert (
            fresh[m]["prefix_rejections"] == stored[m]["prefix_rejections"]
        ), f"{m}: recomputed prefix differs from the stored one"


def test_balanced_half_treated_is_where_every_method_settles(results):
    for method in ("cr1", "cr3", "wild"):
        rate, se = _rate(results, 40, "half", "balanced", method)
        assert abs(rate - 0.05) < 3 * se + 0.005, (method, rate)
    # ... and CR1 with t(G - 1) is already mild at six clusters
    rate, _ = _rate(results, 6, "half", "balanced", "cr1")
    assert 0.06 < rate < 0.11


def test_wild_bootstrap_is_near_nominal_with_similar_clusters(results):
    for G in (6, 10, 20, 40):
        rate, se = _rate(results, G, "half", "balanced", "wild")
        assert 0.04 - 2 * se < rate < 0.08 + 2 * se, (G, rate)


def test_wild_bootstrap_almost_never_rejects_with_two_treated_clusters(results):
    for G in (10, 20, 40):
        rate, _ = _rate(results, G, "two", "balanced", "wild")
        assert rate < 0.02, (G, rate)
    # while the analytic variances over-reject several-fold
    rate, _ = _rate(results, 40, "two", "balanced", "cr1")
    assert rate > 0.25


def test_one_dominant_cluster_breaks_cr1_and_the_bootstrap_but_not_cr3(results):
    for G in (20, 40):
        cr1, _ = _rate(results, G, "half", "unbalanced", "cr1")
        wild, se_w = _rate(results, G, "half", "unbalanced", "wild")
        cr3, se_3 = _rate(results, G, "half", "unbalanced", "cr3")
        assert cr1 > 0.20, (G, cr1)
        assert wild - 2 * se_w > 0.08, (G, wild)
        assert 0.03 < cr3 < 0.08 + 2 * se_3, (G, cr3)
    # more clusters do not help CR1 here: the large one still holds half
    assert (
        _rate(results, 40, "half", "unbalanced", "cr1")[0]
        > _rate(results, 6, "half", "unbalanced", "cr1")[0]
    )


def test_the_hint_says_what_the_study_found():
    text = FEW_CLUSTERS_HINT
    assert "one or two" in text and "almost never rejects" in text
    assert "one cluster" in text and "cr3" in text
    assert "few_clusters_results.json" in text
    assert "keeps correct size" not in text
