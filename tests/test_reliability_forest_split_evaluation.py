"""The split-sample evaluation study for fixed-effects forests.

``tests/reliability/forest_split_evaluation.py`` simulates the size and
power of ``sp.rate``, ``sp.rate_split`` and ``sp.forest_policy_tree`` on a
staggered panel whose population RATE is known. This file checks that the
stored results still come from the code (the first replication of one cell
is recomputed) and pins the statements the documentation makes about them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "forest_split_evaluation.py"
RESULTS = ROOT / "reliability" / "forest_split_evaluation_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("forest_split_evaluation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _table(results, name, b):
    (cell,) = [c for c in results["cells"] if (c["study"], c["b"]) == (name, b)]
    return cell["table"]


def test_stored_prefix_is_reproduced_by_the_code(study, results):
    stored = _table(results, "rate", 0.0)
    fresh = study.one_rate((0, 0.0))
    for key, triple in fresh.items():
        # Same seed, same engine: the forest is deterministic given its
        # random_state. 1e-6 leaves room for BLAS differences in the
        # imputation solve across platforms.
        np.testing.assert_allclose(
            triple, stored[key]["prefix"][0], rtol=1e-6, atol=1e-8
        )


def test_population_values_used_by_the_study(study):
    # AUTOC and QINI of the ideal ranking when tau = 0.3 + b z, z ~ N(0, 1):
    # TOC(q) = b E[z | z above its (1 - q) quantile] = b phi(z_q) / q, so
    # AUTOC = b * int_0^1 phi(Phi^{-1}(1 - q)) / q dq and
    # QINI = b * int_0^1 phi(Phi^{-1}(1 - q)) dq = b / (2 sqrt(pi)).
    from scipy import integrate, stats

    autoc, _ = integrate.quad(
        lambda q: stats.norm.pdf(stats.norm.ppf(1 - q)) / q, 0, 1, limit=200
    )
    assert autoc == pytest.approx(study.AUTOC_PER_B, abs=5e-4)
    assert 1 / (2 * np.sqrt(np.pi)) == pytest.approx(study.QINI_PER_B, abs=5e-5)
    assert stats.norm.pdf(0) == pytest.approx(study.ORACLE_GAIN_PER_B, abs=1e-12)


def test_reusing_the_forest_ranking_over_rejects_and_splitting_does_not(results):
    null = _table(results, "rate", 0.0)
    B = results["B"]
    # A correct 5% test lands within three binomial standard errors of 0.05.
    band = 3 * np.sqrt(0.05 * 0.95 / B)
    assert null["own/AUTOC"]["excludes_zero"] > 0.05 + band
    for key in ("split_bjs/AUTOC", "split_bjs/QINI"):
        assert null[key]["excludes_zero"] < 0.05 + band
        # and the split estimate is centred on the truth, 0
        se_of_mean = null[key]["sd"] / np.sqrt(null[key]["n"])
        assert abs(null[key]["mean"]) < 3 * se_of_mean + 1e-3


def test_splitting_keeps_power_and_coverage_when_heterogeneity_is_real(results):
    alt = _table(results, "rate", 0.5)
    for key in ("split_bjs/AUTOC", "split_bjs/QINI"):
        assert alt[key]["excludes_zero"] > 0.9
        # The estimand is the RATE of the fitted rule, which cannot exceed
        # that of the ideal ranking ("truth") and, with a forest grown on
        # half the units, sits somewhat below it.
        assert 0.7 * alt[key]["truth"] < alt[key]["mean"] < alt[key]["truth"]


def test_policy_tree_gain_has_size_under_no_heterogeneity_and_power_under_it(results):
    B = results["B"]
    band = 3 * np.sqrt(0.05 * 0.95 / B)
    null = _table(results, "policy", 0.0)["split/gain"]
    assert null["excludes_zero"] < 0.05 + band
    alt = _table(results, "policy", 0.8)["split/gain"]
    assert alt["excludes_zero"] > 0.9
    # within 10% of the oracle gain 0.8 * phi(0)
    assert alt["mean"] == pytest.approx(alt["truth"], rel=0.1)
