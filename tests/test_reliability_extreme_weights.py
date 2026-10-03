"""The extreme-weights coverage study: reproducible, and what the warnings rest on.

``tests/reliability/extreme_weights.py`` simulates the coverage of the 95%
interval of a weighted-regression slope under four variance options, as
the weights grow more dispersed and under two readings of what a weight
is. This file checks that the stored file has the declared shape, that
one cell recomputed on its first 60 replications reproduces the stored
counts, that the statements in ``tests/reliability/README.md`` hold with
their Monte Carlo error, and that ``sp.regress`` raises the two warnings
the study motivates and no others.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._agent_summary import kish_effective_n
from statspai.exceptions import AssumptionWarning

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "extreme_weights.py"
RESULTS = ROOT / "reliability" / "extreme_weights_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("extreme_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cov(results, errors, n, sigma, variance):
    (cell,) = [
        c
        for c in results["cells"]
        if (c["errors"], c["n"], c["sigma"]) == (errors, n, sigma)
    ]
    return cell[variance]["coverage"], cell[variance]["mc_se"], cell["kish_n_median"]


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 2000 and results["level"] == 0.95
    keys = {(c["errors"], c["n"], c["sigma"]) for c in results["cells"]}
    assert keys == {
        (e, n, s) for e in study.ERRORS for n in study.N_VALUES for s in study.SIGMAS
    }


def test_a_cell_reproduces_its_stored_prefix(study, results):
    fresh = study.run_cell(200, 2.0, "sampling", study.PREFIX)
    (stored,) = [
        c
        for c in results["cells"]
        if (c["errors"], c["n"], c["sigma"]) == ("sampling", 200, 2.0)
    ]
    for v in study.VARIANCES:
        assert fresh[v]["prefix_hits"] == stored[v]["prefix_hits"], v


def test_equal_weights_cover_at_the_nominal_level(results):
    for errors in ("precision", "sampling"):
        for n in (200, 1000):
            for v in ("classical", "hc1", "hc2", "hc3"):
                cov, se, _ = _cov(results, errors, n, 0.0, v)
                assert abs(cov - 0.95) < 3 * se, (errors, n, v, cov)


def test_classical_is_right_for_precisions_and_wrong_for_sampling_weights(results):
    for n in (200, 1000):
        for sigma in (1.0, 2.0):
            cov, se, _ = _cov(results, "precision", n, sigma, "classical")
            assert abs(cov - 0.95) < 3 * se, (n, sigma, cov)
        mid, _, _ = _cov(results, "sampling", n, 1.0, "classical")
        far, _, _ = _cov(results, "sampling", n, 2.0, "classical")
        assert 0.70 < mid < 0.85
        assert far < 0.55


def test_hc1_shortens_with_the_kish_size_and_hc3_does_not(results):
    for errors in ("precision", "sampling"):
        for n in (200, 1000):
            hc1, _, kish = _cov(results, errors, n, 2.0, "hc1")
            hc3, se3, _ = _cov(results, errors, n, 2.0, "hc3")
            assert kish < 100
            assert hc1 < 0.935, (errors, n, hc1)
            assert hc3 > 0.925, (errors, n, hc3)
            assert hc3 > hc1
    # with a Kish size in the hundreds HC1 is fine
    hc1, se, kish = _cov(results, "sampling", 1000, 1.0, "hc1")
    assert kish > 300 and abs(hc1 - 0.95) < 3 * se + 0.005


# ---------------------------------------------------------------------------
# The diagnostic and the warnings
# ---------------------------------------------------------------------------


def _frame(sigma, n=200, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "y": 1 + 0.5 * x + rng.normal(size=n),
            "w": np.exp(rng.normal(scale=sigma, size=n)),
        }
    )


def test_kish_size_is_n_for_equal_weights_and_small_for_one_dominant_weight():
    assert kish_effective_n(np.ones(50)) == pytest.approx(50.0)
    w = np.r_[1000.0, np.ones(99)]
    assert kish_effective_n(w) == pytest.approx(w.sum() ** 2 / (w**2).sum())
    assert kish_effective_n(w) < 1.3
    assert kish_effective_n([]) == 0.0


def test_classical_with_dispersed_weights_says_what_it_assumes():
    df = _frame(2.0)
    with pytest.warns(AssumptionWarning, match="precisions") as caught:
        res = sp.regress("y ~ x", df, weights="w")
    diag = caught[0].message.diagnostics
    expected = kish_effective_n(df["w"])
    assert diag["n_effective_weights"] == pytest.approx(expected)
    assert diag["kish_ratio"] == pytest.approx(expected / 200)
    assert res.model_info["n_effective_weights"] == pytest.approx(expected)


@pytest.mark.parametrize("kw", [dict(robust="hc1"), dict(vce="hc2")])
def test_hc1_and_hc2_with_a_small_effective_sample_point_at_hc3(kw):
    with pytest.warns(AssumptionWarning, match="too short") as caught:
        sp.regress("y ~ x", _frame(2.0), weights="w", **kw)
    assert "hc3" in caught[0].message.recovery_hint


@pytest.mark.parametrize(
    "sigma,kw",
    [
        (2.0, dict(vce="hc3")),
        (0.3, {}),
        (0.3, dict(robust="hc1")),
        (0.0, {}),
    ],
)
def test_no_warning_where_the_study_found_nominal_coverage(sigma, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = sp.regress("y ~ x", _frame(sigma), weights="w", **kw)
    assert res.model_info["n_effective_weights"] > 0


def test_unweighted_fits_carry_no_weight_diagnostic():
    res = sp.regress("y ~ x", _frame(1.0))
    assert "n_effective_weights" not in res.model_info
