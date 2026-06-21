"""Tests for summary-data meta-analysis (``sp.meta_analysis``).

Validation against closed-form inverse-variance pooling, the
DerSimonian-Laird heterogeneity formulas, and Egger's test behaviour under
symmetric vs. asymmetric funnels.
"""

import numpy as np
import pytest

import statspai as sp


def test_fixed_effect_matches_inverse_variance():
    y = [0.10, 0.25, -0.05, 0.30, 0.15]
    s = [0.05, 0.10, 0.08, 0.12, 0.06]
    r = sp.meta_analysis(y, s, method="fixed")
    w = 1.0 / np.array(s) ** 2
    fe = float(np.sum(w * np.array(y)) / np.sum(w))
    fe_se = float(np.sqrt(1.0 / np.sum(w)))
    assert r.estimate == pytest.approx(fe, abs=1e-10)
    assert r.se == pytest.approx(fe_se, abs=1e-10)
    assert r.q_df == 4
    assert np.sum(r.weights) == pytest.approx(1.0, abs=1e-9)


def test_homogeneous_studies_zero_tau2():
    rng = np.random.default_rng(0)
    se = rng.uniform(0.05, 0.15, 25)
    y = 0.4 + rng.normal(0, se)  # only sampling variation
    r = sp.meta_analysis(y, se)
    assert r.tau2 == pytest.approx(0.0, abs=1e-3)
    assert r.i2 < 0.25
    assert r.random_estimate == pytest.approx(r.fixed_estimate, abs=1e-3)
    assert r.estimate == pytest.approx(0.4, abs=0.05)


def test_heterogeneous_studies_inflate_random_se():
    rng = np.random.default_rng(1)
    mu = rng.normal(0.4, 0.3, 25)  # true between-study heterogeneity
    se = rng.uniform(0.05, 0.15, 25)
    y = mu + rng.normal(0, se)
    r = sp.meta_analysis(y, se)
    assert r.tau2 > 0
    assert r.i2 > 0.5
    assert r.random_se > r.fixed_se
    # prediction interval is wider than the CI of the pooled mean
    pi = r.prediction_interval
    assert pi is not None
    assert (pi[1] - pi[0]) > (r.ci[1] - r.ci[0])


def test_egger_symmetric_vs_asymmetric():
    rng = np.random.default_rng(2)
    se = rng.uniform(0.05, 0.30, 50)
    y_sym = 0.3 + rng.normal(0, se)
    p_sym = sp.meta_analysis(y_sym, se).egger_test()["p_value"]
    assert p_sym > 0.10

    se2 = rng.uniform(0.05, 0.30, 50)
    y_asym = 0.3 + 2.0 * se2 + rng.normal(0, se2)  # small-study effect
    p_asym = sp.meta_analysis(y_asym, se2).egger_test()["p_value"]
    assert p_asym < 0.05


def test_validates_inputs():
    with pytest.raises(ValueError, match="at least 2"):
        sp.meta_analysis([1.0], [0.1])
    with pytest.raises(ValueError, match="positive"):
        sp.meta_analysis([1.0, 2.0], [0.1, -0.1])
    with pytest.raises(ValueError, match="DL|fixed"):
        sp.meta_analysis([1.0, 2.0], [0.1, 0.2], method="bogus")


def test_summary_and_repr():
    r = sp.meta_analysis([0.1, 0.2, 0.15], [0.05, 0.06, 0.07])
    assert "Meta-analysis" in r.summary()
    assert "I^2" in r.summary()
    assert "MetaAnalysisResult" in repr(r)
