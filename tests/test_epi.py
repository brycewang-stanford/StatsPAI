"""Tests for the epidemiology primitives (``sp.epi``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Odds ratio
# ---------------------------------------------------------------------------


def test_odds_ratio_point_estimate_matches_formula():
    # 2x2: a=50, b=20, c=30, d=40
    # OR = (50*40)/(20*30) = 2000/600 = 3.333...
    r = sp.epi.odds_ratio(50, 20, 30, 40)
    assert r.estimate == pytest.approx(3.333333333, rel=1e-6)
    assert r.a == 50 and r.b == 20 and r.c == 30 and r.d == 40
    assert r.ci[0] < r.estimate < r.ci[1]
    assert 0 < r.p_value < 1


def test_odds_ratio_woolf_ci_is_log_symmetric():
    r = sp.epi.odds_ratio(50, 20, 30, 40, method="woolf")
    midpoint_log = (math.log(r.ci[0]) + math.log(r.ci[1])) / 2
    assert midpoint_log == pytest.approx(math.log(r.estimate), rel=1e-6)


def test_odds_ratio_haldane_correction_for_zero_cell():
    # Zero cell must be handled (Haldane add 0.5)
    r = sp.epi.odds_ratio(0, 10, 5, 20)
    assert math.isfinite(r.estimate)
    assert math.isfinite(r.se_log)


def test_odds_ratio_exact_method_runs():
    r = sp.epi.odds_ratio(50, 20, 30, 40, method="exact")
    assert r.method == "exact"
    assert math.isfinite(r.estimate)


def test_odds_ratio_accepts_2x2_array():
    r = sp.epi.odds_ratio([[50, 20], [30, 40]])
    assert r.estimate == pytest.approx(3.333333333, rel=1e-6)


# ---------------------------------------------------------------------------
# Relative risk
# ---------------------------------------------------------------------------


def test_relative_risk_point_estimate():
    # 50/70 vs 30/70 -> 0.714 / 0.429 -> 1.667
    r = sp.epi.relative_risk(50, 20, 30, 40)
    assert r.estimate == pytest.approx((50 / 70) / (30 / 70), rel=1e-6)
    assert r.risk_exposed == pytest.approx(50 / 70, rel=1e-6)
    assert r.risk_unexposed == pytest.approx(30 / 70, rel=1e-6)


def test_prevalence_ratio_delegates_to_rr():
    a = sp.epi.relative_risk(50, 20, 30, 40)
    b = sp.epi.prevalence_ratio(50, 20, 30, 40)
    assert a.estimate == pytest.approx(b.estimate)
    assert b.method == "prevalence-ratio"


# ---------------------------------------------------------------------------
# Risk difference
# ---------------------------------------------------------------------------


def test_risk_difference_wald():
    r = sp.epi.risk_difference(50, 50, 30, 70, method="wald")
    assert r.estimate == pytest.approx(0.5 - 0.3, rel=1e-6)
    assert r.ci[0] < r.estimate < r.ci[1]


def test_risk_difference_newcombe_narrower_or_comparable():
    rw = sp.epi.risk_difference(50, 50, 30, 70, method="wald")
    rn = sp.epi.risk_difference(50, 50, 30, 70, method="newcombe")
    # Newcombe CI shouldn't be wildly off; both should contain the estimate
    for r in (rw, rn):
        assert r.ci[0] < r.estimate < r.ci[1]


# ---------------------------------------------------------------------------
# Attributable risk / PAF
# ---------------------------------------------------------------------------


def test_attributable_risk_levin():
    r = sp.epi.attributable_risk(50, 20, 30, 40)
    # PAF should be in [0, 1] when RR > 1
    assert 0 < r.paf < 1
    assert 0 < r.ar_exposed < 1
    # PAF CI should bracket the point
    assert r.paf_ci[0] <= r.paf <= r.paf_ci[1] + 1e-6


# ---------------------------------------------------------------------------
# Incidence rate ratio
# ---------------------------------------------------------------------------


def test_incidence_rate_ratio_exact():
    r = sp.epi.incidence_rate_ratio(20, 100, 10, 100, method="exact")
    assert r.estimate == pytest.approx(2.0, rel=1e-6)
    assert r.ci[0] < 2.0 < r.ci[1]
    assert r.method == "exact"


def test_incidence_rate_ratio_wald():
    r = sp.epi.incidence_rate_ratio(20, 100, 10, 100, method="wald")
    assert r.estimate == pytest.approx(2.0, rel=1e-6)


def test_incidence_rate_ratio_rejects_zero_person_time():
    with pytest.raises(ValueError):
        sp.epi.incidence_rate_ratio(5, 0, 5, 10)


# ---------------------------------------------------------------------------
# NNT
# ---------------------------------------------------------------------------


def test_number_needed_to_treat_positive():
    r = sp.epi.number_needed_to_treat(30, 70, 50, 50)  # RD = 0.3 - 0.5 = -0.2
    assert r.estimate == pytest.approx(5.0, rel=1e-6)
    assert r.risk_difference < 0


# ---------------------------------------------------------------------------
# Mantel-Haenszel
# ---------------------------------------------------------------------------


def test_mantel_haenszel_or_homogeneous_strata():
    # Two strata with the same OR=2 should give MH-OR ~= 2
    strat1 = [[20, 10], [10, 10]]  # OR = 2
    strat2 = [[40, 20], [20, 20]]  # OR = 2
    mh = sp.epi.mantel_haenszel([strat1, strat2], measure="OR")
    assert mh.estimate == pytest.approx(2.0, rel=0.1)
    assert mh.n_strata == 2
    assert mh.homogeneity_p > 0.05  # homogeneous


def test_mantel_haenszel_rr():
    strat1 = [[20, 10], [10, 10]]
    strat2 = [[40, 20], [20, 20]]
    mh = sp.epi.mantel_haenszel([strat1, strat2], measure="RR")
    assert mh.measure == "RR"
    assert mh.estimate > 1


def test_breslow_day_test_runs():
    strat1 = [[20, 10], [10, 10]]
    strat2 = [[5, 20], [20, 5]]  # Very different OR
    chi2, p = sp.epi.breslow_day_test([strat1, strat2])
    assert chi2 > 0
    assert 0 <= p <= 1


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


def test_direct_standardize_simple():
    events = [10, 20, 30]
    pop = [1000, 1000, 1000]
    weights = [0.3, 0.4, 0.3]
    r = sp.epi.direct_standardize(events, pop, weights)
    expected = 0.3 * 0.01 + 0.4 * 0.02 + 0.3 * 0.03
    assert r.rate == pytest.approx(expected, rel=1e-6)
    assert r.ci[0] < r.rate < r.ci[1]


def test_indirect_standardize_smr():
    # Reference: 3 strata with rates 0.01, 0.02, 0.03
    # Study population distribution same as reference -> expected matches
    # Observed = expected -> SMR = 1
    events_ref = [10, 20, 30]
    pop_ref = [1000, 1000, 1000]
    pop_study = [1000, 1000, 1000]
    expected = 0.01 * 1000 + 0.02 * 1000 + 0.03 * 1000  # 60
    r = sp.epi.indirect_standardize(
        observed=60,
        events_reference=events_ref,
        population_reference=pop_ref,
        population_study=pop_study,
    )
    assert r.smr == pytest.approx(1.0, rel=1e-6)
    assert r.ci[0] < 1.0 < r.ci[1]


# ---------------------------------------------------------------------------
# Bradford-Hill
# ---------------------------------------------------------------------------


def test_bradford_hill_strong_support():
    r = sp.epi.bradford_hill(
        strength=1.0,
        consistency=1.0,
        temporality=1.0,
        biological_gradient=1.0,
        plausibility=1.0,
        specificity=0.5,
        coherence=1.0,
        experiment=1.0,
        analogy=0.5,
    )
    assert r.total > 0.75 * r.max_total
    assert "STRONG" in r.verdict


def test_bradford_hill_missing_temporality_flags():
    r = sp.epi.bradford_hill(
        strength=1.0,
        consistency=1.0,
        temporality=0.0,  # fails prerequisite
    )
    assert "temporality" in r.missing_prerequisites
    assert "INSUFFICIENT" in r.verdict


def test_bradford_hill_rejects_out_of_range():
    with pytest.raises(ValueError):
        sp.epi.bradford_hill(strength=2.0)


def test_bradford_hill_summary_runs():
    r = sp.epi.bradford_hill(strength=0.8, temporality=1.0)
    text = r.summary()
    assert "Bradford-Hill" in text
    assert "Verdict" in text
