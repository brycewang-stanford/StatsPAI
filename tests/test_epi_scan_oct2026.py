"""Option scan of ``epi/`` (2026-10-27).

Each test states the correct behaviour; the defects they found are fixed.
References: R vcd::Kappa and
psych::cohen.kappa (kappa ASE 0.08), mantelhaen.test (statistic 0, p = 1),
poisson.test (p 0.25 and 0.01348929), run as black boxes on 2026-10-27.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import statspai as sp

# --------------------------------------------------------------------------- #
#  Cohen's kappa standard error (Fleiss, Cohen & Everitt 1969 large-sample
#  variance). Table: 40 (0,0), 10 (0,1), 10 (1,0), 40 (1,1), n = 100.
#  p_o = .8, p_e = .5, kappa = .6.
#  A = sum_i p_ii [1 - (p_i. + p_.i)(1 - k)]^2           = .8 * .36 = .288
#  B = (1 - k)^2 sum_{i!=j} p_ij (p_.i + p_j.)^2          = .16 * .2  = .032
#  C = [k - p_e (1 - k)]^2                                 = .16
#  var = (A + B - C) / (n (1 - p_e)^2) = .16 / 25 = .0064, se = .08
# --------------------------------------------------------------------------- #


def _kappa_raters():
    a = np.r_[np.zeros(40), np.zeros(10), np.ones(10), np.ones(40)].astype(int)
    b = np.r_[np.zeros(40), np.ones(10), np.zeros(10), np.ones(40)].astype(int)
    return a, b


def test_kappa_point_estimate_baseline():
    a, b = _kappa_raters()
    r = sp.cohen_kappa(a, b)
    assert r.kappa == pytest.approx(0.6, abs=1e-12)


def test_kappa_unweighted_se_is_fleiss_cohen_everitt():
    a, b = _kappa_raters()
    r = sp.cohen_kappa(a, b)
    assert r.se == pytest.approx(0.08, rel=1e-10)


def test_kappa_unweighted_equals_linear_weighted_with_two_categories():
    # with two categories the linear agreement weights are the identity,
    # so the weighted kappa and its variance are the unweighted ones
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, 250)
    b = np.where(rng.random(250) < 0.7, a, 1 - a)
    u = sp.cohen_kappa(a, b)
    w = sp.cohen_kappa(a, b, weights="linear")
    assert u.kappa == pytest.approx(w.kappa, rel=1e-12)
    assert u.se == pytest.approx(w.se, rel=1e-10)


# --------------------------------------------------------------------------- #
#  Mantel-Haenszel chi-square: Yates term is min(0.5, |delta|)
#  (R mantelhaen.test(correct = TRUE)). Balanced strata: delta = 0, so the
#  statistic is 0 and the p-value 1.
# --------------------------------------------------------------------------- #


def test_mh_balanced_strata_p_value_is_one():
    tables = [[[10, 10], [10, 10]], [[15, 5], [15, 5]]]
    res = sp.mantel_haenszel(tables, measure="OR")
    assert res.estimate == pytest.approx(1.0, abs=1e-12)
    assert res.p_value == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
#  Incidence rate ratio p-value
# --------------------------------------------------------------------------- #


def test_irr_zero_unexposed_events_exact_p():
    # conditional on 3 events and equal person-time, D1 ~ Bin(3, 1/2) under
    # H0: two-sided exact p = P(D1 in {0, 3}) = 0.25 (R poisson.test)
    res = sp.epi.incidence_rate_ratio(3, 100.0, 0, 100.0, method="exact")
    assert res.p_value == pytest.approx(0.25, abs=1e-12)


def test_irr_zero_unexposed_events_wald_p_not_zero():
    res = sp.epi.incidence_rate_ratio(3, 100.0, 0, 100.0, method="wald")
    assert res.p_value > 0.05


def test_irr_exact_p_matches_conditional_binomial():
    e1, t1, e0, t0 = 40, 100.0, 20, 100.0
    res = sp.epi.incidence_rate_ratio(e1, t1, e0, t0, method="exact")
    ref = stats.binomtest(e1, e1 + e0, t1 / (t1 + t0)).pvalue
    assert res.p_value == pytest.approx(ref, rel=1e-10)


def test_irr_exact_p_matches_r_poisson_test():
    # R: poisson.test(c(40, 20), c(100, 100))$p.value
    res = sp.epi.incidence_rate_ratio(40, 100.0, 20, 100.0, method="exact")
    assert res.p_value == pytest.approx(0.01348929, abs=5e-9)


def test_irr_zero_exposed_events_wald_p_not_one():
    # 0 vs 3 events: the old code set log IRR to 0 and reported p = 1
    res = sp.epi.incidence_rate_ratio(0, 100.0, 3, 100.0, method="wald")
    assert res.p_value < 0.5
    assert res.ci[0] > 0 and np.isfinite(res.ci[1])


def test_irr_wald_unchanged_without_zero_counts():
    # identity: with both counts positive the Wald interval is log(IRR) +/- z se
    res = sp.epi.incidence_rate_ratio(40, 100.0, 20, 100.0, method="wald")
    se = np.sqrt(1 / 40 + 1 / 20)
    z = stats.norm.ppf(0.975)
    assert res.ci[0] == pytest.approx(2.0 * np.exp(-z * se), rel=1e-14)
    assert res.p_value == pytest.approx(2 * stats.norm.sf(np.log(2.0) / se), rel=1e-14)


def test_mh_continuity_unchanged_when_delta_large():
    # |delta| >= 0.5: the Yates term is the usual 0.5 (R mantelhaen.test)
    tables = [[[30, 70], [20, 80]], [[40, 60], [25, 75]]]
    res = sp.mantel_haenszel(tables, measure="OR")
    a = np.array([30, 40.0])
    n1 = np.array([100, 100.0])
    n0 = np.array([100, 100.0])
    m1 = np.array([50, 65.0])
    m0 = 200 - m1
    n = 200.0
    delta = abs(a.sum() - (n1 * m1 / n).sum())
    var = (n1 * n0 * m1 * m0 / (n**2 * (n - 1))).sum()
    chi2 = (delta - 0.5) ** 2 / var
    assert res.p_value == pytest.approx(stats.chi2.sf(chi2, 1), rel=1e-14)


# --------------------------------------------------------------------------- #
#  sp.evalue(measure ratio, se=): se is the SE of log(estimate)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("measure", ["RR", "HR", "OR"])
def test_evalue_ratio_se_is_on_the_log_scale(measure):
    est, se = 2.0, 0.15
    z = stats.norm.ppf(0.975)
    by_se = sp.evalue(estimate=est, se=se, measure=measure, rare=True)
    by_ci = sp.evalue(
        estimate=est,
        ci=(est * np.exp(-z * se), est * np.exp(z * se)),
        measure=measure,
        rare=True,
    )
    assert by_se["evalue_ci"] == pytest.approx(by_ci["evalue_ci"], rel=1e-14)
    lo = est * np.exp(-z * se)
    assert by_se["evalue_ci"] == pytest.approx(lo + np.sqrt(lo * (lo - 1)), rel=1e-14)
