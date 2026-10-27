"""Option scan, round 5 (2026-10-27): 2x2 measures and ``sp.ndcg``.

Ground truth
------------
* ``odds_ratio(method="exact")``: the exact conditional interval and the
  conditional MLE recomputed here at 40 significant digits (mpmath bisection
  on the noncentral hypergeometric); Stata 18 ``cci, exact`` and R 4.x
  ``fisher.test`` pinned as run on 2026-10-27.
* PAF: Stata 18 ``punaf`` (Newson, SSC) after ``glm y x, family(binomial)
  link(log)`` with tolerances 1e-15; Stata ``csi`` for the point estimate on
  tables with a zero cell.
* NNT: R ``epiR::epi.2by2`` 2.0.94 ``NNT.strata.wald`` (signed reciprocal
  of the Wald risk-difference limits) and Altman (1998) for the reading of
  an interval that crosses zero.
* ``ndcg`` ties: the expectation of DCG over every ordering of the tied
  cases, which is what ``sklearn.metrics.ndcg_score`` computes by default.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.epi import measures as epi_measures
from statspai.exceptions import DataInsufficient, MethodIncompatibility

INF = float("inf")


# --------------------------------------------------------------------------
# 1. odds_ratio(method="exact")
# --------------------------------------------------------------------------

# table: (conditional MLE, 95% CI, 90% CI), 40-digit mpmath values
EXACT_TRUTH = {
    (50, 20, 30, 40): (
        3.3031695697504837,
        (1.5630296183008857, 7.1631009551872265),
        (1.7421732563944169, 6.3718831133606161),
    ),
    (5, 10, 2, 20): (
        4.7712398743910976,
        (0.64226816094342916, 58.534086238667621),
        (0.82979879061668631, 39.25673548026576),
    ),
    (12, 3, 4, 9): (
        8.1821869349566669,
        (1.2555386506047953, 73.425377936807687),
        (1.5885402540541555, 53.120503446300157),
    ),
}

# Stata 18 `cci a c b d, exact`: r(or), r(lb_or), r(ub_or)
STATA_CCI = {
    (50, 20, 30, 40): (3.333333333333333, 1.56274365488722, 7.164661042669297),
    (5, 10, 2, 20): (5.0, 0.6422685102083194, 58.53200293357927),
}
# R 4.x fisher.test(matrix(c(a, b, c, d), 2, byrow = TRUE))
R_FISHER = {
    (50, 20, 30, 40): (3.30314842212977, 1.56303272537173, 7.16308677629686),
    (5, 10, 2, 20): (4.77125918188935, 0.642268242878259, 58.4633608135252),
    (12, 3, 4, 9): (8.18214773717014, 1.25553807705573, 73.4236088529819),
}


@pytest.mark.parametrize("table", sorted(STATA_CCI))
def test_or_exact_point_is_the_sample_or_as_stata_cci(table):
    # The docstring makes `method` a confidence-interval choice, so the
    # point estimate is the sample OR under both methods, as Stata cc / cci.
    res = sp.epi.odds_ratio(*table, method="exact")
    a, b, c, d = table
    assert res.estimate == pytest.approx(a * d / (b * c), rel=1e-15)
    assert res.estimate == pytest.approx(STATA_CCI[table][0], rel=1e-15)


@pytest.mark.parametrize("table", sorted(EXACT_TRUTH))
def test_or_exact_reports_the_conditional_mle(table):
    res = sp.epi.odds_ratio(*table, method="exact")
    truth = EXACT_TRUTH[table][0]
    assert res.conditional_mle == pytest.approx(truth, rel=1e-10)
    # R fisher.test reports this estimate, solved with uniroot's default
    # tolerance; its value sits within 1e-5 of the 40-digit root.
    assert res.conditional_mle == pytest.approx(R_FISHER[table][0], rel=1e-5)


def test_or_conditional_mle_at_the_boundary_and_woolf():
    assert sp.epi.odds_ratio(0, 10, 5, 20, method="exact").conditional_mle == 0.0
    assert sp.epi.odds_ratio(10, 0, 3, 7, method="exact").conditional_mle == INF
    # Woolf inference has no conditional estimate.
    assert sp.epi.odds_ratio(50, 20, 30, 40).conditional_mle is None


@pytest.mark.parametrize("table", sorted(EXACT_TRUTH))
def test_or_exact_interval_is_the_exact_conditional_interval(table):
    _, ci95, ci90 = EXACT_TRUTH[table]
    r95 = sp.epi.odds_ratio(*table, method="exact")
    r90 = sp.epi.odds_ratio(*table, method="exact", alpha=0.10)
    np.testing.assert_allclose(r95.ci, ci95, rtol=1e-10)
    np.testing.assert_allclose(r90.ci, ci90, rtol=1e-10)


@pytest.mark.parametrize("table", sorted(R_FISHER))
def test_or_exact_interval_vs_r_and_stata_root_finders(table):
    # Both references solve the same two tail equations with a loose
    # root finder: R fisher.test (uniroot default tolerance) is off the
    # 40-digit root by up to 1.2e-3 relative, Stata cci by up to 2.2e-4.
    # StatsPAI (scipy) is on the root to 1e-10 (test above).
    ours = sp.epi.odds_ratio(*table, method="exact").ci
    np.testing.assert_allclose(ours, R_FISHER[table][1:], rtol=2e-3)
    if table in STATA_CCI:
        np.testing.assert_allclose(ours, STATA_CCI[table][1:], rtol=3e-4)


def test_or_exact_summary_names_both_estimates():
    text = sp.epi.odds_ratio(50, 20, 30, 40, method="exact").summary()
    assert "conditional MLE" in text


# --------------------------------------------------------------------------
# 2a. attributable_risk: PAF and its interval
# --------------------------------------------------------------------------

# Stata 18 punaf, atspec(x=0) after glm y x, family(binomial) link(log):
# (paf, lower, upper). The log-binomial fit converges to ~3e-9 relative.
# punaf's default (conditional) variance holds the exposure prevalence
# fixed: prevalence="fixed" (the default "estimated" is pinned to
# punaf, vce(unconditional) in tests/test_r6_open_items.py).
STATA_PUNAF = {
    (30, 70, 20, 80): (0.2, -0.07552027381101922, 0.4049391577415717),
    (50, 20, 30, 40): (0.25, 0.09055489995302313, 0.3814909773322829),
    (40, 60, 20, 80): (1 / 3, 0.09428468344959284, 0.5092890267802856),
}


@pytest.mark.parametrize("table", sorted(STATA_PUNAF))
def test_paf_and_interval_match_stata_punaf(table):
    res = sp.epi.attributable_risk(*table, prevalence="fixed")
    paf, lo, hi = STATA_PUNAF[table]
    assert res.paf == pytest.approx(paf, rel=1e-12)
    np.testing.assert_allclose(res.paf_ci, (lo, hi), rtol=1e-7)


def test_paf_point_with_a_zero_cell_is_the_plug_in_as_stata_csi():
    # Stata `csi 10 5 0 5`: RR 2, AF exposed .5, AF population .3333333.
    res = sp.epi.attributable_risk(10, 0, 5, 5, prevalence="fixed")
    assert res.rr == pytest.approx(2.0, rel=1e-15)
    assert res.ar_exposed == pytest.approx(0.5, rel=1e-15)
    assert res.paf == pytest.approx(1 / 3, rel=1e-15)
    # Katz delta method on the raw counts (finite: a > 0 and c > 0).
    pe, rr = 0.5, 2.0
    se_log_rr = math.sqrt(1 / 10 - 1 / 10 + 1 / 5 - 1 / 10)
    se_lnq = pe * rr / (1 + pe * (rr - 1)) * se_log_rr
    z = 1.959963984540054
    q = 1 - 1 / 3
    np.testing.assert_allclose(
        res.paf_ci,
        (1 - q * math.exp(z * se_lnq), 1 - q * math.exp(-z * se_lnq)),
        rtol=1e-12,
    )
    assert res.paf_ci[0] <= res.paf <= res.paf_ci[1]


def test_paf_without_unexposed_cases_is_one_and_has_no_delta_interval():
    # Stata `csi 5 0 10 20`: AF population 1, RR and its CI missing.
    with pytest.warns(UserWarning, match="no interval"):
        res = sp.epi.attributable_risk(5, 10, 0, 20)
    assert res.paf == 1.0
    assert res.rr == INF
    assert res.ar_exposed == 1.0
    assert all(math.isnan(v) for v in res.paf_ci)


def test_paf_without_exposed_cases_has_no_delta_interval():
    with pytest.warns(UserWarning, match="no interval"):
        res = sp.epi.attributable_risk(0, 10, 5, 20)
    # Plug-in: R = 5/35, R0 = 5/25, PAF = 1 - R0/R.
    assert res.paf == pytest.approx(1 - (5 / 25) / (5 / 35), rel=1e-14)
    assert res.rr == 0.0
    assert all(math.isnan(v) for v in res.paf_ci)


def test_paf_without_any_case_is_refused():
    with pytest.raises(DataInsufficient):
        sp.epi.attributable_risk(0, 10, 0, 20)


# --------------------------------------------------------------------------
# 2b. number_needed_to_treat
# --------------------------------------------------------------------------

# epiR::epi.2by2(method = "cohort.count") NNT.strata.wald: (est, lower, upper)
EPIR_NNT = {
    (30, 70, 20, 80): (10.0, -52.0292585148676, 4.56162852803912),
    (20, 80, 40, 60): (-5.0, -13.1508013435669, -3.08681030641652),
    (30, 70, 50, 50): (-5.0, -14.9100680567131, -3.00362334609592),
}


@pytest.mark.parametrize("table", [(20, 80, 40, 60), (30, 70, 50, 50)])
def test_nnt_interval_matches_epir_when_rd_excludes_zero(table):
    est, lo, hi = EPIR_NNT[table]
    res = sp.epi.number_needed_to_treat(*table)
    assert res.estimate == pytest.approx(abs(est), rel=1e-12)
    np.testing.assert_allclose(res.ci, (abs(hi), abs(lo)), rtol=1e-12)
    np.testing.assert_allclose(res.ci_signed, (lo, hi), rtol=1e-12)
    assert res.ci_disjoint is False


def test_nnt_interval_crossing_zero_is_altman_disjoint():
    table = (30, 70, 20, 80)  # RD 0.1, Wald CI (-0.0192, 0.2192)
    est, lo, hi = EPIR_NNT[table]
    res = sp.epi.number_needed_to_treat(*table)
    assert res.estimate == pytest.approx(est, rel=1e-12)
    assert res.ci_disjoint is True
    # signed reciprocals of the RD limits, as epiR reports them
    np.testing.assert_allclose(res.ci_signed, (lo, hi), rtol=1e-12)
    # |NNT| compatible with the data: from 1/max(|L|, U) to infinity
    assert res.ci[0] == pytest.approx(hi, rel=1e-12)
    assert res.ci[1] == INF
    assert res.ci[0] <= res.estimate
    text = res.summary()
    assert "NNTH 4.56 to infinity to NNTB 52.03" in text


def test_nnt_rd_limit_at_exactly_zero_does_not_divide_by_zero(monkeypatch):
    real = epi_measures.risk_difference

    def _rd(*args, **kwargs):
        r = real(*args, **kwargs)
        r.ci = (0.0, 0.25)
        return r

    monkeypatch.setattr(epi_measures, "risk_difference", _rd)
    res = sp.epi.number_needed_to_treat(30, 70, 20, 80)
    assert res.ci == (4.0, INF)
    assert res.ci_signed == (4.0, INF)
    assert res.ci_disjoint is False


def test_nnt_zero_risk_difference():
    res = sp.epi.number_needed_to_treat(30, 70, 30, 70)
    assert res.estimate == INF
    rd = sp.epi.risk_difference(30, 70, 30, 70)
    assert res.ci_disjoint is True
    assert res.ci[0] == pytest.approx(1 / rd.ci[1], rel=1e-12)
    assert res.ci[1] == INF


def test_nnt_alpha_changes_the_interval():
    r95 = sp.epi.number_needed_to_treat(20, 80, 40, 60)
    r90 = sp.epi.number_needed_to_treat(20, 80, 40, 60, alpha=0.10)
    rd90 = sp.epi.risk_difference(20, 80, 40, 60, alpha=0.10)
    np.testing.assert_allclose(
        r90.ci, (1 / abs(rd90.ci[0]), 1 / abs(rd90.ci[1])), rtol=1e-12
    )
    assert r90.ci[1] < r95.ci[1]


# --------------------------------------------------------------------------
# 4. ndcg with tied scores
# --------------------------------------------------------------------------


def test_ndcg_ties_are_invariant_to_row_order():
    y = np.array([1, 0, 0, 1, 0, 0])
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1, 0.0])
    # rank 1 is a hit; ranks 2-4 hold one hit among three tied cases, so
    # rank 2 earns 1/3 on average over the orderings of the tie.
    d = 1 / np.log2(3)
    truth = (1 + d / 3) / (1 + d)
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.permutation(len(y))
        assert sp.ndcg(y[p], s[p], k=2) == pytest.approx(truth, rel=1e-14)


def test_ndcg_ties_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=400) < 0.1).astype(int)
    s = np.round(rng.normal(size=400) + y, 1)  # many ties
    for k in (5, 20, 0.05, 0.2):
        kn = int(round(400 * k)) if k < 1 else k
        ref = sk.ndcg_score(y[None, :], s[None, :], k=kn)
        assert sp.ndcg(y, s, k=k) == pytest.approx(ref, rel=1e-12)


def test_ndcg_without_ties_is_the_ranked_sum():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    dcg = 1 + 1 / np.log2(4)
    ideal = 1 + 1 / np.log2(3)
    assert sp.ndcg(y, s, k=3) == pytest.approx(dcg / ideal, rel=1e-15)


def test_ndcg_non_integer_count_is_refused():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    with pytest.raises(MethodIncompatibility, match="whole number"):
        sp.ndcg(y, s, k=2.5)
    assert sp.ndcg(y, s, k=2.0) == sp.ndcg(y, s, k=2)


def test_measures_emit_no_runtime_warnings_on_regular_tables():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.epi.attributable_risk(30, 70, 20, 80)
        sp.epi.number_needed_to_treat(30, 70, 20, 80)
        sp.epi.odds_ratio(50, 20, 30, 40, method="exact")


# --------------------------------------------------------------------------
# 5. bradford_hill: None in the evidence dict means "not assessed"
# --------------------------------------------------------------------------


def test_bradford_hill_none_in_evidence_dict_is_not_assessed():
    a = sp.epi.bradford_hill({"temporality": 1.0, "strength": 1.0, "analogy": None})
    b = sp.epi.bradford_hill(temporality=1.0, strength=1.0)
    assert (a.total, a.max_total) == (b.total, b.max_total) == (2.0, 2.0)
    assert math.isnan(a.scores["analogy"])
    with pytest.raises(MethodIncompatibility, match="Unknown viewpoint"):
        sp.epi.bradford_hill({"strenght": 1.0})
    with pytest.raises(MethodIncompatibility, match=r"\[0, 1\]"):
        sp.epi.bradford_hill(strength=2.0)
