"""Tier D P2 known-truth upgrades — epidemiology 2x2 / confusion-matrix measures.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). All three were graded ``weak`` by
``scripts/tierd_classify.py``. Each is a pure closed form from a 2x2 table, so
the assertions are exact:

    sp.attributable_risk  AF_exposed = (RR-1)/RR;  PAF = Pe(RR-1)/[1+Pe(RR-1)].
    sp.diagnostic_test    sensitivity/specificity/PPV/NPV/LR from TP,FP,FN,TN.
    sp.breslow_day_test    homogeneous odds ratios across strata -> not rejected.

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.attributable_risk — Levin (1953)
# ---------------------------------------------------------------------------
class TestAttributableRiskAnalytic:
    # a = exposed cases, b = exposed non-cases, c = unexposed cases,
    # d = unexposed non-cases.

    def test_af_exposed_and_rr_closed_form(self):
        a, b, c, d = 40, 60, 20, 80
        rr = (a / (a + b)) / (c / (c + d))  # = 2.0
        res = sp.attributable_risk(a, b, c, d)
        assert res.rr == pytest.approx(rr, abs=1e-12)
        assert res.ar_exposed == pytest.approx((rr - 1) / rr, abs=1e-12)

    def test_paf_closed_form(self):
        a, b, c, d = 40, 60, 20, 80
        rr = (a / (a + b)) / (c / (c + d))
        pe = (a + b) / (a + b + c + d)  # prevalence of exposure = 0.5
        paf = pe * (rr - 1) / (1 + pe * (rr - 1))
        res = sp.attributable_risk(a, b, c, d)
        assert res.prevalence_exposed == pytest.approx(pe, abs=1e-12)
        assert res.paf == pytest.approx(paf, abs=1e-12)

    def test_no_association_gives_zero_fractions(self):
        # Equal risk in exposed and unexposed -> RR = 1 -> AF = PAF = 0.
        res = sp.attributable_risk(30, 70, 30, 70)
        assert res.rr == pytest.approx(1.0, abs=1e-12)
        assert res.ar_exposed == pytest.approx(0.0, abs=1e-12)
        assert res.paf == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# sp.diagnostic_test — sensitivity / specificity / PPV / NPV
# ---------------------------------------------------------------------------
class TestDiagnosticTestAnalytic:

    def test_confusion_matrix_arithmetic(self):
        tp, fp, fn, tn = 80, 20, 10, 90
        res = sp.diagnostic_test(tp=tp, fp=fp, fn=fn, tn=tn)
        assert res.sensitivity == pytest.approx(tp / (tp + fn), abs=1e-12)
        assert res.specificity == pytest.approx(tn / (tn + fp), abs=1e-12)
        assert res.ppv == pytest.approx(tp / (tp + fp), abs=1e-12)
        assert res.npv == pytest.approx(tn / (tn + fn), abs=1e-12)

    def test_likelihood_ratios(self):
        tp, fp, fn, tn = 80, 20, 10, 90
        res = sp.diagnostic_test(tp=tp, fp=fp, fn=fn, tn=tn)
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        assert res.lr_pos == pytest.approx(sens / (1 - spec), abs=1e-9)
        assert res.lr_neg == pytest.approx((1 - sens) / spec, abs=1e-9)

    def test_perfect_classifier(self):
        res = sp.diagnostic_test(tp=100, fp=0, fn=0, tn=100)
        assert res.sensitivity == pytest.approx(1.0)
        assert res.specificity == pytest.approx(1.0)
        assert res.ppv == pytest.approx(1.0)
        assert res.npv == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# sp.breslow_day_test — homogeneity of the odds ratio across strata
# ---------------------------------------------------------------------------
class TestBreslowDayAnalytic:

    def test_homogeneous_odds_ratios_not_rejected(self):
        # Both strata share OR = 4 -> the statistic is ~0 and not significant.
        tables = np.array([[[20, 10], [10, 20]], [[40, 20], [20, 40]]])
        stat, pval = sp.breslow_day_test(tables)
        assert stat == pytest.approx(0.0, abs=1e-6)
        assert pval > 0.99

    def test_heterogeneous_odds_ratios_rejected(self):
        # Stratum 1 OR = 4, stratum 2 OR = 1/16 -> strong heterogeneity.
        tables = np.array([[[20, 10], [10, 20]], [[10, 40], [40, 10]]])
        stat, pval = sp.breslow_day_test(tables)
        assert stat > 10
        assert pval < 0.01
