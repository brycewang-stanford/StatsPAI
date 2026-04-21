"""Tests for clinical-diagnostic primitives (sensitivity/specificity,
ROC, Cohen's kappa)."""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Sensitivity / specificity
# ---------------------------------------------------------------------------


def test_sensitivity_specificity_from_counts():
    r = sp.sensitivity_specificity(tp=90, fn=10, fp=5, tn=95)
    assert r.sensitivity == pytest.approx(0.9)
    assert r.specificity == pytest.approx(0.95)
    assert r.ppv == pytest.approx(90 / 95)
    assert r.npv == pytest.approx(95 / 105)
    assert r.lr_pos == pytest.approx(0.9 / 0.05)
    assert r.lr_neg == pytest.approx(0.1 / 0.95)


def test_sensitivity_specificity_from_vectors():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0])
    r = sp.sensitivity_specificity(y_true, y_pred)
    assert r.tp == 3 and r.fn == 1
    assert r.fp == 1 and r.tn == 3
    assert r.sensitivity == pytest.approx(0.75)
    assert r.specificity == pytest.approx(0.75)


def test_sensitivity_specificity_requires_either_vectors_or_counts():
    with pytest.raises(ValueError):
        sp.sensitivity_specificity()  # no args


def test_diagnostic_test_is_alias():
    r1 = sp.sensitivity_specificity(tp=90, fn=10, fp=5, tn=95)
    r2 = sp.diagnostic_test(tp=90, fn=10, fp=5, tn=95)
    assert r1.sensitivity == r2.sensitivity
    assert r1.specificity == r2.specificity


def test_sensitivity_specificity_ci_brackets_point():
    r = sp.sensitivity_specificity(tp=90, fn=10, fp=5, tn=95)
    assert r.sensitivity_ci[0] <= r.sensitivity <= r.sensitivity_ci[1]
    assert r.specificity_ci[0] <= r.specificity <= r.specificity_ci[1]


def test_sensitivity_specificity_summary_runs():
    r = sp.sensitivity_specificity(tp=90, fn=10, fp=5, tn=95)
    text = r.summary()
    assert "Sensitivity" in text
    assert "Specificity" in text


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------


def test_roc_curve_perfect_separator():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    r = sp.roc_curve(y, scores)
    assert r.auc == pytest.approx(1.0)
    assert 0.0 <= r.auc_ci[0] <= 1.0
    assert 0.0 <= r.auc_ci[1] <= 1.0


def test_roc_curve_random_scores():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.binomial(1, 0.5, n)
    scores = rng.normal(0, 1, n)
    r = sp.roc_curve(y, scores)
    # Random scores -> AUC ≈ 0.5
    assert abs(r.auc - 0.5) < 0.1


def test_roc_curve_rejects_single_class():
    y = np.ones(10)
    scores = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        sp.roc_curve(y, scores)


def test_auc_shortcut_matches():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert sp.auc(y, s) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------


def test_cohen_kappa_perfect_agreement():
    a = np.array([1, 2, 3, 1, 2, 3])
    b = np.array([1, 2, 3, 1, 2, 3])
    r = sp.cohen_kappa(a, b)
    assert r.kappa == pytest.approx(1.0)
    assert r.n_categories == 3


def test_cohen_kappa_zero_agreement_chance():
    # When P_o == P_e, kappa == 0
    a = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    b = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    r = sp.cohen_kappa(a, b)
    # Not necessarily exactly 0 but should be near 0
    assert abs(r.kappa) < 0.3


def test_cohen_kappa_weighted():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    r_unw = sp.cohen_kappa(a, b, weights="unweighted")
    r_lin = sp.cohen_kappa(a, b, weights="linear")
    r_quad = sp.cohen_kappa(a, b, weights="quadratic")
    # All should show perfect agreement
    for r in (r_unw, r_lin, r_quad):
        assert r.kappa == pytest.approx(1.0, abs=1e-6)


def test_cohen_kappa_interpretation():
    a = [1, 2, 3, 1, 2, 3]
    b = [1, 2, 3, 1, 2, 3]
    r = sp.cohen_kappa(a, b)
    assert "almost perfect" in r.interpretation()


def test_cohen_kappa_rejects_invalid_weights():
    with pytest.raises(ValueError):
        sp.cohen_kappa([0, 1], [0, 1], weights="bogus")


def test_cohen_kappa_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        sp.cohen_kappa([0, 1, 2], [0, 1])
