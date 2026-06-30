"""Regression guard for FINDINGS F-002 (recommend_hit_rate benchmark).

Two invariants on a selection-on-observables design:

A. ``recommend`` must lead with a confounding-adjusting estimator (PSM / DML),
   NOT bare OLS — naive OLS is biased under confounding (Dehejia-Wahba 1999),
   and leading with it produces a plausible-but-wrong headline.
B. ``sp.audit(result, treatment=...)`` on a causal-adjustment regression must
   add the overlap / balance / OVB-sensitivity checks a referee demands, while
   a descriptive regression (no treatment declared) is never flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture()
def soo_data():
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # treatment selected on covariates (confounding)
    p = 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2)))
    treat = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + 2.0 * treat + 1.5 * x1 - 1.0 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})


def test_observational_leads_with_adjusting_estimator(soo_data):
    """A: PSM/DML leads; bare OLS is demoted to a labelled baseline."""
    rec = sp.recommend(soo_data, y="y", treatment="treat", covariates=["x1", "x2"])
    assert rec.design == "observational"
    top1 = rec.recommendations[0]["method"].lower()
    assert any(k in top1 for k in ("propensity", "matching", "double ml", "dml")), top1
    assert "ols" not in top1
    # OLS must still be present, but clearly labelled as a naive baseline.
    labels = " | ".join(r["method"].lower() for r in rec.recommendations)
    assert "ols" in labels and "baseline" in labels


def test_audit_treatment_aware_adds_observational_checks(soo_data):
    """B: declaring a treatment surfaces overlap/balance/OVB; default does not."""
    r = sp.regress("y ~ treat + x1 + x2", data=soo_data, robust="hc1")
    base = {c["name"] for c in sp.audit(r)["checks"]}
    twa = {c["name"] for c in sp.audit(r, treatment="treat")["checks"]}
    added = twa - base
    assert {"overlap", "balance_after", "ovb_sensitivity"} <= added
    # Descriptive regression (no treatment) must NOT carry these.
    assert "overlap" not in base and "balance_after" not in base


def test_audit_treatment_arg_is_backward_compatible(soo_data):
    """audit(result) without treatment behaves exactly as before."""
    r = sp.regress("y ~ treat + x1 + x2", data=soo_data, robust="hc1")
    card = sp.audit(r)
    assert card["method_family"] == "regression"
    assert card["summary"]["n_total"] == len(card["checks"])
