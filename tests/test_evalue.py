"""Focused tests for sp.evalue (VanderWeele & Ding 2017).

Includes a regression test for the confidence-interval null-crossing
guard: when a CI already contains the null (RR = 1), the E-value for the
CI must be exactly 1.0 — not the (spurious) E-value of the confidence
limit on the far side of the null. This bug was surfaced by the
NHEFS / *What If* reproduction (see docs/joss_validation_dossier.md).
"""
from __future__ import annotations

import numpy as np
import pytest

import statspai as sp


def test_point_estimate_matches_closed_form():
    rr = 1.3251
    ev = sp.evalue(estimate=rr, measure="RR")
    closed = rr + np.sqrt(rr * (rr - 1.0))
    assert ev["evalue_estimate"] == pytest.approx(closed, abs=1e-9)
    assert ev["evalue_estimate"] == pytest.approx(1.9814, abs=1e-3)


def test_ci_clearing_null_uses_nearest_limit():
    # RR > 1, lower limit 1.10 > 1: E-value computed from the lower limit.
    ev = sp.evalue(estimate=1.33, ci=(1.10, 1.60), measure="RR")
    assert ev["evalue_ci"] == pytest.approx(1.4317, abs=1e-3)


def test_protective_ci_clearing_null_uses_upper_limit():
    # RR < 1, upper limit 0.88 < 1: protective effect, E-value > 1.
    ev = sp.evalue(estimate=0.70, ci=(0.55, 0.88), measure="RR")
    assert ev["evalue_ci"] > 1.0
    assert ev["evalue_ci"] == pytest.approx(1.53, abs=1e-2)


@pytest.mark.parametrize("estimate,ci", [
    (0.90, (0.79, 1.22)),   # RR < 1, CI crosses the null
    (1.10, (0.95, 1.27)),   # RR > 1, CI crosses the null
    (1.00, (0.80, 1.25)),   # exactly null
])
def test_ci_crossing_null_returns_one(estimate, ci):
    """Regression: a CI that already contains RR=1 has E-value 1.0."""
    ev = sp.evalue(estimate=estimate, ci=ci, measure="RR")
    assert ev["evalue_ci"] == pytest.approx(1.0, abs=1e-12)


def test_borderline_limit_at_null_is_one():
    # Lower limit exactly at the null -> E-value 1.0 (matches R EValue).
    ev = sp.evalue(estimate=1.30, ci=(1.00, 1.60), measure="RR")
    assert ev["evalue_ci"] == pytest.approx(1.0, abs=1e-12)
