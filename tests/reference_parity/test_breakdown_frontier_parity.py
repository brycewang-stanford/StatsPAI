"""Analytical parity: sp.breakdown_frontier closed-form identities.

The additive-violation breakdown frontier (Masten & Poirier 2020 framework;
see paper.bib masten2020inference) reports exact closed forms of
(estimate, se, max_violation, alpha):

    breakdown_point    == estimate            (violation flipping sign at 0)
    breakdown_point_ci == estimate - z_{1-alpha/2} * se
    lower / upper      == estimate -/+ max_violation
    robust_at_max_violation == (lower > 0) for a positive conclusion

All hold to machine precision.
"""

from __future__ import annotations

import warnings

import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

EST, SE, ALPHA = 2.0, 0.5, 0.05


@pytest.fixture(scope="module")
def result():
    return sp.breakdown_frontier(estimate=EST, se=SE)


def test_breakdown_point_identities(result):
    mi = result.model_info
    assert mi["breakdown_point"] == pytest.approx(EST, abs=1e-15)
    z = stats.norm.ppf(1 - ALPHA / 2)
    assert mi["breakdown_point_ci"] == pytest.approx(EST - z * SE, abs=1e-12)


def test_bounds_are_estimate_plus_minus_violation(result):
    mi = result.model_info
    v = float(mi["max_violation"])
    assert result.lower == pytest.approx(EST - v, abs=1e-15)
    assert result.upper == pytest.approx(EST + v, abs=1e-15)
    assert bool(mi["robust_at_max_violation"]) == (result.lower > 0)
