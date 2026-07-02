"""Analytical parity: sp.tF_critical_value monotone adjustment (LMMP 2022).

The Lee, McCrary, Moreira & Porter (2022) tF procedure inflates the t critical
value when the first-stage F is small. Structural properties that hold exactly:

    tF(F) is non-increasing in F,
    tF(F) -> 1.96 as F grows large (5% two-sided),
    tF(F) >= 1.96 for all F.

Analytical evidence tier (monotone lookup behaviour; the exact table values are
the package's own LMMP implementation, not asserted against an external copy).
"""

from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def test_monotone_non_increasing_in_f():
    grid = [10, 15, 20, 30, 50, 100, 500, 5000]
    vals = [float(sp.tF_critical_value(f)) for f in grid]
    for a, b in zip(vals, vals[1:]):
        assert a >= b - 1e-9


def test_converges_to_z_and_bounded_below():
    assert float(sp.tF_critical_value(1e6)) == pytest.approx(1.96, abs=1e-2)
    for f in (10, 20, 50, 100, 1000):
        assert float(sp.tF_critical_value(f)) >= 1.96 - 1e-9


def test_weak_first_stage_inflates_critical_value():
    # A weak first stage (F=10) carries a materially larger critical value
    # than the strong-instrument limit.
    assert float(sp.tF_critical_value(10)) > 1.96 + 0.5
