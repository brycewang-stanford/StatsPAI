"""Analytical parity: sp.breslow_day_test homogeneity decision.

Breslow-Day tests homogeneity of the odds ratio across strata. With per-stratum
ORs equal, the statistic is zero (identity) and p=1; with stratified ORs that
differ, the statistic is large and p→0. Analytical evidence tier
(known-truth behaviour on a deterministic stratified table).
"""

from __future__ import annotations

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def test_homogeneous_OR_statistic_is_zero():
    # Each stratum has the same 2x2 OR = 4.
    tables = np.array(
        [
            [[20, 10], [10, 20]],
            [[30, 15], [15, 30]],
            [[10, 5], [5, 10]],
        ],
        dtype=float,
    )
    chi2, p = sp.breslow_day_test(tables)
    assert chi2 == 0.0
    assert p == 1.0


def test_heterogeneous_OR_rejects_homogeneity():
    # Stratum 2 has a wildly different OR.
    tables = np.array(
        [
            [[20, 10], [10, 20]],  # OR = 4
            [[20, 30], [30, 5]],  # OR = 0.111
            [[10, 5], [5, 10]],  # OR = 4
        ],
        dtype=float,
    )
    chi2, p = sp.breslow_day_test(tables)
    assert chi2 > 10
    assert p < 0.001


def test_zero_two_by_two_degenerate_returns_zero():
    # Empty cell forces fallback; the closed-form variance guard returns 0.
    tables = np.array(
        [[[0, 50], [50, 0]]],
        dtype=float,
    )
    chi2, p = sp.breslow_day_test(tables)
    assert chi2 == 0.0
    assert 0.0 <= p <= 1.0
