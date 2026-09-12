"""Tests for ``sp.yun_nonlinear`` — the Bauer-Sinning / Yun-weights nonlinear
Oaxaca-Blinder decomposition (previously untested).

The decomposition is an identity, so the strongest correctness checks are the
accounting laws it must satisfy exactly: the total gap equals the difference
in group rates, and explained + unexplained = gap, with the per-variable
detailed contributions summing back to the explained component.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def binary_two_group():
    rng = np.random.RandomState(0)
    n = 600
    g = rng.binomial(1, 0.5, n)
    x1 = rng.randn(n) + 0.4 * g
    x2 = rng.randn(n)
    y = (0.6 * x1 + 0.2 * x2 + 0.3 * g + rng.randn(n) > 0).astype(int)
    return pd.DataFrame({"y": y, "grp": g, "x1": x1, "x2": x2})


def test_gap_equals_rate_difference(binary_two_group):
    r = sp.yun_nonlinear(binary_two_group, "y", "grp", ["x1", "x2"], model="logit")
    assert r.gap == pytest.approx(r.rate_a - r.rate_b, abs=1e-9)


def test_explained_plus_unexplained_equals_gap(binary_two_group):
    r = sp.yun_nonlinear(binary_two_group, "y", "grp", ["x1", "x2"], model="logit")
    assert r.explained + r.unexplained == pytest.approx(r.gap, abs=1e-8)


def test_detailed_contributions_sum_to_explained(binary_two_group):
    r = sp.yun_nonlinear(binary_two_group, "y", "grp", ["x1", "x2"], model="logit")
    detailed = r.detailed
    total = float(detailed["contribution"].sum())
    assert total == pytest.approx(r.explained, abs=1e-8)
    # One row per explanatory variable.
    assert set(detailed["variable"]) == {"x1", "x2"}


def test_group_counts_partition_the_sample(binary_two_group):
    r = sp.yun_nonlinear(binary_two_group, "y", "grp", ["x1"], model="logit")
    assert r.n_a + r.n_b == len(binary_two_group)
    # Delta-method SEs (mvdcmp) for the explained, unexplained and total gap.
    assert set(r.se) == {"explained", "unexplained", "gap"}
    assert all(v >= 0.0 for v in r.se.values())


def test_probit_link_also_satisfies_identity(binary_two_group):
    # Switching the link must not break the decomposition accounting.
    r = sp.yun_nonlinear(binary_two_group, "y", "grp", ["x1", "x2"], model="probit")
    assert r.explained + r.unexplained == pytest.approx(r.gap, abs=1e-8)


def test_missing_outcome_column_raises(binary_two_group):
    with pytest.raises((ValueError, KeyError)):
        sp.yun_nonlinear(binary_two_group, "nope", "grp", ["x1"], model="logit")
