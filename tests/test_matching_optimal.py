"""Optimal + cardinality matching tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.matching.optimal import optimal_match, cardinality_match


@pytest.fixture(scope="module")
def selection_dgp():
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    pz = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = (rng.uniform(size=n) < pz).astype(int)
    y = 1.0 * x1 + 0.5 * x2 + 2.0 * t + rng.standard_normal(n)
    return pd.DataFrame({"t": t, "y": y, "x1": x1, "x2": x2})


# ------------------------------------------------------------------ optimal
def test_optimal_match_recovers_att(selection_dgp):
    res = optimal_match(selection_dgp, "t", "y", ["x1", "x2"])
    assert abs(res.ate - 2.0) < 0.7     # within reason for matching


def test_optimal_match_pairs_every_treated(selection_dgp):
    res = optimal_match(selection_dgp, "t", "y", ["x1", "x2"])
    assert res.n_matched == res.n_treated


def test_optimal_match_caliper_filters(selection_dgp):
    res = optimal_match(selection_dgp, "t", "y", ["x1", "x2"], caliper=0.2)
    assert res.n_matched < res.n_treated


def test_optimal_match_requires_enough_controls():
    # more treated than controls
    df = pd.DataFrame({"t": [1, 1, 1, 0], "y": [1, 2, 3, 4],
                       "x1": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="n_control"):
        optimal_match(df, "t", "y", ["x1"])


# --------------------------------------------------------------- cardinality
def test_cardinality_match_recovers_att(selection_dgp):
    res = cardinality_match(selection_dgp, "t", "y", ["x1", "x2"],
                            smd_tolerance=0.1)
    assert abs(res.ate - 2.0) < 0.3


def test_cardinality_match_enforces_balance(selection_dgp):
    res = cardinality_match(selection_dgp, "t", "y", ["x1", "x2"],
                            smd_tolerance=0.1)
    assert (res.balance["|SMD|"] <= 0.1 + 1e-6).all()


def test_cardinality_match_tighter_tolerance_keeps_fewer(selection_dgp):
    r_loose = cardinality_match(selection_dgp, "t", "y", ["x1", "x2"],
                                smd_tolerance=0.5)
    r_tight = cardinality_match(selection_dgp, "t", "y", ["x1", "x2"],
                                smd_tolerance=0.05)
    assert r_tight.n_matched_pairs <= r_loose.n_matched_pairs


def test_exported_at_sp_dot():
    import statspai as sp
    assert callable(sp.optimal_match)
    assert callable(sp.cardinality_match)
