"""
Tests for sp.sharp_ope_unobserved + sp.causal_policy_forest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def test_sharp_ope_bounds_widen_with_gamma():
    rng = np.random.default_rng(131)
    n = 500
    actions = rng.integers(0, 2, size=n)
    rewards = rng.normal(0, 1, size=n)
    logging = np.where(actions == 1, 0.6, 0.4)  # behaviour policy prob of chosen action
    target = np.where(actions == 1, 0.7, 0.3)
    df = pd.DataFrame({"a": actions, "r": rewards, "e": logging, "pi": target})
    r1 = sp.sharp_ope_unobserved(
        df, actions="a", rewards="r", logging_prob="e", target_prob="pi",
        gamma=1.0,
    )
    r2 = sp.sharp_ope_unobserved(
        df, actions="a", rewards="r", logging_prob="e", target_prob="pi",
        gamma=2.0,
    )
    width1 = r1.upper_bound - r1.lower_bound
    width2 = r2.upper_bound - r2.lower_bound
    # Gamma = 1 collapses bounds to point.
    assert width1 < 1e-8
    assert width2 > width1
    assert "Sharp OPE" in r1.summary()


def test_sharp_ope_rejects_bad_gamma():
    df = pd.DataFrame({
        "a": [0, 1], "r": [0.1, 0.2], "e": [0.5, 0.5], "pi": [0.3, 0.7],
    })
    with pytest.raises(ValueError, match="gamma must be"):
        sp.sharp_ope_unobserved(
            df, actions="a", rewards="r", logging_prob="e", target_prob="pi",
            gamma=0.5,
        )


def test_causal_policy_forest_prefers_correct_action():
    """DGP where action 1 is uniformly better than action 0 when x1 > 0."""
    rng = np.random.default_rng(137)
    n = 400
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    a = rng.integers(0, 2, size=n)
    # Reward: action 1 gives +1 when x1>0, action 0 gives +1 when x1<=0
    r = np.where(
        (a == 1) & (x1 > 0), 1.0 + 0.1 * rng.normal(size=n),
        np.where((a == 0) & (x1 <= 0), 1.0 + 0.1 * rng.normal(size=n),
                 0.1 * rng.normal(size=n))
    )
    df = pd.DataFrame({"a": a, "r": r, "x1": x1, "x2": x2})
    res = sp.causal_policy_forest(
        df, actions="a", rewards="r", covariates=["x1", "x2"],
        n_trees=15, depth=3, random_state=137,
    )
    # Majority of units with x1>0 should be assigned action 1
    positive_x1 = df["x1"].to_numpy() > 0
    action1_rate = (res.assignments[positive_x1] == 1).mean()
    assert action1_rate > 0.6, action1_rate
    action0_rate = (res.assignments[~positive_x1] == 0).mean()
    assert action0_rate > 0.6, action0_rate
    assert res.policy_value > 0.2, res.policy_value
    assert res.policy_value_se >= 0


def test_ope_extensions_in_registry():
    fns = set(sp.list_functions())
    assert "sharp_ope_unobserved" in fns
    assert "causal_policy_forest" in fns
