"""
Tests for sp.sharp_ope_unobserved + sp.causal_policy_forest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def test_sharp_ope_bounds_widen_with_gamma():
    rng = np.random.default_rng(131)
    n = 500
    actions = rng.integers(0, 2, size=n)
    rewards = rng.normal(0, 1, size=n)
    logging = np.where(actions == 1, 0.6, 0.4)  # behaviour policy prob of chosen action
    target = np.where(actions == 1, 0.7, 0.3)
    df = pd.DataFrame({"a": actions, "r": rewards, "e": logging, "pi": target})
    r1 = sp.sharp_ope_unobserved(
        df,
        actions="a",
        rewards="r",
        logging_prob="e",
        target_prob="pi",
        gamma=1.0,
    )
    r2 = sp.sharp_ope_unobserved(
        df,
        actions="a",
        rewards="r",
        logging_prob="e",
        target_prob="pi",
        gamma=2.0,
    )
    width1 = r1.upper_bound - r1.lower_bound
    width2 = r2.upper_bound - r2.lower_bound
    reward_arr = df["r"].to_numpy(float)
    logging_arr = df["e"].to_numpy(float)
    target_arr = df["pi"].to_numpy(float)
    ips = np.mean(target_arr * reward_arr / logging_arr)
    gamma = 2.0
    lower_ratio = target_arr / (gamma * logging_arr)
    upper_ratio = gamma * target_arr / logging_arr
    expected_lower = np.mean(
        np.where(reward_arr >= 0.0, lower_ratio, upper_ratio) * reward_arr
    )
    expected_upper = np.mean(
        np.where(reward_arr >= 0.0, upper_ratio, lower_ratio) * reward_arr
    )
    # Gamma = 1 collapses bounds to point.
    assert width1 < 1e-8
    np.testing.assert_allclose(r1.point_estimate, ips)
    np.testing.assert_allclose(r1.lower_bound, ips)
    np.testing.assert_allclose(r1.upper_bound, ips)
    np.testing.assert_allclose(r2.lower_bound, expected_lower)
    np.testing.assert_allclose(r2.upper_bound, expected_upper)
    assert width2 > width1
    assert "Sharp OPE" in r1.summary()


def test_sharp_ope_rejects_bad_gamma():
    df = pd.DataFrame(
        {
            "a": [0, 1],
            "r": [0.1, 0.2],
            "e": [0.5, 0.5],
            "pi": [0.3, 0.7],
        }
    )
    with pytest.raises(MethodIncompatibility, match="gamma must be"):
        sp.sharp_ope_unobserved(
            df,
            actions="a",
            rewards="r",
            logging_prob="e",
            target_prob="pi",
            gamma=0.5,
        )


def test_sharp_ope_rejects_bad_columns_and_probabilities_with_taxonomy():
    df = pd.DataFrame(
        {
            "a": [0, 1],
            "r": [0.1, 0.2],
            "e": [0.5, 1.2],
            "pi": [0.3, 0.7],
        }
    )
    with pytest.raises(MethodIncompatibility, match="Missing columns"):
        sp.sharp_ope_unobserved(
            df,
            actions="a",
            rewards="missing",
            logging_prob="e",
            target_prob="pi",
        )
    with pytest.raises(MethodIncompatibility, match="logging_prob"):
        sp.sharp_ope_unobserved(
            df,
            actions="a",
            rewards="r",
            logging_prob="e",
            target_prob="pi",
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
        (a == 1) & (x1 > 0),
        1.0 + 0.1 * rng.normal(size=n),
        np.where(
            (a == 0) & (x1 <= 0),
            1.0 + 0.1 * rng.normal(size=n),
            0.1 * rng.normal(size=n),
        ),
    )
    df = pd.DataFrame({"a": a, "r": r, "x1": x1, "x2": x2})
    res = sp.causal_policy_forest(
        df,
        actions="a",
        rewards="r",
        covariates=["x1", "x2"],
        n_trees=15,
        depth=3,
        random_state=137,
    )
    # Majority of units with x1>0 should be assigned action 1
    positive_x1 = df["x1"].to_numpy() > 0
    expected_policy = positive_x1.astype(int)
    np.testing.assert_allclose(res.assignments, expected_policy)
    action1_rate = (res.assignments[positive_x1] == 1).mean()
    assert action1_rate > 0.6, action1_rate
    action0_rate = (res.assignments[~positive_x1] == 0).mean()
    assert action0_rate > 0.6, action0_rate
    assert set(res.action_counts) == {0, 1}
    assert sum(res.action_counts.values()) == len(df)
    assert res.n_trees == 15
    assert res.depth == 3
    assert res.policy_value > 0.2, res.policy_value
    assert res.policy_value_se >= 0


def test_causal_policy_forest_rejects_contract_errors_with_taxonomy():
    df = pd.DataFrame(
        {
            "a": [0, 1] * 20,
            "r": np.linspace(0.0, 1.0, 40),
            "x": np.linspace(-1.0, 1.0, 40),
        }
    )
    with pytest.raises(MethodIncompatibility, match="Missing columns"):
        sp.causal_policy_forest(
            df,
            actions="a",
            rewards="r",
            covariates=["missing"],
        )
    with pytest.raises(MethodIncompatibility, match="n_trees"):
        sp.causal_policy_forest(
            df,
            actions="a",
            rewards="r",
            covariates="x",
            n_trees=0,
        )
    with pytest.raises(DataInsufficient, match="at least 30"):
        sp.causal_policy_forest(
            df.iloc[:10],
            actions="a",
            rewards="r",
            covariates="x",
        )


def test_ope_extensions_in_registry():
    fns = set(sp.list_functions())
    assert "sharp_ope_unobserved" in fns
    assert "causal_policy_forest" in fns
