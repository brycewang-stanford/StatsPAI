"""
Tests for sp.causal_bandit / sp.counterfactual_policy_optimization /
sp.structural_mdp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# -------------------------------------------------------------------------
# causal_bandit
# -------------------------------------------------------------------------


def test_causal_bandit_picks_best_arm():
    rng = np.random.default_rng(151)

    def reward_fn(arm, context):
        means = {"A": 0.1, "B": 0.5, "C": 0.3}
        return float(means[arm] + rng.normal(0, 0.2))

    res = sp.causal_bandit(
        arms=["A", "B", "C"], reward_fn=reward_fn,
        context={"u": 42}, n_samples=400, rng_seed=151,
    )
    assert res.arm_labels == ["A", "B", "C"]
    assert res.optimal_arm == 1  # "B"
    assert res.context == {"u": 42}
    assert "Causal" in res.summary()


def test_causal_bandit_requires_two_arms():
    with pytest.raises(ValueError, match=">= 2 arms"):
        sp.causal_bandit(arms=["A"], reward_fn=lambda a, c: 0.0)


# -------------------------------------------------------------------------
# counterfactual_policy_optimization
# -------------------------------------------------------------------------


def test_cfpo_detects_better_policy():
    """A policy that sets a = +2*s should dominate random logged actions."""
    rng = np.random.default_rng(157)
    n = 300
    s = rng.normal(0, 1, size=n)
    a = rng.normal(0, 1, size=n)
    r = 0.5 * s + 1.5 * a + rng.normal(0, 0.3, size=n)
    df = pd.DataFrame({"s": s, "a": a, "r": r})
    res = sp.counterfactual_policy_optimization(
        df, state="s", action="a", reward="r",
        target_policy=lambda si: 2.0 * si,  # high positive corr with a-coef
    )
    # The target policy should systematically improve expected reward.
    assert res.expected_value_target > res.expected_value_logged
    assert "Counterfactual" in res.summary()


def test_cfpo_missing_columns_errors():
    df = pd.DataFrame({"s": [0.1, 0.2], "a": [0.3, 0.4], "r": [0.5, 0.6]})
    with pytest.raises(ValueError, match="Missing"):
        sp.counterfactual_policy_optimization(
            df, state="bogus", action="a", reward="r",
            target_policy=lambda s: s,
        )


# -------------------------------------------------------------------------
# structural_mdp
# -------------------------------------------------------------------------


def test_structural_mdp_fits_linear_dynamics():
    """Simulate s_{t+1} = 0.9 s_t + 0.5 a_t + noise, r = s + a."""
    rng = np.random.default_rng(163)
    T = 400
    s = np.zeros(T)
    a = rng.normal(0, 1, size=T)
    r = np.zeros(T)
    for t in range(1, T):
        s[t] = 0.9 * s[t - 1] + 0.5 * a[t - 1] + rng.normal(0, 0.1)
    r = s + a + rng.normal(0, 0.1, size=T)
    df = pd.DataFrame({"s": s, "a": a, "r": r, "t": np.arange(T)})
    res = sp.structural_mdp(
        df, state_cols=["s"], action_cols=["a"], reward="r", time="t",
    )
    # A matrix should be close to [[0.9]]
    assert abs(res.A[0, 0] - 0.9) < 0.05, res.A
    assert abs(res.B[0, 0] - 0.5) < 0.1, res.B
    # Reward coefs should be near [1, 1]
    assert abs(res.reward_coef[0] - 1.0) < 0.2
    assert abs(res.reward_coef[1] - 1.0) < 0.2
    # Counterfactual rollout runs
    rollout = res.counterfactual_rollout(
        np.array([0.0]), policy=lambda s: np.array([0.5]), horizon=5,
    )
    assert rollout["states"].shape == (6, 1)
    assert rollout["actions"].shape == (5, 1)
    assert rollout["rewards"].shape == (5,)


def test_structural_mdp_requires_time_or_next_state():
    df = pd.DataFrame({"s": [0.1, 0.2], "a": [0.3, 0.4], "r": [0.5, 0.6]})
    with pytest.raises(ValueError, match="next_state_cols.*time"):
        sp.structural_mdp(df, state_cols=["s"], action_cols=["a"], reward="r")


def test_causal_rl_in_registry():
    fns = set(sp.list_functions())
    assert "causal_bandit" in fns
    assert "counterfactual_policy_optimization" in fns
    assert "structural_mdp" in fns
