"""Analytical parity: sp.sharp_ope_unobserved + sp.causal_policy_forest.

Off-policy evaluation family, analytical evidence tier (no external
reference implementation; anchors are exact algebraic identities of the
marginal-sensitivity model plus a deterministic policy-learning DGP).

sharp_ope_unobserved (Kallus-Mao-Uehara-style MSM bounds): the sensitivity
model constrains the true importance weight w_i to
``[pi_i / (Gamma * e_i), Gamma * pi_i / e_i]``. Three exact identities:

1. With ``target_prob == logging_prob`` the importance weights are all 1,
   so the IPS point estimate equals the sample mean reward exactly; with
   ``Gamma = 1`` the weight interval degenerates and both sharp bounds
   collapse onto that point estimate (machine precision).
2. The interval ``[lower, upper]`` must always contain the IPS point
   estimate, and its width is monotonically increasing in Gamma (strictly,
   for non-degenerate rewards).
3. For non-negative rewards the adversarial weight choice is w_hi for the
   upper bound and w_lo for the lower bound on *every* unit, hence
   ``upper = Gamma * IPS`` and ``lower = IPS / Gamma`` exactly, even when
   ``target_prob != logging_prob``.

causal_policy_forest (DR policy tree ensemble): on a 3-action DGP where
the reward-maximising action is a step function of a single covariate
(action 0 for x0 < -0.5, action 1 for -0.5 <= x0 < 0.5, action 2
otherwise; reward = 1{a = opt(x0)} + N(0, 0.2) noise, uniform logging),
the learned policy must recover the true partition for >90% of units and
its estimated value must be near the known optimal value 1.0. Analytical
evidence tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# ---------------------------------------------------------------------------
# sharp_ope_unobserved
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ope_data():
    """Logged bandit data with target_prob == logging_prob (weights = 1)."""
    rng = np.random.default_rng(42)
    n = 400
    actions = rng.integers(0, 2, n)
    # Mixed-sign rewards so the adversarial weight flip is exercised and
    # the bound width is strictly increasing in gamma.
    rewards = rng.normal(0.4, 1.0, n)
    logging = np.where(actions == 1, 0.6, 0.4)
    return pd.DataFrame(
        {"a": actions, "r": rewards, "e": logging, "pi": logging.copy()}
    )


def test_sharp_ope_gamma_one_collapses_to_mean_reward(ope_data):
    # pi == e_hat -> importance weights are exactly 1, so IPS == mean(R);
    # gamma=1 -> the weight interval is a single point, so both bounds
    # equal the IPS estimate. Pure algebra: machine-precision tolerance.
    res = sp.sharp_ope_unobserved(
        ope_data,
        actions="a",
        rewards="r",
        logging_prob="e",
        target_prob="pi",
        gamma=1.0,
    )
    mean_r = float(ope_data["r"].mean())
    assert res.point_estimate == pytest.approx(mean_r, abs=1e-12)
    assert res.lower_bound == pytest.approx(mean_r, abs=1e-12)
    assert res.upper_bound == pytest.approx(mean_r, abs=1e-12)
    assert res.n == len(ope_data)


def test_sharp_ope_bounds_widen_monotonically_and_contain_point(ope_data):
    # The feasible weight set grows with gamma, so the sharp interval is
    # nested/monotone: width strictly increases (rewards are mixed-sign,
    # so both endpoints move) and always brackets the point estimate.
    widths = []
    for gamma in (1.0, 1.25, 1.5, 2.0, 3.0):
        res = sp.sharp_ope_unobserved(
            ope_data,
            actions="a",
            rewards="r",
            logging_prob="e",
            target_prob="pi",
            gamma=gamma,
        )
        assert res.lower_bound <= res.point_estimate <= res.upper_bound
        widths.append(res.upper_bound - res.lower_bound)
    assert widths[0] == pytest.approx(0.0, abs=1e-12)
    assert all(w2 > w1 for w1, w2 in zip(widths, widths[1:]))


def test_sharp_ope_nonnegative_reward_scaling_identity():
    # With R >= 0 the adversarial choice is the same on every unit:
    # upper = gamma * IPS and lower = IPS / gamma exactly, even when
    # target_prob != logging_prob. Exact identity -> tight rtol.
    rng = np.random.default_rng(43)
    n = 400
    actions = rng.integers(0, 2, n)
    rewards = np.abs(rng.normal(1.0, 0.5, n))
    logging = np.where(actions == 1, 0.6, 0.4)
    target = np.where(actions == 1, 0.8, 0.2)
    df = pd.DataFrame({"a": actions, "r": rewards, "e": logging, "pi": target})
    gamma = 2.0
    res = sp.sharp_ope_unobserved(
        df,
        actions="a",
        rewards="r",
        logging_prob="e",
        target_prob="pi",
        gamma=gamma,
    )
    assert res.upper_bound == pytest.approx(gamma * res.point_estimate, rel=1e-12)
    assert res.lower_bound == pytest.approx(res.point_estimate / gamma, rel=1e-12)


# ---------------------------------------------------------------------------
# causal_policy_forest
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forest_fit():
    """3-action step-function DGP with known optimal policy value 1.0."""
    rng = np.random.default_rng(7)
    n = 600
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.normal(0, 1, n)  # irrelevant covariate
    optimal = np.where(x0 < -0.5, 0, np.where(x0 < 0.5, 1, 2))
    actions = rng.integers(0, 3, n)  # uniform logging policy
    rewards = (actions == optimal).astype(float) + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x0": x0, "x1": x1, "a": actions, "r": rewards})
    res = sp.causal_policy_forest(
        df,
        actions="a",
        rewards="r",
        covariates=["x0", "x1"],
        n_trees=15,
        depth=3,
        random_state=7,
    )
    return res, optimal


def test_policy_forest_recovers_step_function_policy(forest_fit):
    # A depth-3 tree can represent the two-threshold partition exactly;
    # forest voting should recover it for the vast majority of units.
    # Observed accuracy ~99.7%; require >90% to absorb seed sensitivity.
    res, optimal = forest_fit
    accuracy = float((res.assignments == optimal).mean())
    assert accuracy > 0.90, accuracy


def test_policy_forest_value_near_known_optimum(forest_fit):
    # Optimal policy value is exactly 1.0 (reward = indicator + mean-zero
    # noise). The DR value estimate carries nuisance-model (boosting) bias
    # plus O(1/sqrt(n)) sampling noise; abs=0.1 is generous for that yet
    # far above any non-optimal benchmark (the best constant-action policy
    # is worth <= 0.375 on this DGP, a uniform-random policy ~1/3).
    res, _ = forest_fit
    assert res.policy_value == pytest.approx(1.0, abs=0.1)
    assert np.isfinite(res.policy_value_se)
    assert res.policy_value_se >= 0.0


def test_policy_forest_assigns_every_unit_across_all_actions(forest_fit):
    # Bookkeeping identity: one assignment per unit, all three actions
    # used (each action is optimal on >= 25% of the covariate space).
    res, _ = forest_fit
    assert sum(res.action_counts.values()) == 600
    assert set(res.action_counts) == {0, 1, 2}
    assert all(count > 0 for count in res.action_counts.values())
