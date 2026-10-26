"""Option scan (Oct 2026) of ``statspai.policy_learning``."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def test_policy_tree_gain_is_unbiased_under_null():
    """No unit benefits, so every policy gains exactly 0.

    The in-sample gain is >= 0 by construction (its 95%
    interval covering 0 in 22 of 100 draws before the fix). The cross-fitted
    gain is centred on 0. Its interval is still short of nominal at this
    exact null (81% at n = 400, 85% at n = 1600 in a 100-draw study; 92-97%
    with heterogeneity), because the folds' trees are grown on each other's
    scores; the docstring says so. The assertion is the bias, which the
    construction removes, plus a coverage floor well above the old 22%.
    """
    gains, covered, gains_in = [], 0, []
    reps = 60
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        n = 400
        df = pd.DataFrame(
            {
                "y": rng.normal(size=n),
                "d": rng.integers(0, 2, n),
                "x1": rng.uniform(size=n),
                "x2": rng.uniform(size=n),
            }
        )
        scores = rng.normal(size=n)
        res = sp.policy_tree(
            df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            max_depth=2,
            min_leaf_size=25,
            scores=scores,
        )
        g, se = res["value_gain"], res["value_gain_se"]
        gains.append(g)
        gains_in.append(res["value_gain_in_sample"])
        covered += abs(g) <= 1.96 * se
    gains = np.asarray(gains)
    # MC standard error of the mean gain over 60 draws.
    assert abs(gains.mean()) < 3 * gains.std(ddof=1) / np.sqrt(reps)
    assert np.mean(gains_in) > 5 * gains.std(ddof=1) / np.sqrt(reps)
    assert covered / reps >= 0.7, covered / reps


def test_ips_fitted_propensity_with_actions_not_starting_at_zero():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 2))  # unrelated to the logged action
    A = np.where(rng.uniform(size=n) < 0.7, 2, 1)
    R = (A == 2) + rng.normal(scale=0.5, size=n)
    target = np.zeros((n, 3))
    target[:, 2] = 1.0  # always play action 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = sp.ips(X, A, R, pi_target=target)
        true_pb = np.where(A == 2, 0.7, 0.3)
        known = sp.ips(X, A, R, pi_target=target, pi_behavior=true_pb)
    assert abs(fitted.value - known.value) < 0.05


def _noise_frame(n: int, seed: int):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": rng.integers(0, 2, n),
            "x1": rng.uniform(size=n),
            "x2": rng.uniform(size=n),
        }
    )
    scores = 0.8 * (df["x1"].to_numpy() > 0.5) - 0.3 + rng.normal(size=n)
    return df, scores


def test_policy_tree_cross_fit_value_is_the_held_out_value():
    """Identity: the honest value is the mean of each held-out row's score
    under the tree grown on the other folds."""
    from sklearn.model_selection import KFold

    df, scores = _noise_frame(500, 11)
    kw = dict(
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        max_depth=2,
        min_leaf_size=25,
        n_folds=5,
        random_state=3,
    )
    res = sp.policy_tree(df, scores=scores, **kw)
    pi = np.empty(len(df), dtype=int)
    u = np.empty(len(df))
    X = df[["x1", "x2"]].to_numpy()
    for tr, te in KFold(5, shuffle=True, random_state=3).split(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sub = sp.policy_tree(
                df.iloc[tr], scores=scores[tr], value_evaluation="in_sample", **kw
            )
        pi[te] = sub["tree"].predict(X[te])
        u[te] = 1.0 if scores[tr].mean() > 0 else 0.0
    assert res["value_evaluation"] == "cross_fit"
    assert res["value_policy"] == pytest.approx(np.mean(scores * pi), abs=1e-12)
    assert res["value_gain"] == pytest.approx(np.mean(scores * (pi - u)), abs=1e-12)


def test_policy_tree_in_sample_value_warns_and_keeps_old_numbers():
    df, scores = _noise_frame(500, 12)
    kw = dict(
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        max_depth=2,
        min_leaf_size=25,
        scores=scores,
    )
    with pytest.warns(UserWarning, match="optimistic"):
        old = sp.policy_tree(df, value_evaluation="in_sample", **kw)
    pol = old["policy"]
    assert old["value_policy"] == pytest.approx(np.mean(scores * pol), abs=1e-14)
    new = sp.policy_tree(df, **kw)
    # Same deployed tree either way; only the evaluation differs.
    np.testing.assert_array_equal(new["policy"], pol)
    assert new["value_policy_in_sample"] == old["value_policy"]


def test_policy_tree_rejects_unknown_value_evaluation():
    df, scores = _noise_frame(200, 13)
    with pytest.raises(ValueError, match="value_evaluation"):
        sp.policy_tree(
            df,
            y="y",
            treat="d",
            covariates=["x1"],
            scores=scores,
            value_evaluation="honest",
        )
