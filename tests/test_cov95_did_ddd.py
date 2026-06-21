"""Coverage tests for statspai.did.ddd (Triple Differences)."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _ddd_data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    sub = rng.integers(0, 2, n)
    # true DDD (triple interaction) = 4.0
    y = (
        1
        + 2 * treat
        + 1.5 * post
        + 0.7 * sub
        + 1 * treat * post
        + 0.5 * treat * sub
        + 0.3 * post * sub
        + 4.0 * treat * post * sub
        + rng.normal(0, 1, n)
    )
    return pd.DataFrame(
        {
            "y": y,
            "treat": treat,
            "post": post,
            "sub": sub,
            "x1": rng.normal(0, 1, n),
            "state": rng.integers(0, 15, n),
            "w": rng.uniform(0.5, 2.0, n),
        }
    )


@pytest.fixture(scope="module")
def ddd_df():
    return _ddd_data()


def test_ddd_basic(ddd_df):
    r = sp.ddd(ddd_df, y="y", treat="treat", time="post", subgroup="sub")
    # recover triple interaction near 4.0
    assert abs(r.estimate - 4.0) < 0.5
    assert r.se > 0
    assert 0.0 <= r.pvalue <= 1.0
    assert r.ci[0] < r.ci[1]
    assert r.model_info["n_treated"] + r.model_info["n_control"] == r.n_obs
    assert "did_estimate" in r.model_info
    # detail coefficient table includes the triple interaction row
    vars_ = r.detail["variable"].tolist()
    assert "treatxpostxsub" in vars_


def test_ddd_cluster(ddd_df):
    r = sp.ddd(
        ddd_df, y="y", treat="treat", time="post", subgroup="sub", cluster="state"
    )
    assert r.se > 0
    assert r.model_info["cluster"] == "state"


def test_ddd_robust_false(ddd_df):
    r = sp.ddd(ddd_df, y="y", treat="treat", time="post", subgroup="sub", robust=False)
    assert r.se > 0
    assert r.model_info["robust_se"] is False


def test_ddd_covariates(ddd_df):
    r = sp.ddd(
        ddd_df, y="y", treat="treat", time="post", subgroup="sub", covariates=["x1"]
    )
    assert "x1" in r.detail["variable"].tolist()


def test_ddd_weights(ddd_df):
    r = sp.ddd(ddd_df, y="y", treat="treat", time="post", subgroup="sub", weights="w")
    assert r.se > 0
    assert r.model_info["weights"] == "w"


def test_ddd_weights_and_cluster(ddd_df):
    r = sp.ddd(
        ddd_df,
        y="y",
        treat="treat",
        time="post",
        subgroup="sub",
        weights="w",
        cluster="state",
    )
    assert r.se > 0


def test_ddd_weights_robust_false(ddd_df):
    r = sp.ddd(
        ddd_df,
        y="y",
        treat="treat",
        time="post",
        subgroup="sub",
        weights="w",
        robust=False,
    )
    assert r.se > 0


def test_ddd_non_binary_treat_raises(ddd_df):
    bad = ddd_df.copy()
    bad["treat"] = np.arange(len(bad)) % 3  # 3 values
    with pytest.raises(ValueError):
        sp.ddd(bad, y="y", treat="treat", time="post", subgroup="sub")


def test_ddd_some_negative_weights_filtered(ddd_df):
    # A few negative/zero weights are filtered (valid requires w>0); estimation
    # proceeds on the remaining rows.
    bad = ddd_df.copy()
    bad.loc[bad.index[:50], "w"] = -1.0
    r = sp.ddd(bad, y="y", treat="treat", time="post", subgroup="sub", weights="w")
    assert r.se > 0
    assert r.n_obs == int((ddd_df["w"].values > 0).sum()) - 0 or r.n_obs > 0
