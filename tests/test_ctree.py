"""Properties and edge cases of ``sp.ctree``.

Numerical agreement with ``partykit`` is in
``tests/reference_parity/test_ctree_partykit_parity.py``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _step(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"x": rng.uniform(size=n), "z1": rng.normal(size=n), "z2": rng.normal(size=n)}
    )
    df["y"] = np.where(df["x"] > 0.6, 3.0, 0.0) + rng.normal(size=n)
    return df


def test_recovers_a_planted_cut_point():
    df = _step()
    fit = sp.ctree(df, y="y", covariates=["x", "z1", "z2"])
    root = fit.nodes.loc[1]
    assert root["variable"] == "x"
    # The cut is the largest observed value sent left: just below 0.6.
    assert 0.55 < root["split"] <= 0.6
    assert df.loc[df["x"] <= root["split"], "x"].max() == root["split"]


def test_root_test_is_the_squared_correlation_statistic():
    # One numeric covariate, numeric outcome: the quadratic form is
    # (n - 1) r^2, the permutation form of the correlation test.
    df = _step(seed=3)
    fit = sp.ctree(df, y="y", covariates=["z1"], test="univariate")
    r = np.corrcoef(df["z1"], df["y"])[0, 1]
    row = fit.tests.iloc[0]
    assert row["statistic"] == pytest.approx((len(df) - 1) * r**2, rel=1e-12)
    assert row["df"] == 1


def test_nominal_covariate_statistic_is_the_one_way_layout():
    # Level indicators against a numeric outcome: the statistic is
    # (n - 1) * between-group sum of squares / total sum of squares.
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"g": rng.choice(list("abcd"), size=300)})
    df["y"] = rng.normal(size=300) + (df["g"] == "c")
    fit = sp.ctree(df, y="y", covariates=["g"])
    total = ((df["y"] - df["y"].mean()) ** 2).sum()
    grp = df.groupby("g")["y"]
    between = (grp.size() * (grp.mean() - df["y"].mean()) ** 2).sum()
    row = fit.tests.iloc[0]
    assert row["statistic"] == pytest.approx((len(df) - 1) * between / total, rel=1e-10)
    assert row["df"] == 3


def test_binary_outcome_gives_the_same_tree_as_regression_on_it():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"x": rng.normal(size=500), "w": rng.normal(size=500)})
    df["y"] = (rng.uniform(size=500) < 1 / (1 + np.exp(-2 * df["x"]))).astype(int)
    cls = sp.ctree(df, y="y", covariates=["x", "w"], task="classification")
    reg = sp.ctree(df, y="y", covariates=["x", "w"])
    assert reg.task == "regression" and cls.task == "classification"
    assert (cls.node_id == reg.node_id).all()
    np.testing.assert_allclose(
        cls.tests["statistic"], reg.tests["statistic"], rtol=1e-10
    )
    np.testing.assert_allclose(
        cls.predict(df, type="prob")[:, 1], reg.fitted.astype(float)
    )


def test_size_under_the_null_is_at_most_alpha():
    # Nothing predicts the outcome: with the Bonferroni adjustment the
    # root should split in no more than about 5% of samples. 400 samples,
    # so the upper 99.9% binomial bound for a 5% event is 35.
    rng = np.random.default_rng(11)
    splits = 0
    for _ in range(400):
        df = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("abcdef"))
        df["y"] = rng.normal(size=120)
        splits += sp.ctree(df, y="y", covariates=list("abcdef")).n_leaves > 1
    assert 4 <= splits <= 35


def test_constant_outcome_and_depth_zero_give_a_single_leaf():
    df = _step()
    df["k"] = 1.0
    assert sp.ctree(df, y="k", covariates=["x"]).n_leaves == 1
    stump = sp.ctree(df, y="y", covariates=["x"], max_depth=0)
    assert stump.n_leaves == 1 and len(stump.nodes) == 1
    assert stump.fitted.astype(float) == pytest.approx(df["y"].mean())


def test_constant_covariate_is_not_tested_and_not_counted():
    df = _step()
    df["const"] = 2.0
    with_const = sp.ctree(df, y="y", covariates=["x", "z1", "const"])
    without = sp.ctree(df, y="y", covariates=["x", "z1"])
    assert "const" not in set(with_const.tests["variable"])
    np.testing.assert_allclose(with_const.tests["p_value"], without.tests["p_value"])


def test_children_respect_the_size_limits():
    df = _step(n=300, seed=2)
    fit = sp.ctree(
        df, y="y", covariates=["x", "z1", "z2"], alpha=0.5, min_bucket=25, min_split=60
    )
    leaves = fit.nodes[fit.nodes["variable"].isna()]
    assert leaves["n"].min() >= 25
    inner = fit.nodes[fit.nodes["variable"].notna()]
    assert inner["n"].min() >= 60
    assert leaves["n"].sum() == len(df)


def test_predict_types_and_string_classes():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"x": rng.normal(size=400), "s": rng.choice(["m", "f"], size=400)}
    )
    share = 0.1 + 0.55 * (df["s"] == "f") + 0.3 * (df["x"] > 0)
    df["y"] = np.where(rng.uniform(size=400) < share, "yes", "no")
    fit = sp.ctree(df, y="y", covariates=["x", "s"])
    assert fit.task == "classification" and fit.classes == ["no", "yes"]
    prob = fit.predict(df, type="prob")
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)
    assert (fit.predict(df) == fit.fitted).all()
    assert (
        np.asarray(fit.classes, dtype=object)[prob.argmax(axis=1)] == fit.fitted
    ).all()
    assert "s in {" in fit.summary()
    with pytest.raises(MethodIncompatibility):
        fit.predict(df, type="link")
    with pytest.raises(MethodIncompatibility):
        fit.predict(df.assign(s="other"))
    with pytest.raises(MethodIncompatibility):
        fit.predict(df.drop(columns="s"))


def test_inputs_that_are_refused():
    df = _step()
    with pytest.raises(DataInsufficient):
        sp.ctree(df.assign(x=df["x"].where(df["x"] > 0.01)), y="y", covariates=["x"])
    with pytest.raises(DataInsufficient):
        sp.ctree(df.assign(y=df["y"].where(df["x"] > 0.01)), y="y", covariates=["x"])
    many = df.assign(g=[f"l{i % 13}" for i in range(len(df))])
    with pytest.raises(MethodIncompatibility, match="13 unordered levels"):
        sp.ctree(many, y="y", covariates=["g"])
    for bad in (
        {"alpha": 1.5},
        {"test": "montecarlo"},
        {"split_stat": "gini"},
        {"task": "survival"},
        {"min_bucket": 0},
        {"max_depth": -1},
    ):
        with pytest.raises(MethodIncompatibility):
            sp.ctree(df, y="y", covariates=["x"], **bad)
    with pytest.raises(MethodIncompatibility):
        sp.ctree(df, y="y", covariates=[])
    with pytest.raises(MethodIncompatibility):
        sp.ctree(df, y="y", covariates=["x", "nope"])
    with pytest.raises(MethodIncompatibility):
        sp.ctree(df, y="y", covariates=["x", "y"])
    with pytest.raises(MethodIncompatibility):
        sp.ctree(df.assign(lab="a"), y="lab", covariates=["x"], task="regression")
    with pytest.raises(MethodIncompatibility):
        sp.ctree(df, y="y", covariates=["x"]).predict(df, type="prob")


def test_no_marginal_association_means_no_split():
    # An exclusive-or outcome depends on both covariates and on neither
    # marginally. The tests are marginal, so the tree does not split.
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"x": rng.normal(size=400), "s": rng.choice(["m", "f"], size=400)}
    )
    df["y"] = np.where((df["x"] > 0) ^ (df["s"] == "f"), "yes", "no")
    fit = sp.ctree(df, y="y", covariates=["x", "s"])
    assert fit.n_leaves == 1
    assert (fit.tests["p_value"] > 0.05).all()
