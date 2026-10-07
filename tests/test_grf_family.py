"""Correctness and boundary tests for the GRF-family forests.

Numerical agreement with grf is pinned in ``tests/reference_parity``
(operators exactly, forests statistically); these tests cover recovery of
known truths on small designs, the public interfaces and the failure
modes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

NT = 400  # trees: enough for stable OOB predictions, fast enough for CI


def _iv_frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    Z = rng.binomial(1, 0.5, n)
    U = rng.normal(size=n)
    W = (1.2 * Z + 0.6 * U + rng.normal(scale=0.5, size=n) > 0.6).astype(int)
    tau = 1 + X[:, 0]
    Y = tau * W + X[:, 1] + U + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    return df.assign(y=Y, w=W, z=Z, tau=tau)


def _arm_frame(n: int = 1500, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    A = rng.integers(0, 3, n)
    Y = X[:, 1] + (1 + X[:, 0]) * (A == 1) - 0.5 * (A == 2) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    return df.assign(y=Y, a=A)


def _surv_frame(n: int = 1500, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    W = rng.binomial(1, 0.5, n)
    r0 = np.exp(0.3 * X[:, 1])
    r1 = r0 * np.exp(-0.7 * (X[:, 0] > 0))
    T = rng.exponential(1 / np.where(W == 1, r1, r0))
    C = rng.exponential(3.0, n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    h = 1.5
    tau = (1 - np.exp(-r1 * h)) / r1 - (1 - np.exp(-r0 * h)) / r0
    return df.assign(t=np.minimum(T, C), d=(T <= C).astype(int), w=W, tau=tau)


COV = ["x1", "x2", "x3"]


# --------------------------------------------------------------------------- #
#  Instrumental forest
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def iv_fit():
    df = _iv_frame()
    return df, sp.iv_forest(
        df, y="y", treat="w", instrument="z", covariates=COV, n_estimators=NT
    )


def test_iv_forest_recovers_conditional_late(iv_fit):
    df, fit = iv_fit
    assert np.corrcoef(fit.cate, df.tau)[0, 1] > 0.8
    assert abs(fit.late - df.tau.mean()) < 3 * fit.se
    lo, hi = fit.ci
    assert lo < fit.late < hi
    assert fit.cate_variance is not None and np.all(fit.cate_variance >= 0)


def test_iv_forest_interfaces(iv_fit):
    df, fit = iv_fit
    assert fit.predict(df[COV].head(5)).shape == (5, 1)
    pv = fit.predict(df[COV].head(5), estimate_variance=True)
    assert list(pv.columns) == ["predictions", "variance_estimates"]
    blp = sp.best_linear_projection(fit, A=df[["x1"]])
    assert list(blp.index) == ["Intercept", "x1"]
    assert blp.loc["x1", "coef"] == pytest.approx(1.0, abs=0.35)
    assert sp.get_scores(fit).shape == (len(df),)
    vi = sp.variable_importance(fit)
    assert vi.sum() == pytest.approx(1.0) and vi.idxmax() == "x1"
    assert sp.get_provenance(fit).function == "sp.iv_forest"
    d = fit.to_dict()
    assert "late" in d and "_engine" not in d


def test_iv_forest_array_interface_alias_and_deprecations():
    df = _iv_frame(n=600, seed=3)
    with pytest.warns(DeprecationWarning, match="n_bootstrap"):
        fit = sp.instrumental_forest(
            y=df.y.to_numpy(),
            treat=df.w.to_numpy(),
            instrument=df.z.to_numpy(),
            covariates=df[COV].to_numpy(),
            n_trees=200,
            n_bootstrap=10,
        )
    assert fit.num_trees == 200
    assert fit.feature_names == ["covariates1", "covariates2", "covariates3"]


def test_iv_forest_boundaries():
    df = _iv_frame(n=300, seed=4)
    with pytest.raises(MethodIncompatibility, match="Missing columns"):
        sp.iv_forest(df, y="y", treat="w", instrument="nope", covariates=COV)
    with pytest.raises(DataInsufficient, match="no variation"):
        sp.iv_forest(df.assign(z=1), y="y", treat="w", instrument="z", covariates=COV)
    with pytest.raises(MethodIncompatibility, match="reduced_form_weight"):
        sp.iv_forest(
            df,
            y="y",
            treat="w",
            instrument="z",
            covariates=COV,
            reduced_form_weight=1.5,
        )
    with pytest.raises(MethodIncompatibility, match="max_samples"):
        sp.iv_forest(
            df, y="y", treat="w", instrument="z", covariates=COV, max_samples=0.8
        )


def test_iv_forest_drops_missing_rows():
    df = _iv_frame(n=500, seed=5)
    df.loc[:9, "x1"] = np.nan
    fit = sp.iv_forest(
        df, y="y", treat="w", instrument="z", covariates=COV, n_estimators=200
    )
    assert fit.n_obs == 490 and fit.detail["n_dropped_missing"] == 10


# --------------------------------------------------------------------------- #
#  Multi-arm forest and lm forest
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def arm_fit():
    df = _arm_frame()
    return df, sp.multi_arm_forest(
        df, y="y", treat="a", covariates=COV, n_estimators=NT
    )


def test_multi_arm_forest_recovers_effects(arm_fit):
    df, fit = arm_fit
    assert fit.arms == [0, 1, 2] and fit.reference == 0
    assert abs(fit.ate[1] - (1 + df.x1).mean()) < 3 * fit.ate_se[1]
    assert abs(fit.ate[2] + 0.5) < 3 * fit.ate_se[2]
    assert np.corrcoef(fit.cate[1], 1 + df.x1)[0, 1] > 0.8
    tab = fit.average_treatment_effect()
    assert list(tab.index) == ["1 - 0", "2 - 0"]
    assert fit.predict(df[COV].head(4)).shape == (4, 2)
    blp = fit.best_linear_projection(A=df[["x1"]])
    assert blp.index.nlevels == 2 and blp.shape[0] == 4


def test_multi_arm_forest_reference_and_labels():
    df = _arm_frame(n=800, seed=6)
    df["arm"] = df.a.map({0: "ctrl", 1: "drugA", 2: "drugB"})
    fit = sp.multi_arm_forest(
        df, y="y", treat="arm", covariates=COV, reference="drugB", n_estimators=200
    )
    assert fit.reference == "drugB"
    assert set(fit.ate) == {"ctrl", "drugA"}
    assert fit.get_scores().shape == (len(df), 2)


def test_multi_arm_forest_boundaries():
    df = _arm_frame(n=300, seed=7)
    with pytest.raises(MethodIncompatibility, match="reference"):
        sp.multi_arm_forest(df, y="y", treat="a", covariates=COV, reference=9)
    with pytest.raises(DataInsufficient, match="two treatment arms"):
        sp.multi_arm_forest(df.assign(a=0), y="y", treat="a", covariates=COV)
    with pytest.raises(MethodIncompatibility, match="propensity_bounds"):
        sp.multi_arm_forest(
            df, y="y", treat="a", covariates=COV, propensity_bounds=(0.9, 0.1)
        )


def test_lm_forest_recovers_varying_coefficients():
    rng = np.random.default_rng(8)
    n = 1500
    X = rng.normal(size=(n, 3))
    W = rng.normal(size=(n, 2))
    Y = X[:, 1] + (1 + X[:, 0]) * W[:, 0] - W[:, 1] + rng.normal(size=n)
    fit = sp.lm_forest(y=Y, regressors=W, covariates=X, n_estimators=NT)
    assert fit.coefficients.shape == (n, 2, 1)
    assert np.corrcoef(fit.coefficients[:, 0, 0], 1 + X[:, 0])[0, 1] > 0.9
    assert fit.coefficients[:, 1, 0].mean() == pytest.approx(-1.0, abs=0.1)
    out = fit.predict(X[:3], estimate_variance=True)
    assert out["predictions"].shape == (3, 2, 1)
    assert np.all(out["variance_estimates"] >= 0)


# --------------------------------------------------------------------------- #
#  Prediction forests
# --------------------------------------------------------------------------- #


def test_regression_and_multi_regression_forests():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(1000, 3))
    y = X[:, 0] ** 2 + rng.normal(scale=0.5, size=1000)
    rf = sp.regression_forest(y=y, covariates=X, n_estimators=300)
    assert np.corrcoef(rf.predictions, X[:, 0] ** 2)[0, 1] > 0.85
    out = rf.predict(X[:4], estimate_variance=True)
    assert out.shape == (4, 2)
    Y2 = np.c_[X[:, 0], -X[:, 0]] + rng.normal(size=(1000, 2))
    mf = sp.multi_regression_forest(y=Y2, covariates=X, n_estimators=300)
    assert mf.predictions.shape == (1000, 2)
    assert np.corrcoef(mf.predictions[:, 1], -X[:, 0])[0, 1] > 0.8


def test_probability_forest_returns_probabilities():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(900, 2))
    labels = np.where(X[:, 0] > 0.5, "b", np.where(X[:, 0] < -0.5, "a", "c"))
    pf = sp.probability_forest(y=labels, covariates=X, n_estimators=300)
    assert pf.classes == ["a", "b", "c"]
    np.testing.assert_allclose(pf.predictions.sum(axis=1), 1.0, atol=1e-12)
    assert np.mean(np.array(pf.classes)[pf.predictions.argmax(1)] == labels) > 0.9
    with pytest.raises(DataInsufficient, match="two classes"):
        sp.probability_forest(y=np.zeros(50), covariates=X[:50])


def test_quantile_forest_orders_and_tracks_scale():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1500, 2))
    sd = 1 + (X[:, 1] > 0)
    y = X[:, 0] + sd * rng.normal(size=1500)
    qf = sp.quantile_forest(y=y, covariates=X, n_estimators=400)
    q = qf.predictions
    assert np.all(q[:, 0] <= q[:, 1]) and np.all(q[:, 1] <= q[:, 2])
    width = q[:, 2] - q[:, 0]
    assert width[X[:, 1] > 0].mean() > 1.5 * width[X[:, 1] <= 0].mean()
    assert list(qf.predict(X[:2], quantiles=[0.25, 0.75]).columns) == ["q0.25", "q0.75"]
    with pytest.raises(MethodIncompatibility, match="between 0 and 1"):
        sp.quantile_forest(y=y, covariates=X, quantiles=[0.5, 1.0])


def test_survival_forest_curves():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(1000, 2))
    rate = np.exp(0.8 * X[:, 0])
    T = rng.exponential(1 / rate)
    C = rng.exponential(2.0, 1000)
    sf = sp.survival_forest(
        time=np.minimum(T, C),
        event=(T <= C).astype(int),
        covariates=X,
        n_estimators=300,
    )
    S = sf.predictions
    assert np.all(np.diff(S, axis=1) <= 1e-12) and np.all((S >= 0) & (S <= 1))
    at = sf.predict(failure_times=[0.5])
    truth = np.exp(-rate * 0.5)
    assert np.corrcoef(at.iloc[:, 0], truth)[0, 1] > 0.9
    na = sf.predict(X[:3], prediction_type="Nelson-Aalen")
    assert na.shape[0] == 3
    with pytest.raises(DataInsufficient, match="no observed events"):
        sp.survival_forest(time=T, event=np.zeros(1000), covariates=X)
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.survival_forest(time=-T, event=(T <= C).astype(int), covariates=X)


# --------------------------------------------------------------------------- #
#  Causal survival forest
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target", ["RMST", "survival_probability"])
def test_causal_survival_forest_recovers_effect(target):
    df = _surv_frame()
    fit = sp.causal_survival_forest(
        df,
        time="t",
        event="d",
        treat="w",
        covariates=COV,
        horizon=1.5,
        target=target,
        n_estimators=NT,
    )
    assert fit.target == target and fit.horizon == 1.5
    if target == "RMST":
        assert abs(fit.ate - df.tau.mean()) < 3 * fit.se
        assert fit.ate_rmst == fit.ate
        assert np.corrcoef(fit.cate, df.tau)[0, 1] > 0.6
    else:
        assert np.isnan(fit.ate_rmst)
    assert fit.best_linear_projection(A=df[["x1"]]).shape[0] == 2
    assert fit.predict(df[COV].head(3)).shape == (3, 1)


def test_causal_survival_forest_boundaries():
    df = _surv_frame(n=400, seed=13)
    with pytest.raises(MethodIncompatibility, match="target"):
        sp.causal_survival_forest(
            df, time="t", event="d", treat="w", covariates=COV, target="median"
        )
    with pytest.raises(MethodIncompatibility, match="binary"):
        sp.causal_survival_forest(
            df.assign(w=df.w * 2), time="t", event="d", treat="w", covariates=COV
        )
    with pytest.raises(MethodIncompatibility, match="exceed"):
        sp.causal_survival_forest(
            df, time="t", event="d", treat="w", covariates=COV, horizon=1e-9
        )


# --------------------------------------------------------------------------- #
#  Post-estimation tools and seeding
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cf_fit():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(1200, 3))
    T = rng.binomial(1, 0.5, 1200)
    Y = (1 + X[:, 0]) * T + X[:, 1] + rng.normal(size=1200)
    return X, sp.causal_forest(Y=Y, T=T, X=X, n_estimators=NT, random_state=1)


def test_tools_accept_causal_forest(cf_fit):
    X, cf = cf_fit
    vi = sp.variable_importance(cf)
    assert vi.sum() == pytest.approx(1.0) and vi.index[vi.argmax()] == vi.index[0]
    for vt in ("HC0", "HC1", "HC2", "HC3"):
        blp = sp.best_linear_projection(cf, A=X[:, :1], vcov_type=vt)
        assert blp.loc["A1", "coef"] == pytest.approx(1.0, abs=0.3)
    scores = sp.get_scores(cf)
    assert scores.shape == (1200,)


def test_tools_reject_bad_inputs(cf_fit):
    X, cf = cf_fit
    with pytest.raises(MethodIncompatibility, match="vcov_type"):
        sp.best_linear_projection(cf, A=X[:, :1], vcov_type="HC9")
    with pytest.raises(MethodIncompatibility, match="one row per training"):
        sp.best_linear_projection(cf, A=X[:10, :1])
    with pytest.raises(MethodIncompatibility, match="max_depth"):
        sp.variable_importance(cf, max_depth=0)
    with pytest.raises(MethodIncompatibility, match="unsupported"):
        sp.variable_importance(object())
    rf = sp.regression_forest(y=X[:, 0], covariates=X, n_estimators=100)
    with pytest.raises(MethodIncompatibility, match="no doubly-robust scores"):
        sp.get_scores(rf)


def test_seeds_are_reproducible_and_independent():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] + rng.normal(size=500)
    a = sp.regression_forest(y=y, covariates=X, n_estimators=200, random_state=1)
    b = sp.regression_forest(y=y, covariates=X, n_estimators=200, random_state=1)
    c = sp.regression_forest(y=y, covariates=X, n_estimators=200, random_state=2)
    np.testing.assert_array_equal(a.predictions, b.predictions)
    # Consecutive seeds used to share all but one group of trees.
    assert np.mean(np.abs(a.predictions - c.predictions)) > 0.02


def test_causal_forest_variable_importance_default_is_split(cf_fit):
    """The default became grf's split measure in 1.39, as announced."""
    _, cf = cf_fit
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        default = cf.variable_importance()
        split = cf.variable_importance(method="split")
        perm = cf.variable_importance(method="permutation")
    pd.testing.assert_series_equal(default, split)
    pd.testing.assert_series_equal(split, sp.variable_importance(cf))
    assert split.sum() == pytest.approx(1.0) and perm.sum() == pytest.approx(1.0)


def test_causal_survival_alias_is_the_same_function():
    assert sp.causal_survival is sp.causal_survival_forest
