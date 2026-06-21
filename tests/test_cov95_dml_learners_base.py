"""Coverage tests for statspai.dml._learners and _base.

Targets the string-alias learner resolver (every regressor / classifier
branch, error paths) and the base-class plumbing (sample_weight
validation, weighted-fit fallback, multi-rep aggregation, instrument
validation). Real small synthetic data only — no mocks of the numeric
path.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from statspai.dml._learners import (
    resolve_learner,
    _build_regressor,
    _build_classifier,
    _is_estimator_like,
    _alias_error_message,
)
from statspai.dml._base import _DoubleMLBase
from statspai.dml import dml
from statspai.exceptions import MethodIncompatibility

_HAS_LGBM = importlib.util.find_spec("lightgbm") is not None
_HAS_XGB = importlib.util.find_spec("xgboost") is not None


# --------------------------------------------------------------------------
# _learners.py — regressor aliases
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alias,cls_name",
    [
        ("rf", "RandomForestRegressor"),
        ("random_forest", "RandomForestRegressor"),
        ("randomforest", "RandomForestRegressor"),
        ("gbm", "GradientBoostingRegressor"),
        ("gbr", "GradientBoostingRegressor"),
        ("gradient_boosting", "GradientBoostingRegressor"),
        ("lasso", "LassoCV"),
        ("ridge", "RidgeCV"),
        ("linear", "LinearRegression"),
        ("ols", "LinearRegression"),
        ("RF", "RandomForestRegressor"),  # case-insensitive
    ],
)
def test_build_regressor_aliases(alias, cls_name):
    est = _build_regressor(alias)
    assert type(est).__name__ == cls_name
    assert hasattr(est, "fit") and hasattr(est, "get_params")


@pytest.mark.parametrize(
    "alias,cls_name",
    [
        ("rf", "RandomForestClassifier"),
        ("gbm", "GradientBoostingClassifier"),
        ("gbc", "GradientBoostingClassifier"),
        ("lasso", "LogisticRegressionCV"),
        ("ridge", "LogisticRegressionCV"),
        ("linear", "LogisticRegression"),
        ("logistic", "LogisticRegression"),
        ("logit", "LogisticRegression"),
        ("LOGISTIC", "LogisticRegression"),
    ],
)
def test_build_classifier_aliases(alias, cls_name):
    est = _build_classifier(alias)
    assert type(est).__name__ == cls_name


def test_build_classifier_lasso_is_l1():
    est = _build_classifier("lasso")
    assert est.get_params()["penalty"] == "l1"


def test_build_classifier_ridge_is_l2():
    est = _build_classifier("ridge")
    assert est.get_params()["penalty"] == "l2"


def test_classifier_ols_rejected():
    with pytest.raises(ValueError, match="not valid for a classifier"):
        _build_classifier("ols")


def test_unknown_regressor_alias():
    with pytest.raises(ValueError, match="Unknown regressor alias"):
        _build_regressor("not_a_model")


def test_unknown_classifier_alias():
    with pytest.raises(ValueError, match="Unknown classifier alias"):
        _build_classifier("not_a_model")


@pytest.mark.skipif(not _HAS_XGB, reason="xgboost not installed")
def test_xgb_regressor_alias():
    assert type(_build_regressor("xgb")).__name__ == "XGBRegressor"


@pytest.mark.skipif(_HAS_XGB, reason="xgboost IS installed")
def test_xgb_regressor_missing_raises():
    with pytest.raises(ImportError, match="xgboost"):
        _build_regressor("xgb")


@pytest.mark.skipif(_HAS_XGB, reason="xgboost IS installed")
def test_xgb_classifier_missing_raises():
    with pytest.raises(ImportError, match="xgboost"):
        _build_classifier("xgboost")


@pytest.mark.skipif(not _HAS_LGBM, reason="lightgbm not installed")
def test_lgbm_regressor_alias():
    assert type(_build_regressor("lgbm")).__name__ == "LGBMRegressor"


@pytest.mark.skipif(not _HAS_LGBM, reason="lightgbm not installed")
def test_lgbm_classifier_alias():
    assert type(_build_classifier("lightgbm")).__name__ == "LGBMClassifier"


def test_alias_error_message_lists_universe():
    msg = _alias_error_message("foo", kind="regressor")
    assert "regressor" in msg and "foo" in msg


def test_is_estimator_like():
    from sklearn.linear_model import LinearRegression

    assert _is_estimator_like(LinearRegression())
    assert not _is_estimator_like("rf")
    assert not _is_estimator_like(42)


# --------------------------------------------------------------------------
# resolve_learner top-level dispatch
# --------------------------------------------------------------------------
def test_resolve_none_raises():
    with pytest.raises(ValueError, match="callers must supply"):
        resolve_learner(None, kind="regressor", role="ml_g")


def test_resolve_string_regressor():
    assert (
        type(resolve_learner("rf", kind="regressor", role="ml_g")).__name__
        == "RandomForestRegressor"
    )


def test_resolve_string_classifier():
    assert (
        type(resolve_learner("rf", kind="classifier", role="ml_m")).__name__
        == "RandomForestClassifier"
    )


def test_resolve_bad_kind():
    with pytest.raises(ValueError, match="kind must be"):
        resolve_learner("rf", kind="banana", role="ml_g")


def test_resolve_estimator_passthrough():
    from sklearn.linear_model import LinearRegression

    est = LinearRegression()
    assert resolve_learner(est, kind="regressor", role="ml_g") is est


def test_resolve_bad_type_raises():
    with pytest.raises(TypeError, match="not a"):
        resolve_learner(42, kind="regressor", role="ml_g")


# --------------------------------------------------------------------------
# _base.py — sample_weight validation & weighted-fit fallback
# --------------------------------------------------------------------------
@pytest.fixture
def plr_df():
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n)
    y = 1.5 * d + x1 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def test_sample_weight_string_column_missing(plr_df):
    with pytest.raises(ValueError, match="sample_weight column"):
        dml(
            plr_df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            sample_weight="no_such_col",
        )


def test_sample_weight_wrong_length(plr_df):
    with pytest.raises(ValueError, match="must be 1-D of length"):
        dml(plr_df, y="y", treat="d", covariates=["x1", "x2"], sample_weight=np.ones(5))


def test_sample_weight_negative(plr_df):
    w = np.ones(len(plr_df))
    w[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        dml(plr_df, y="y", treat="d", covariates=["x1", "x2"], sample_weight=w)


def test_sample_weight_nonfinite(plr_df):
    w = np.ones(len(plr_df))
    w[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        dml(plr_df, y="y", treat="d", covariates=["x1", "x2"], sample_weight=w)


def test_sample_weight_zero_mass(plr_df):
    w = np.zeros(len(plr_df))
    with pytest.raises(ValueError, match="zero total mass"):
        dml(plr_df, y="y", treat="d", covariates=["x1", "x2"], sample_weight=w)


def test_sample_weight_column_name(plr_df):
    df = plr_df.copy()
    df["w"] = np.abs(np.random.default_rng(1).normal(size=len(df))) + 0.1
    res = dml(df, y="y", treat="d", covariates=["x1", "x2"], sample_weight="w")
    assert np.isfinite(res.estimate)


def test_sample_weight_unsupported_model_raises():
    # plr/irm/pliv/iivm support weights; nothing else is registered, so
    # exercise the typed unsupported-model guard via a subclass.
    class NoWeightModel(_DoubleMLBase):
        _MODEL_TAG = "FOO"
        _SUPPORTS_SAMPLE_WEIGHT = False

    df = pd.DataFrame({"y": [1.0, 2, 3], "d": [0.0, 1, 0], "x1": [1.0, 2, 3]})
    with pytest.raises(MethodIncompatibility, match="not yet supported"):
        NoWeightModel(df, y="y", treat="d", covariates=["x1"], sample_weight=np.ones(3))


def test_fit_weighted_fallback_warns():
    # A learner whose .fit does not accept sample_weight should trigger
    # the RuntimeWarning fallback path in _fit_weighted.
    from sklearn.base import BaseEstimator, RegressorMixin

    class NoWeightReg(BaseEstimator, RegressorMixin):
        def fit(self, X, y):  # no sample_weight kwarg
            self.mean_ = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean_)

    X = np.random.default_rng(0).normal(size=(20, 2))
    y = np.random.default_rng(1).normal(size=20)
    with pytest.warns(RuntimeWarning, match="does not accept"):
        clf = _DoubleMLBase._fit_weighted(NoWeightReg(), X, y, np.ones(20))
    assert hasattr(clf, "mean_")


def test_fit_weighted_no_weights():
    from sklearn.linear_model import LinearRegression

    X = np.random.default_rng(0).normal(size=(20, 2))
    y = np.random.default_rng(1).normal(size=20)
    clf = _DoubleMLBase._fit_weighted(LinearRegression(), X, y, None)
    assert hasattr(clf, "coef_")


def test_aggregate_diagnostics_mixed_types():
    per_rep = [
        {"flag": True, "count": 2, "val": 1.0, "lst": [1, 2], "label": "a"},
        {"flag": False, "count": 3, "val": 3.0, "lst": [3], "label": "b"},
    ]
    merged = _DoubleMLBase._aggregate_diagnostics(per_rep)
    assert merged["flag"] is True  # any()
    assert merged["count"] == 5  # sum
    assert merged["val"] == pytest.approx(2.0)  # mean
    assert merged["lst"] == [1, 2, 3]  # concat
    assert merged["label"] == "a"  # passthrough (last-of-first)


def test_aggregate_diagnostics_empty():
    assert _DoubleMLBase._aggregate_diagnostics([]) == {}


def test_aggregate_diagnostics_all_nan():
    merged = _DoubleMLBase._aggregate_diagnostics(
        [{"v": float("nan")}, {"v": float("nan")}]
    )
    assert np.isnan(merged["v"])


def test_base_fit_one_rep_not_implemented():
    df = pd.DataFrame({"y": [1.0, 2, 3], "d": [0.0, 1, 0], "x1": [1.0, 2, 3]})
    base = _DoubleMLBase(df, y="y", treat="d", covariates=["x1"])
    with pytest.raises(NotImplementedError):
        base._fit_one_rep(None, None, None, None, 3, 0)


# --------------------------------------------------------------------------
# _base.py — instrument validation
# --------------------------------------------------------------------------
def test_instrument_supplied_to_noniv_raises(plr_df):
    df = plr_df.copy()
    df["z"] = np.random.default_rng(2).normal(size=len(df))
    with pytest.raises(ValueError, match="only valid when"):
        dml(df, y="y", treat="d", covariates=["x1", "x2"], model="plr", instrument="z")


def test_iv_model_without_instrument_raises(plr_df):
    with pytest.raises(ValueError, match="requires an"):
        dml(plr_df, y="y", treat="d", covariates=["x1", "x2"], model="pliv")


def test_iivm_multiple_scalar_instruments_rejected():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=n)
    z1 = rng.binomial(1, 0.5, n).astype(float)
    z2 = rng.binomial(1, 0.5, n).astype(float)
    d = rng.binomial(1, 0.5, n).astype(float)
    y = d + x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x, "z1": z1, "z2": z2})
    with pytest.raises(ValueError, match="single scalar"):
        dml(
            df,
            y="y",
            treat="d",
            covariates=["x"],
            model="iivm",
            instrument=["z1", "z2"],
        )


def test_string_alias_through_dml(plr_df):
    res = dml(
        plr_df, y="y", treat="d", covariates=["x1", "x2"], ml_g="linear", ml_m="linear"
    )
    assert np.isfinite(res.estimate)
    assert res.model_info["ml_g"] == "LinearRegression"


# --------------------------------------------------------------------------
# double_ml.py — dispatcher + legacy DoubleML façade
# --------------------------------------------------------------------------
def test_dml_invalid_model_raises(plr_df):
    with pytest.raises(ValueError, match="model must be"):
        dml(plr_df, y="y", treat="d", covariates=["x1"], model="nope")


def test_legacy_doubleml_fit(plr_df):
    from statspai.dml import DoubleML

    m = DoubleML(
        plr_df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        model="plr",
        ml_g="linear",
        ml_m="linear",
    )
    res = m.fit()
    assert np.isfinite(res.estimate)
    # exercise the legacy attribute properties
    assert m.model == "plr"
    assert m.y == "y"
    assert m.treat == "d"
    assert m.covariates == ["x1", "x2"]
    assert m.instrument is None
    assert m.n_folds == 5
    assert m.n_rep == 1
    assert m.alpha == 0.05
    assert m.ml_g is not None
    assert m.ml_m is not None
    assert m.ml_r is not None
    assert m.data is plr_df


def test_legacy_doubleml_invalid_model(plr_df):
    from statspai.dml import DoubleML

    with pytest.raises(ValueError, match="model must be"):
        DoubleML(plr_df, y="y", treat="d", covariates=["x1"], model="nope")
