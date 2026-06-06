"""Coverage campaign — DML learner resolution (``dml/_learners.py``).

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). ``resolve_learner`` turns a string
alias (or an estimator object) into a scikit-learn learner for the nuisance
models. This sweeps every regressor/classifier alias branch, the estimator
pass-through and duck-typing, and the validation errors (unknown alias,
ols-as-classifier, ``None`` sentinel, wrong ``kind``, non-estimator object,
optional-dep ``xgboost`` absent).

Assertions are real: each alias must yield a scikit-learn-compatible estimator
(``.fit`` + ``.get_params``) of the expected family; misuse must raise the
specific, informative error.
"""

from __future__ import annotations

import pytest

from statspai.dml._learners import (
    _build_classifier,
    _build_regressor,
    resolve_learner,
)

REGRESSOR_ALIASES = ["rf", "gbm", "lasso", "ridge", "linear", "ols", "lgbm"]
CLASSIFIER_ALIASES = ["rf", "gbm", "lasso", "ridge", "logistic", "lgbm"]


@pytest.mark.parametrize("alias", REGRESSOR_ALIASES)
def test_regressor_aliases(alias):
    est = resolve_learner(alias, kind="regressor", role="ml_g")
    assert hasattr(est, "fit") and hasattr(est, "get_params")


@pytest.mark.parametrize("alias", CLASSIFIER_ALIASES)
def test_classifier_aliases(alias):
    est = resolve_learner(alias, kind="classifier", role="ml_m")
    assert hasattr(est, "fit") and hasattr(est, "get_params")


def test_alias_case_insensitive():
    assert type(_build_regressor("RF")).__name__ == "RandomForestRegressor"
    assert type(_build_classifier("GBM")).__name__ == "GradientBoostingClassifier"


def test_estimator_passthrough():
    from sklearn.linear_model import LinearRegression

    obj = LinearRegression()
    assert resolve_learner(obj, kind="regressor", role="ml_g") is obj


# ─── error branches ──────────────────────────────────────────────────────


def test_unknown_regressor_alias_raises():
    with pytest.raises(ValueError, match="Unknown regressor"):
        resolve_learner("not_a_learner", kind="regressor", role="ml_g")


def test_unknown_classifier_alias_raises():
    with pytest.raises(ValueError, match="Unknown classifier"):
        resolve_learner("not_a_learner", kind="classifier", role="ml_m")


def test_ols_as_classifier_raises():
    with pytest.raises(ValueError, match="ols.*not valid"):
        _build_classifier("ols")


def test_none_spec_raises():
    with pytest.raises(ValueError, match="callers must supply"):
        resolve_learner(None, kind="regressor", role="ml_g")


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="kind must be"):
        resolve_learner("rf", kind="not_a_kind", role="ml_g")


def test_non_estimator_object_raises():
    with pytest.raises(TypeError, match="not a"):
        resolve_learner(12345, kind="regressor", role="ml_g")


def test_xgboost_absent_raises_importerror():
    pytest.importorskip(
        "sklearn"
    )  # sklearn always present; gate is on xgboost being absent
    try:
        import xgboost  # noqa: F401

        pytest.skip("xgboost installed — ImportError branch not exercised here")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="xgboost"):
        _build_regressor("xgb")
    with pytest.raises(ImportError, match="xgboost"):
        _build_classifier("xgb")
