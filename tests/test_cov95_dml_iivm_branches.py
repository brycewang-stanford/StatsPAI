"""Coverage tests for statspai.dml.iivm estimator branches.

Targets a successful LATE fit (full score path + diagnostics), the
degenerate first-stage RuntimeError, and the subgroup-fallback helpers
(_fit_predict_subgroup / _fit_predict_classifier) including the
small-subgroup mean fallback and the empty-subgroup IdentificationFailure.
Real synthetic binary-D / binary-Z data only.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.dml import dml
from statspai.dml.iivm import DoubleMLIIVM


@pytest.fixture
def iivm_df():
    rng = np.random.default_rng(101)
    n = 1500
    x = rng.normal(size=n)
    pz = 1.0 / (1.0 + np.exp(-(0.5 * x)))
    z = rng.binomial(1, pz, n).astype(float)
    # Compliance: Z strongly shifts D.
    pd_ = 1.0 / (1.0 + np.exp(-(0.3 * x + 2.0 * z - 1.0)))
    d = rng.binomial(1, pd_, n).astype(float)
    y = 1.5 * d + x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "d": d, "x": x, "z": z})


def test_iivm_full_fit_and_diagnostics(iivm_df):
    res = dml(
        iivm_df,
        y="y",
        treat="d",
        covariates=["x"],
        model="iivm",
        instrument="z",
        n_folds=3,
    )
    assert np.isfinite(res.estimate)
    assert res.estimand == "LATE"
    diags = res.model_info["diagnostics"]
    assert "first_stage_E_psi_b" in diags
    assert "pscore_z_min" in diags
    assert abs(diags["first_stage_E_psi_b"]) > 1e-6


def test_iivm_weighted(iivm_df):
    w = np.abs(np.random.default_rng(102).normal(size=len(iivm_df))) + 0.1
    res = dml(
        iivm_df,
        y="y",
        treat="d",
        covariates=["x"],
        model="iivm",
        instrument="z",
        n_folds=3,
        sample_weight=w,
    )
    assert np.isfinite(res.estimate)
    assert res.model_info["diagnostics"]["weighted"] is True


def test_iivm_requires_d_variation():
    rng = np.random.default_rng(103)
    n = 400
    x = rng.normal(size=n)
    z = rng.binomial(1, 0.5, n).astype(float)
    d = np.zeros(n)  # no compliance variation
    y = x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x, "z": z})
    with pytest.raises(ValueError, match="variation in D"):
        dml(df, y="y", treat="d", covariates=["x"], model="iivm", instrument="z")


def test_iivm_requires_z_variation():
    rng = np.random.default_rng(104)
    n = 400
    x = rng.normal(size=n)
    z = np.ones(n)  # constant instrument
    d = rng.binomial(1, 0.5, n).astype(float)
    y = d + x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x, "z": z})
    with pytest.raises(ValueError, match="variation in Z"):
        dml(df, y="y", treat="d", covariates=["x"], model="iivm", instrument="z")


def test_iivm_too_few_per_z_arm():
    rng = np.random.default_rng(105)
    n = 40
    x = rng.normal(size=n)
    z = np.zeros(n)
    z[:3] = 1.0  # only 3 with Z=1
    d = rng.binomial(1, 0.5, n).astype(float)
    y = d + x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x": x, "z": z})
    with pytest.raises(ValueError, match="under each instrument arm"):
        dml(
            df,
            y="y",
            treat="d",
            covariates=["x"],
            model="iivm",
            instrument="z",
            n_folds=5,
        )


# --- subgroup helpers directly ---
def test_fit_predict_subgroup_mean_fallback():
    # Subgroup smaller than _MIN_SUBGROUP_FIT → mean fallback.
    X_sub = np.random.default_rng(0).normal(size=(3, 2))
    y_sub = np.array([1.0, 2.0, 3.0])
    X_te = np.random.default_rng(1).normal(size=(5, 2))
    preds, used_fb = DoubleMLIIVM._fit_predict_subgroup(
        learner=None,
        X_sub=X_sub,
        y_sub=y_sub,
        X_te=X_te,
        fallback_y=y_sub,
        arm_label="test",
    )
    assert used_fb is True
    assert np.allclose(preds, 2.0)


def test_fit_predict_subgroup_weighted_mean_fallback():
    X_sub = np.random.default_rng(0).normal(size=(3, 2))
    y_sub = np.array([1.0, 2.0, 3.0])
    X_te = np.zeros((4, 2))
    w_sub = np.array([1.0, 0.0, 0.0])  # weight only the first
    preds, used_fb = DoubleMLIIVM._fit_predict_subgroup(
        learner=None,
        X_sub=X_sub,
        y_sub=y_sub,
        X_te=X_te,
        fallback_y=y_sub,
        weights_sub=w_sub,
        arm_label="test",
    )
    assert used_fb is True
    assert np.allclose(preds, 1.0)


def test_fit_predict_subgroup_empty_raises():
    from statspai.exceptions import IdentificationFailure

    with pytest.raises(IdentificationFailure):
        DoubleMLIIVM._fit_predict_subgroup(
            learner=None,
            X_sub=np.empty((0, 2)),
            y_sub=np.array([]),
            X_te=np.zeros((3, 2)),
            fallback_y=np.array([]),
            arm_label="empty arm",
        )


def test_fit_predict_classifier_mean_fallback():
    # Small subgroup → fall back to mean(d_sub).
    X_sub = np.zeros((4, 2))
    d_sub = np.array([0.0, 1.0, 1.0, 1.0])
    X_te = np.zeros((6, 2))
    preds, used_fb = DoubleMLIIVM._fit_predict_classifier(
        learner=None,
        X_sub=X_sub,
        d_sub=d_sub,
        X_te=X_te,
        arm_label="test",
    )
    assert used_fb is True
    assert np.allclose(preds, 0.75)


def test_fit_predict_classifier_single_class_fallback():
    # >= min size but only one unique class → fall back to mean.
    X_sub = np.zeros((20, 2))
    d_sub = np.ones(20)  # all 1 → single class
    X_te = np.zeros((5, 2))
    preds, used_fb = DoubleMLIIVM._fit_predict_classifier(
        learner=None,
        X_sub=X_sub,
        d_sub=d_sub,
        X_te=X_te,
        arm_label="t",
    )
    assert used_fb is True
    assert np.allclose(preds, 1.0)


def test_fit_predict_classifier_empty_raises():
    from statspai.exceptions import IdentificationFailure

    with pytest.raises(IdentificationFailure):
        DoubleMLIIVM._fit_predict_classifier(
            learner=None,
            X_sub=np.empty((0, 2)),
            d_sub=np.array([]),
            X_te=np.zeros((3, 2)),
            arm_label="empty",
        )


def test_iivm_m_regressor_predict_path(iivm_df):
    # Pass a *regressor* for ml_m so the `predict` (no predict_proba)
    # branch in the m(X) estimation is exercised. We bypass the
    # classifier coercion by handing an estimator object directly.
    from sklearn.linear_model import LinearRegression

    res = dml(
        iivm_df,
        y="y",
        treat="d",
        covariates=["x"],
        model="iivm",
        instrument="z",
        n_folds=3,
        ml_m=LinearRegression(),
    )
    assert np.isfinite(res.estimate)
