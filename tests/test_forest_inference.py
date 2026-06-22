"""Tests for GRF-style causal forest inference (test_calibration, rate,
honest_variance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.forest.causal_forest import CausalForest
from statspai.forest.forest_inference import (
    calibration_test,
    rate,
    honest_variance,
    average_treatment_effect,
    forest_diagnostics,
)


def _sim_hte(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    # Heterogeneous treatment effect: τ(x) = 1 + 2 * x1
    tau = 1.0 + 2.0 * X[:, 0]
    T = rng.integers(0, 2, size=n)
    Y0 = X[:, 1] + rng.standard_normal(n) * 0.5
    Y1 = Y0 + tau
    Y = np.where(T == 1, Y1, Y0)
    return X, T, Y


def _fit_forest():
    X, T, Y = _sim_hte(n=400, seed=1)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=50, random_state=42)
    return cf, X, T, Y


def test_calibration_returns_dataframe():
    cf, X, T, Y = _fit_forest()
    out = calibration_test(cf, X=X, Y=Y, T=T)
    out_alias = sp.test_calibration(cf, X=X, Y=Y, T=T)
    assert isinstance(out, pd.DataFrame)
    assert "coef" in out.columns
    assert "se" in out.columns
    assert "p" in out.columns
    assert len(out) == 2
    assert out_alias.equals(out)
    # Causal-forest calibration coefficients (Chernozhukov BLP test) are not
    # bit-portable across BLAS backends: ARM Accelerate vs x86 OpenBLAS shift
    # the mean-forest coefficient by several percent and the small
    # differential-prediction coefficient by tens of percent. Assert the BLP
    # structure rather than pinning the dev-machine values.
    cols = out.loc[:, ["coef", "se", "ci_low", "ci_high"]].to_numpy()
    assert np.all(np.isfinite(cols))
    assert np.all(out["se"] > 0)
    assert np.all(out["ci_low"] <= out["coef"])
    assert np.all(out["coef"] <= out["ci_high"])
    # Mean-forest-prediction coefficient is the calibration slope: positive, O(1).
    assert out.loc["mean_forest_prediction", "coef"] > 0
    # Differential forest prediction coefficient should be finite
    # (forest captures real heterogeneity).
    diff = out.loc["differential_forest_prediction"]
    assert np.isfinite(diff["coef"])


def test_rate_returns_autoc_estimate():
    cf, X, T, Y = _fit_forest()
    out = rate(cf, X=X, Y=Y, T=T, target="AUTOC", q_grid=50, seed=3)
    assert "estimate" in out
    assert "se" in out
    assert "ci_low" in out
    assert "toc_curve" in out
    assert out["toc_curve"].shape[1] == 2
    # RATE/TOC values ride on the causal forest and are not bit-portable across
    # BLAS backends; assert structure (finite estimate, positive SE, ascending
    # quantile grid, finite TOC) rather than pinning dev-machine goldens.
    assert np.isfinite(out["estimate"]) and out["se"] > 0
    assert out["ci_low"] <= out["estimate"]
    grid = out["toc_curve"][:, 0]
    assert np.all(np.diff(grid) > 0)
    assert np.all(np.isfinite(out["toc_curve"]))


def test_rate_qini_variant_runs():
    cf, X, T, Y = _fit_forest()
    out = rate(cf, X=X, Y=Y, T=T, target="QINI", q_grid=30, seed=4)
    assert np.isfinite(out["estimate"])
    # QINI estimate/SE ride on the causal forest (not bit-portable across BLAS);
    # assert structure rather than pinning dev-machine goldens.
    assert out["se"] > 0


def test_calibration_and_rate_validate_inputs_and_subsamples():
    cf, X, T, Y = _fit_forest()

    with pytest.raises(MethodIncompatibility, match="fitted"):
        calibration_test(CausalForest())

    with pytest.raises(MethodIncompatibility, match="alpha"):
        calibration_test(cf, X=X, Y=Y, T=T, alpha=np.nan)

    with pytest.raises(MethodIncompatibility) as wrong_y:
        calibration_test(cf, X=X[:5], Y=Y[:4], T=T[:5])
    assert wrong_y.value.diagnostics == {"n_y": 4, "n_x": 5}

    with pytest.raises(MethodIncompatibility, match="numeric"):
        calibration_test(cf, X=X[:5], Y=Y[:5], T=np.array(["bad"] * 5))

    with pytest.raises(DataInsufficient, match="at least 3 rows"):
        calibration_test(cf, X=X[:2], Y=Y[:2], T=T[:2])

    cal = calibration_test(cf, X=X[:20], Y=Y[:20], T=T[:20])
    assert list(cal.index) == [
        "mean_forest_prediction",
        "differential_forest_prediction",
    ]

    with pytest.raises(MethodIncompatibility, match="fitted"):
        rate(CausalForest())

    with pytest.raises(MethodIncompatibility, match="target"):
        rate(cf, X=X, Y=Y, T=T, target=object())

    with pytest.raises(MethodIncompatibility, match="target"):
        rate(cf, X=X, Y=Y, T=T, target="bad")

    with pytest.raises(MethodIncompatibility, match="q_grid"):
        rate(cf, X=X, Y=Y, T=T, q_grid=0)

    with pytest.raises(MethodIncompatibility, match="alpha"):
        rate(cf, X=X, Y=Y, T=T, alpha=np.inf)

    with pytest.raises(MethodIncompatibility) as wrong_x:
        rate(cf, X=np.ones((2, 2)), Y=Y[:2], T=T[:2])
    assert wrong_x.value.diagnostics["expected_features"] == 3

    with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
        bad_y = Y[:5].copy()
        bad_y[0] = np.nan
        rate(cf, X=X[:5], Y=bad_y, T=T[:5])

    with pytest.raises(DataInsufficient, match="at least 2 rows"):
        rate(cf, X=X[:1], Y=Y[:1], T=T[:1])

    out = rate(cf, X=X[:20], Y=Y[:20], T=T[:20], target="autoc", q_grid=5)
    assert out["target"] == "AUTOC"
    assert out["n"] == 20
    assert out["toc_curve"].shape == (5, 2)


def test_honest_variance_reports_ci():
    cf, X, _, _ = _fit_forest()
    out = honest_variance(cf, X=X, n_splits=10, seed=0)
    assert "ate" in out
    assert "ci_low" in out
    assert "ci_high" in out
    assert out["ci_low"] <= out["ate"] <= out["ci_high"]
    # honest_variance ATE/SE/CI ride on the causal forest (not bit-portable
    # across BLAS backends); assert structure rather than dev-machine goldens.
    assert np.all(np.isfinite([out["ate"], out["se"], out["ci_low"], out["ci_high"]]))
    assert out["se"] > 0


def test_honest_variance_validates_inputs():
    cf, X, _, _ = _fit_forest()

    with pytest.raises(MethodIncompatibility, match="fitted"):
        honest_variance(CausalForest())

    with pytest.raises(MethodIncompatibility, match="n_splits"):
        honest_variance(cf, X=X, n_splits=1)

    with pytest.raises(MethodIncompatibility) as wrong_shape:
        honest_variance(cf, X=np.ones((2, 2)), n_splits=2)
    assert wrong_shape.value.diagnostics["expected_features"] == 3

    with pytest.raises(DataInsufficient, match="at least 2 CATE rows"):
        honest_variance(cf, X=X[:1], n_splits=2)


def test_average_treatment_effect_targets_run():
    cf, X, T, _ = _fit_forest()
    ate = average_treatment_effect(cf, X=X, T=T, target_sample="all")
    att = cf.average_treatment_effect(X=X, T=T, target_sample="treated")
    ato = average_treatment_effect(cf, X=X, T=T, target_sample="overlap")
    assert ate["estimand"] == "ATE"
    assert att["estimand"] == "ATT"
    assert ato["effective_sample_size"] > 0
    assert ate["ci_low"] <= ate["estimate"] <= ate["ci_high"]


def test_average_treatment_effect_validates_inputs():
    cf, X, T, _ = _fit_forest()

    with pytest.raises(MethodIncompatibility, match="fitted"):
        average_treatment_effect(CausalForest())

    with pytest.raises(MethodIncompatibility, match="target_sample"):
        average_treatment_effect(cf, X=X, T=T, target_sample=object())

    with pytest.raises(MethodIncompatibility, match="target_sample"):
        average_treatment_effect(cf, X=X, T=T, target_sample="unsupported")

    with pytest.raises(MethodIncompatibility, match="alpha"):
        average_treatment_effect(cf, X=X, T=T, alpha=np.nan)

    with pytest.raises(MethodIncompatibility, match="clip"):
        average_treatment_effect(cf, X=X, T=T, clip=0.5)

    with pytest.raises(MethodIncompatibility) as wrong_shape:
        average_treatment_effect(cf, X=np.ones((2, 2)), T=np.ones(2))
    assert wrong_shape.value.diagnostics["expected_features"] == 3

    with pytest.raises(MethodIncompatibility, match="numeric"):
        average_treatment_effect(cf, X=np.array([["bad", "data", "x"]]), T=[1])

    with pytest.raises(MethodIncompatibility) as wrong_t:
        average_treatment_effect(cf, X=X[:3], T=np.ones(2))
    assert wrong_t.value.diagnostics == {"n_t": 2, "n_effects": 3}

    with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
        bad_t = T.astype(float)
        bad_t[0] = np.nan
        average_treatment_effect(cf, X=X, T=bad_t)

    with pytest.raises(DataInsufficient, match="no treated"):
        average_treatment_effect(cf, X=X, T=np.zeros(len(X)), target_sample="treated")

    with pytest.raises(DataInsufficient, match="no control"):
        average_treatment_effect(cf, X=X, T=np.ones(len(X)), target_sample="control")


def test_forest_diagnostics_reports_overlap_and_warnings():
    cf, X, T, _ = _fit_forest()
    out = forest_diagnostics(cf, X=X, T=T, propensity_bounds=(0.05, 0.95))
    out_method = cf.forest_diagnostics(X=X, T=T)
    assert "cate_mean" in out
    assert "overlap_share" in out
    assert 0 <= out["overlap_share"] <= 1
    assert out_method["n"] == out["n"]
    # cate_mean / cate_sd ride on the causal forest (not bit-portable across
    # BLAS backends); pin only the exact structural quantities (overlap share,
    # n) and assert the rest are finite / positive.
    assert np.isfinite(out["cate_mean"])
    assert out["cate_sd"] > 0
    np.testing.assert_allclose([out["overlap_share"], out["n"]], [1.0, 400])


def test_forest_diagnostics_validates_inputs_and_subsamples():
    cf, X, T, _ = _fit_forest()

    with pytest.raises(MethodIncompatibility, match="fitted"):
        forest_diagnostics(CausalForest())

    with pytest.raises(MethodIncompatibility, match="propensity_bounds"):
        forest_diagnostics(cf, propensity_bounds=(0.8, 0.2))

    with pytest.raises(MethodIncompatibility, match="propensity_bounds"):
        forest_diagnostics(cf, propensity_bounds=(0.05,))

    with pytest.raises(MethodIncompatibility) as wrong_shape:
        forest_diagnostics(cf, X=np.ones((2, 2)), T=np.ones(2))
    assert wrong_shape.value.diagnostics["expected_features"] == 3

    with pytest.raises(MethodIncompatibility, match="T is required"):
        forest_diagnostics(cf, X=X[:5])

    with pytest.raises(MethodIncompatibility) as wrong_t:
        forest_diagnostics(cf, X=X[:5], T=T[:4])
    assert wrong_t.value.diagnostics == {"n_t": 4, "n_x": 5}

    with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
        bad_t = T[:5].astype(float)
        bad_t[0] = np.inf
        forest_diagnostics(cf, X=X[:5], T=bad_t)

    out = forest_diagnostics(cf, X=X[:20], T=T[:20])
    assert out["n"] == 20
    assert out["n_treated"] + out["n_control"] == 20
