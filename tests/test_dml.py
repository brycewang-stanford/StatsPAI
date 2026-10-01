"""
Tests for Double/Debiased Machine Learning module.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.core.results import CausalResult
from statspai.dml import DoubleML, dml


@pytest.fixture
def plr_data():
    """
    Partially linear DGP:
        Y = 2.0*D + sin(X1) + X2^2 + eps
        D = cos(X1) + X2 + v

    True ATE = 2.0. Confounding through X1, X2.
    """
    rng = np.random.default_rng(42)
    n = 1500

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    v = rng.normal(0, 0.5, n)
    eps = rng.normal(0, 0.5, n)

    D = np.cos(X1) + X2 + v
    Y = 2.0 * D + np.sin(X1) + X2**2 + eps

    return pd.DataFrame({"y": Y, "d": D, "x1": X1, "x2": X2})


@pytest.fixture
def irm_data():
    """
    Binary treatment DGP for IRM:
        P(D=1|X) = logistic(0.5*X1 + X2)
        Y = 3.0*D + X1 + X2^2 + eps

    True ATE = 3.0.
    """
    rng = np.random.default_rng(42)
    n = 2000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)

    logit = 0.5 * X1 + X2
    prob = 1 / (1 + np.exp(-logit))
    D = rng.binomial(1, prob, n).astype(float)

    Y = 3.0 * D + X1 + X2**2 + eps

    return pd.DataFrame({"y": Y, "d": D, "x1": X1, "x2": X2})


class TestDMLPartiallyLinear:

    def test_plr_basic(self, plr_data):
        """PLR should recover ATE ≈ 2.0"""
        result = dml(plr_data, y="y", treat="d", covariates=["x1", "x2"], model="plr")

        assert isinstance(result, CausalResult)
        assert (
            abs(result.estimate - 2.0) < 0.5
        ), f"DML estimate = {result.estimate:.2f}, expected ≈ 2.0"

    def test_plr_significance(self, plr_data):
        """Effect should be significant."""
        result = dml(plr_data, y="y", treat="d", covariates=["x1", "x2"])
        assert result.pvalue < 0.05

    def test_plr_ci(self):
        """CI should contain true value (larger sample for stability)."""
        rng = np.random.default_rng(100)
        n = 3000
        X1 = rng.normal(0, 1, n)
        X2 = rng.normal(0, 1, n)
        v = rng.normal(0, 0.5, n)
        eps = rng.normal(0, 0.5, n)
        D = np.cos(X1) + X2 + v
        Y = 2.0 * D + np.sin(X1) + X2**2 + eps
        df = pd.DataFrame({"y": Y, "d": D, "x1": X1, "x2": X2})

        result = dml(df, y="y", treat="d", covariates=["x1", "x2"], alpha=0.05)
        assert result.ci[0] < 2.0 < result.ci[1]

    def test_plr_custom_ml(self, plr_data):
        """Custom sklearn model should work."""
        from sklearn.ensemble import RandomForestRegressor

        ml = RandomForestRegressor(n_estimators=50, random_state=42)

        result = dml(
            plr_data, y="y", treat="d", covariates=["x1", "x2"], ml_g=ml, ml_m=ml
        )
        assert abs(result.estimate - 2.0) < 1.0

    def test_plr_multiple_reps(self, plr_data):
        """Multiple repetitions should work (median aggregation)."""
        result = dml(plr_data, y="y", treat="d", covariates=["x1", "x2"], n_rep=3)
        assert abs(result.estimate - 2.0) < 0.5
        assert result.model_info["n_rep"] == 3


class TestDMLInteractive:

    def test_irm_basic(self, irm_data):
        """IRM should recover ATE ≈ 3.0"""
        result = dml(irm_data, y="y", treat="d", covariates=["x1", "x2"], model="irm")

        assert isinstance(result, CausalResult)
        assert (
            abs(result.estimate - 3.0) < 1.0
        ), f"IRM estimate = {result.estimate:.2f}, expected ≈ 3.0"

    def test_irm_method_label(self, irm_data):
        """Should be labeled as IRM."""
        result = dml(irm_data, y="y", treat="d", covariates=["x1", "x2"], model="irm")
        assert "IRM" in result.method


class TestDMLGeneral:

    def test_model_info(self, plr_data):
        result = dml(plr_data, y="y", treat="d", covariates=["x1", "x2"])

        info = result.model_info
        assert "dml_model" in info
        assert info["dml_model"] == "PLR"
        assert "ml_g" in info
        assert "n_folds" in info

    def test_summary(self, plr_data):
        result = dml(plr_data, y="y", treat="d", covariates=["x1", "x2"])
        s = result.summary()
        assert "Double ML" in s

    def test_citation(self, plr_data):
        result = dml(plr_data, y="y", treat="d", covariates=["x1", "x2"])
        assert "chernozhukov" in result.cite().lower()

    def test_missing_column(self, plr_data):
        with pytest.raises(ValueError, match="not found"):
            dml(plr_data, y="nonexistent", treat="d", covariates=["x1"])

    def test_invalid_model(self, plr_data):
        with pytest.raises(ValueError, match="model must be"):
            dml(plr_data, y="y", treat="d", covariates=["x1"], model="invalid")

    def test_invalid_nfolds(self, plr_data):
        with pytest.raises(ValueError, match="n_folds"):
            dml(plr_data, y="y", treat="d", covariates=["x1"], n_folds=1)

    def test_default_plr_learners_carry_a_coverage_note(self, plr_data):
        """The default PLR learners have below-nominal coverage evidence.

        The fit says so in ``model_info`` and ``summary()``; a supplied
        learner, and the IRM model, carry no such note.
        """
        from sklearn.linear_model import LinearRegression

        default = dml(plr_data, y="y", treat="d", covariates=["x1", "x2"])
        assert default.model_info["default_learners"] is True
        assert "0.88" in default.model_info["learner_note"]
        assert "Learner Note" in default.summary()

        supplied = dml(
            plr_data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            ml_g=LinearRegression(),
            ml_m=LinearRegression(),
        )
        assert "learner_note" not in supplied.model_info
        # The note changes no number.
        assert (
            default.estimate
            == dml(plr_data, y="y", treat="d", covariates=["x1", "x2"]).estimate
        )

    def test_default_irm_learners_carry_no_note(self):
        rng = np.random.default_rng(0)
        n = 400
        x = rng.normal(size=(n, 2))
        d = (rng.uniform(size=n) < 1 / (1 + np.exp(-x[:, 0]))).astype(int)
        y = d + x[:, 0] + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x1": x[:, 0], "x2": x[:, 1]})
        fit = dml(df, y="y", treat="d", covariates=["x1", "x2"], model="irm")
        assert fit.model_info["default_learners"] is True
        assert "learner_note" not in fit.model_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
