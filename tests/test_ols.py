"""
Tests for OLS regression functionality
"""

import warnings

import pytest
import numpy as np
import pandas as pd
from patsy import dmatrices
from scipy import stats
from statspai import regress
from statspai.core.utils import (
    _try_simple_numeric_design_matrices,
    create_design_matrices,
)
from statspai.exceptions import DataInsufficient
from statspai.exceptions import MethodIncompatibility
from statspai.exceptions import NumericalInstability
from statspai.regression.ols import OLSEstimator, OLSRegression


class TestOLSRegression:
    """Test cases for OLS regression"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n = 100

        # Generate data
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        epsilon = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + epsilon

        df = pd.DataFrame(
            {"y": y, "x1": x1, "x2": x2, "group": np.random.choice(["A", "B", "C"], n)}
        )

        return df

    def test_basic_ols(self, sample_data):
        """Test basic OLS functionality"""
        results = regress("y ~ x1 + x2", data=sample_data)

        # Check that results object is created
        assert results is not None
        assert len(results.params) == 3  # Intercept + 2 variables

        # Check parameter names
        expected_names = ["Intercept", "x1", "x2"]
        assert list(results.params.index) == expected_names

        # Check that coefficients are reasonable (given true values 1, 2, 3)
        assert abs(results.params["Intercept"] - 1) < 0.5
        assert abs(results.params["x1"] - 2) < 0.5
        assert abs(results.params["x2"] - 3) < 0.5

    def test_robust_standard_errors(self, sample_data):
        """Test robust standard errors"""
        results_nonrobust = regress("y ~ x1 + x2", data=sample_data, robust="nonrobust")
        results_hc1 = regress("y ~ x1 + x2", data=sample_data, robust="hc1")

        # Standard errors should be different
        assert not np.allclose(results_nonrobust.std_errors, results_hc1.std_errors)

        # But parameters should be the same
        assert np.allclose(results_nonrobust.params, results_hc1.params)

    def test_robust_option_is_case_insensitive(self, sample_data):
        """Users often write HC1; it should match hc1 exactly."""
        lower = regress("y ~ x1 + x2", data=sample_data, robust="hc1")
        upper = regress("y ~ x1 + x2", data=sample_data, robust="HC1")

        pd.testing.assert_series_equal(lower.params, upper.params)
        pd.testing.assert_series_equal(lower.std_errors, upper.std_errors)

    @pytest.mark.parametrize(
        "weights, match",
        [
            (np.ones(4), "weights length"),
            (np.array([1.0, 1.0, np.nan, 1.0, 1.0]), "finite"),
            (np.array([1.0, 1.0, 0.0, 1.0, 1.0]), "strictly positive"),
        ],
    )
    def test_estimator_rejects_invalid_direct_weights(self, weights, match):
        """Direct OLSEstimator callers get the same weight guards as regress()."""
        y = np.arange(5.0)
        X = np.column_stack([np.ones(5), np.arange(5.0)])

        with pytest.raises(ValueError, match=match):
            OLSEstimator().estimate(y, X, weights=weights)

    def test_estimator_rejects_misaligned_direct_arrays(self):
        y = np.arange(5.0)
        X = np.column_stack([np.ones(4), np.arange(4.0)])

        with pytest.raises(ValueError, match="y has 5 rows but X has 4 rows"):
            OLSEstimator().estimate(y, X)

    def test_model_rejects_var_name_count_mismatch(self):
        y = np.arange(5.0)
        X = np.column_stack([np.ones(5), np.arange(5.0)])
        model = OLSRegression(y=y, X=X, var_names=["only_one"])

        with pytest.raises(ValueError, match="var_names has 1 entries"):
            model.fit()

    def test_regress_rejects_nonfinite_design_values(self, sample_data):
        df = sample_data.copy()
        df.loc[df.index[3], "x1"] = np.inf

        with pytest.raises(ValueError, match="X contains non-finite values"):
            regress("y ~ x1 + x2", data=df)

    def test_regress_rejects_perfectly_collinear_regressors(self, sample_data):
        df = sample_data.copy()
        df["x1_twice"] = 2.0 * df["x1"]

        with pytest.raises(NumericalInstability, match="perfectly collinear"):
            regress("y ~ x1 + x1_twice + x2", data=df)

    def test_regress_rejects_low_order_linear_dependence(self):
        rng = np.random.default_rng(20260617)
        x1 = rng.normal(size=80)
        x2 = rng.normal(size=80)
        df = pd.DataFrame(
            {
                "y": 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.1, size=80),
                "x1": x1,
                "x2": x2,
                "x_sum": x1 + x2,
            }
        )

        with pytest.raises(NumericalInstability, match="exact linear combination"):
            regress("y ~ x1 + x2 + x_sum", data=df)

    def test_regress_rejects_no_residual_degrees_of_freedom(self):
        df = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})

        with pytest.raises(DataInsufficient, match="residual df=0"):
            regress("y ~ x", data=df)

    def test_exact_fit_diagnostics_do_not_emit_runtime_warnings(self):
        df = pd.DataFrame(
            {
                "y": [1.0, 3.0, 5.0, 7.0],
                "x": [0.0, 1.0, 2.0, 3.0],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = regress("y ~ x", data=df)

        assert np.isinf(result.diagnostics["Log-Likelihood"])
        assert np.isneginf(result.diagnostics["AIC"])
        assert np.isneginf(result.diagnostics["BIC"])

    def test_constant_outcome_diagnostics_do_not_emit_runtime_warnings(self):
        df = pd.DataFrame(
            {
                "y": np.ones(20),
                "x": np.linspace(-1.0, 1.0, 20),
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = regress("y ~ x", data=df)

        assert np.isnan(result.diagnostics["R-squared"])
        assert np.isnan(result.diagnostics["Adj. R-squared"])
        assert np.isnan(result.diagnostics["F-statistic"])

    def test_weighted_regress_aligns_after_formula_dropna(self, sample_data):
        """Weight columns are aligned to the Patsy/fast-path estimation sample."""
        df = sample_data.copy()
        df["w"] = np.linspace(0.5, 2.0, len(df))
        df.loc[df.index[[1, 7]], "x1"] = np.nan
        df.loc[df.index[[4]], "y"] = np.nan

        with_missing = regress("y ~ x1 + x2", data=df, weights="w")
        complete = df.dropna(subset=["y", "x1", "x2"])
        explicit_complete = regress("y ~ x1 + x2", data=complete, weights="w")

        pd.testing.assert_series_equal(with_missing.params, explicit_complete.params)
        pd.testing.assert_series_equal(
            with_missing.std_errors, explicit_complete.std_errors
        )
        assert with_missing.data_info["nobs"] == len(complete)

    def test_cluster_standard_errors(self, sample_data):
        """Test clustered standard errors"""
        results = regress("y ~ x1 + x2", data=sample_data, cluster="group")

        # Should run without error
        assert results is not None
        assert len(results.params) == 3

    def test_cluster_standard_errors_align_after_formula_dropna(self, sample_data):
        """Cluster labels follow the Patsy/fast-path estimation sample."""
        df = sample_data.copy()
        df.loc[df.index[[1, 7]], "x1"] = np.nan
        df.loc[df.index[[4]], "y"] = np.nan

        with_missing = regress("y ~ x1 + x2", data=df, cluster="group")
        complete = df.dropna(subset=["y", "x1", "x2"])
        explicit_complete = regress("y ~ x1 + x2", data=complete, cluster="group")

        pd.testing.assert_series_equal(with_missing.params, explicit_complete.params)
        pd.testing.assert_series_equal(
            with_missing.std_errors, explicit_complete.std_errors
        )
        assert with_missing.data_info["nobs"] == len(complete)

    def test_cluster_standard_errors_single_cluster_raises(self, sample_data):
        """Cluster-robust OLS fails loudly when the design has one cluster."""
        df = sample_data.copy()
        df["one_cluster"] = "only"

        with pytest.raises(DataInsufficient, match="at least two clusters"):
            regress("y ~ x1 + x2", data=df, cluster="one_cluster")

    def test_summary_output(self, sample_data):
        """Test summary output generation"""
        results = regress("y ~ x1 + x2", data=sample_data)
        summary = results.summary()

        # Check that summary is a string and contains expected elements
        assert isinstance(summary, str)
        assert "OLS" in summary
        assert "Coefficient" in summary
        assert "Std. Error" in summary
        assert "R-squared" in summary

    def test_model_class_interface(self, sample_data):
        """Test the model class interface"""
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        results = model.fit()

        assert model.is_fitted
        assert results is not None

        # Test prediction on training data
        predictions = model.predict()
        assert len(predictions) == len(sample_data)

    def test_predict_requires_fitted_model(self, sample_data):
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)

        with pytest.raises(MethodIncompatibility, match="fitted before prediction"):
            model.predict()

    def test_predict_out_of_sample_requires_formula_fit(self, sample_data):
        y = sample_data["y"].to_numpy()
        X = np.column_stack(
            [
                np.ones(len(sample_data)),
                sample_data["x1"].to_numpy(),
                sample_data["x2"].to_numpy(),
            ]
        )
        model = OLSRegression(y=y, X=X, var_names=["Intercept", "x1", "x2"])
        model.fit()

        with pytest.raises(MethodIncompatibility, match="fit with a formula"):
            model.predict(sample_data.head(2))

    def test_predict_out_of_sample_requires_fitted_variable_names(self, sample_data):
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        model.fit()
        model.var_names = None

        with pytest.raises(ValueError, match="variable names are unavailable"):
            model.predict(sample_data.head(2))

    def test_predict_confidence_uses_full_covariance_matrix(self):
        rng = np.random.default_rng(20260618)
        n = 80
        x1 = rng.normal(size=n)
        x2 = 0.85 * x1 + rng.normal(scale=0.35, size=n)
        y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        model = OLSRegression(formula="y ~ x1 + x2", data=df)
        result = model.fit()
        new_data = pd.DataFrame({"x1": [1.25], "x2": [-0.75]})

        pred = model.predict(new_data, what="confidence")
        x_new = np.array([[1.0, 1.25, -0.75]])
        cov = result.data_info["var_cov"]
        se_mean = float(np.sqrt(x_new @ cov @ x_new.T)[0, 0])
        t_crit = stats.t.ppf(0.975, result.data_info["df_resid"])
        expected_lower = float(pred["yhat"].iloc[0] - t_crit * se_mean)
        diag_cov = np.diag(result.std_errors.to_numpy() ** 2)
        diag_se = float(np.sqrt(x_new @ diag_cov @ x_new.T)[0, 0])
        diag_lower = float(pred["yhat"].iloc[0] - t_crit * diag_se)

        assert pred["lower"].iloc[0] == pytest.approx(expected_lower)
        assert abs(pred["lower"].iloc[0] - diag_lower) > 0.05

    def test_predict_rejects_malformed_covariance_matrix(self, sample_data):
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        result = model.fit()
        result.data_info["var_cov"] = np.eye(2)

        with pytest.raises(MethodIncompatibility, match="covariance matrix shape"):
            model.predict(sample_data.head(2), what="confidence")

    @pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5, np.nan, "wide"])
    def test_predict_rejects_invalid_interval_alpha(self, sample_data, alpha):
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        model.fit()

        with pytest.raises(MethodIncompatibility, match="alpha"):
            model.predict(sample_data.head(2), what="confidence", alpha=alpha)

    @pytest.mark.parametrize("what", ["median", "interval", ["confidence"]])
    def test_predict_rejects_invalid_interval_type(self, sample_data, what):
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        model.fit()

        with pytest.raises(MethodIncompatibility, match="what"):
            model.predict(sample_data.head(2), what=what)

    def test_predict_mean_return_df_in_sample(self, sample_data):
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        result = model.fit()

        pred = model.predict(what="mean", return_df=True, alpha=np.nan)

        assert list(pred.columns) == ["yhat"]
        np.testing.assert_allclose(pred["yhat"].to_numpy(), result.fitted_values())

    def test_predict_mean_return_df_out_of_sample(self, sample_data):
        model = OLSRegression(formula="y ~ x1 + x2", data=sample_data)
        result = model.fit()
        new_data = sample_data.head(3).copy()

        pred = model.predict(new_data, what="mean", return_df=True, alpha=np.nan)
        expected = (
            result.params["Intercept"]
            + result.params["x1"] * new_data["x1"].to_numpy()
            + result.params["x2"] * new_data["x2"].to_numpy()
        )

        assert list(pred.columns) == ["yhat"]
        np.testing.assert_allclose(pred["yhat"].to_numpy(), expected)

    def test_predict_reuses_training_design_for_categorical_formula(self):
        train = pd.DataFrame(
            {
                "y": [1.0, 3.0, 4.0, 6.0, 5.0, 8.0],
                "x": [0.0, 1.0, 0.0, 1.0, 2.0, 2.0],
                "g": ["a", "a", "b", "b", "a", "b"],
            }
        )
        model = OLSRegression(formula="y ~ x + C(g)", data=train)
        result = model.fit()

        new_data = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "g": ["a", "a", "b", "b"]})
        pred = model.predict(new_data, what="mean", return_df=True)
        expected = (
            result.params["Intercept"]
            + result.params["x"] * new_data["x"].to_numpy()
            + result.params["C(g)[T.b]"] * (new_data["g"].to_numpy() == "b")
        )

        assert list(pred.columns) == ["yhat"]
        np.testing.assert_allclose(pred["yhat"].to_numpy(), expected)

    def test_predict_rejects_unseen_categorical_levels(self):
        train = pd.DataFrame(
            {
                "y": [1.0, 3.0, 4.0, 6.0, 5.0, 8.0],
                "x": [0.0, 1.0, 0.0, 1.0, 2.0, 2.0],
                "g": ["a", "a", "b", "b", "a", "b"],
            }
        )
        model = OLSRegression(formula="y ~ x + C(g)", data=train)
        model.fit()
        new_data = pd.DataFrame({"x": [0.0], "g": ["c"]})

        with pytest.raises(MethodIncompatibility, match="prediction design matrix"):
            model.predict(new_data, what="mean", return_df=True)

    def test_no_constant(self, sample_data):
        """Test regression without constant term"""
        results = regress("y ~ x1 + x2 - 1", data=sample_data)

        # Should have only 2 parameters (no intercept)
        assert len(results.params) == 2
        assert "Intercept" not in results.params.index

    def test_simple_numeric_design_fast_path_matches_patsy(self):
        """Plain numeric formulas use the direct builder without changing semantics."""
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, np.nan, 4.0],
                "x1": [0.5, 1.0, 1.5, 2.0],
                "x2": [3.0, np.nan, 5.0, 6.0],
            },
            index=[10, 11, 12, 13],
        )
        formula = "y ~ x1 + x2"

        assert _try_simple_numeric_design_matrices(formula, df) is not None
        y_fast, X_fast = create_design_matrices(formula, df)
        y_patsy, X_patsy = dmatrices(formula, df, return_type="dataframe")

        pd.testing.assert_frame_equal(y_fast, y_patsy)
        pd.testing.assert_frame_equal(X_fast, X_patsy)

    def test_simple_numeric_no_intercept_fast_path_matches_patsy(self):
        """The direct builder preserves Patsy's no-intercept column contract."""
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0],
                "x1": [0.5, 1.0, 1.5],
                "x2": [3.0, 4.0, 5.0],
            }
        )
        formula = "y ~ x1 + x2 - 1"

        assert _try_simple_numeric_design_matrices(formula, df) is not None
        y_fast, X_fast = create_design_matrices(formula, df)
        y_patsy, X_patsy = dmatrices(formula, df, return_type="dataframe")

        pd.testing.assert_frame_equal(y_fast, y_patsy)
        pd.testing.assert_frame_equal(X_fast, X_patsy)

    def test_complex_formula_stays_on_patsy_path(self, sample_data):
        """Categorical/transformed formulas are left to Patsy."""
        assert (
            _try_simple_numeric_design_matrices("y ~ x1 + C(group)", sample_data)
            is None
        )

    def test_confidence_intervals(self, sample_data):
        """Test confidence interval calculation"""
        results = regress("y ~ x1 + x2", data=sample_data)
        conf_int = results.conf_int()

        # Check structure
        assert isinstance(conf_int, pd.DataFrame)
        assert conf_int.shape[0] == len(results.params)
        assert conf_int.shape[1] == 2  # Lower and upper bounds

        # Check that confidence intervals make sense
        for param in results.params.index:
            lower = conf_int.loc[param].iloc[0]
            upper = conf_int.loc[param].iloc[1]
            estimate = results.params[param]

            assert lower < estimate < upper


if __name__ == "__main__":
    pytest.main([__file__])
