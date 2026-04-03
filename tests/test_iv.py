"""
Tests for IV (2SLS) regression functionality

Uses a known DGP with endogeneity so we can verify:
1. 2SLS corrects the bias from OLS
2. Standard errors are computed correctly
3. Diagnostics (first-stage F, Sargan, Hausman) behave as expected
"""

import pytest
import numpy as np
import pandas as pd
from statspai import ivreg, regress
from statspai.regression.iv import IVRegression


class TestIVRegression:
    """Test cases for IV/2SLS regression"""

    @pytest.fixture
    def endogenous_data(self):
        """
        DGP with endogeneity:

            y = 1 + 2*x_endog + 3*x_exog + eps
            x_endog = 0.5*z1 + 0.8*z2 + 0.6*eps + eta

        x_endog is endogenous because it depends on eps.
        z1, z2 are valid instruments (relevant + exogenous).
        """
        np.random.seed(42)
        n = 2000

        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)
        x_exog = np.random.normal(0, 1, n)
        eps = np.random.normal(0, 1, n)
        eta = np.random.normal(0, 0.5, n)

        # Endogenous regressor (correlated with eps)
        x_endog = 0.5 * z1 + 0.8 * z2 + 0.6 * eps + eta

        # Outcome
        y = 1 + 2 * x_endog + 3 * x_exog + eps

        df = pd.DataFrame({
            'y': y,
            'x_endog': x_endog,
            'x_exog': x_exog,
            'z1': z1,
            'z2': z2,
            'group': np.random.choice(['A', 'B', 'C', 'D'], n),
        })
        return df

    @pytest.fixture
    def just_identified_data(self):
        """
        DGP with exactly one instrument for one endogenous variable.
        """
        np.random.seed(123)
        n = 1000

        z = np.random.normal(0, 1, n)
        eps = np.random.normal(0, 1, n)
        x_endog = 0.7 * z + 0.5 * eps + np.random.normal(0, 0.3, n)
        y = 3 + 1.5 * x_endog + eps

        return pd.DataFrame({
            'y': y,
            'x_endog': x_endog,
            'z': z,
        })

    # ----------------------------------------------------------------
    # Basic functionality
    # ----------------------------------------------------------------

    def test_basic_iv(self, endogenous_data):
        """2SLS should recover true coefficients (beta_endog ≈ 2)"""
        result = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )

        assert result is not None
        assert len(result.params) == 3  # Intercept, x_exog, x_endog

        # 2SLS should be close to true values
        assert abs(result.params['x_endog'] - 2.0) < 0.3
        assert abs(result.params['x_exog'] - 3.0) < 0.3
        assert abs(result.params['Intercept'] - 1.0) < 0.5

    def test_iv_corrects_ols_bias(self, endogenous_data):
        """OLS is biased upward; 2SLS should give lower x_endog coefficient"""
        ols_result = regress("y ~ x_endog + x_exog", data=endogenous_data)
        iv_result = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )

        # OLS overestimates because x_endog is positively correlated with eps
        ols_endog = ols_result.params['x_endog']
        iv_endog = iv_result.params['x_endog']

        # OLS should be biased upward (> 2.0)
        assert ols_endog > 2.2, f"OLS should be biased: {ols_endog:.3f}"
        # IV should be closer to true value of 2.0
        assert abs(iv_endog - 2.0) < abs(ols_endog - 2.0), (
            f"IV ({iv_endog:.3f}) should be closer to 2.0 than OLS ({ols_endog:.3f})"
        )

    def test_just_identified(self, just_identified_data):
        """Just-identified case: 1 instrument for 1 endogenous variable"""
        result = ivreg(
            "y ~ (x_endog ~ z)",
            data=just_identified_data,
        )

        assert result is not None
        assert abs(result.params['x_endog'] - 1.5) < 0.5

    # ----------------------------------------------------------------
    # Standard errors
    # ----------------------------------------------------------------

    def test_robust_standard_errors(self, endogenous_data):
        """Robust SEs should differ from classical but params stay the same"""
        result_classical = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
            robust='nonrobust',
        )
        result_hc1 = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
            robust='hc1',
        )

        # Parameters identical
        np.testing.assert_allclose(
            result_classical.params.values,
            result_hc1.params.values,
        )
        # Standard errors differ
        assert not np.allclose(
            result_classical.std_errors.values,
            result_hc1.std_errors.values,
        )

    def test_cluster_standard_errors(self, endogenous_data):
        """Clustered SEs should run without error"""
        result = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
            cluster='group',
        )
        assert result is not None
        assert len(result.params) == 3

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def test_first_stage_f_statistic(self, endogenous_data):
        """First-stage F should be large (strong instruments)"""
        model = IVRegression(
            formula="y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )
        model.fit()

        fs = model.first_stage
        assert len(fs) == 1  # One endogenous variable
        assert fs[0]['f_statistic'] > 10, (
            f"First-stage F = {fs[0]['f_statistic']:.1f}, should be > 10"
        )

    def test_sargan_test_overidentified(self, endogenous_data):
        """Sargan test should be available when over-identified"""
        model = IVRegression(
            formula="y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )
        model.fit()

        sargan = model.sargan_test
        assert sargan is not None
        assert sargan['df'] == 1  # 2 instruments - 1 endogenous = 1
        # With valid instruments, p-value should not reject
        assert sargan['pvalue'] > 0.01

    def test_sargan_not_available_just_identified(self, just_identified_data):
        """Sargan test unavailable when just-identified"""
        model = IVRegression(
            formula="y ~ (x_endog ~ z)",
            data=just_identified_data,
        )
        model.fit()

        assert model.sargan_test is None

    def test_hausman_test(self, endogenous_data):
        """Hausman test should reject exogeneity (because x_endog IS endogenous)"""
        model = IVRegression(
            formula="y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )
        model.fit()

        hausman = model.hausman_test
        assert hausman is not None
        # Should reject H0: x_endog is exogenous
        assert hausman['pvalue'] < 0.05, (
            f"Hausman p-value = {hausman['pvalue']:.4f}, "
            f"should reject exogeneity"
        )

    def test_weak_instrument_warning(self):
        """Should warn when first-stage F < 10"""
        np.random.seed(99)
        n = 200

        z = np.random.normal(0, 1, n)
        eps = np.random.normal(0, 1, n)
        # Weak instrument: z barely predicts x_endog
        x_endog = 0.05 * z + eps
        y = 1 + 2 * x_endog + eps

        df = pd.DataFrame({'y': y, 'x_endog': x_endog, 'z': z})

        with pytest.warns(UserWarning, match="Weak instrument"):
            ivreg("y ~ (x_endog ~ z)", data=df)

    # ----------------------------------------------------------------
    # Error handling
    # ----------------------------------------------------------------

    def test_under_identification_error(self, endogenous_data):
        """Should raise error if fewer instruments than endogenous vars"""
        # Try 2 endogenous with only 1 instrument
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'y': np.random.normal(size=n),
            'x1': np.random.normal(size=n),
            'x2': np.random.normal(size=n),
            'z1': np.random.normal(size=n),
        })

        with pytest.raises(ValueError, match="Under-identified"):
            ivreg("y ~ (x1 + x2 ~ z1)", data=df)

    def test_missing_iv_formula_error(self):
        """Should raise error if formula has no IV syntax"""
        df = pd.DataFrame({'y': [1, 2, 3], 'x': [1, 2, 3]})

        with pytest.raises(ValueError, match="IV formula must specify"):
            ivreg("y ~ x", data=df)

    def test_missing_variable_error(self, endogenous_data):
        """Should raise error if variable not found in data"""
        with pytest.raises(ValueError, match="not found in data"):
            ivreg("y ~ (x_endog ~ z_nonexistent) + x_exog", data=endogenous_data)

    # ----------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------

    def test_summary_output(self, endogenous_data):
        """Summary should contain IV-specific information"""
        result = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )
        summary = result.summary()

        assert isinstance(summary, str)
        assert 'IV-2SLS' in summary
        assert 'Two-Stage Least Squares' in summary
        assert 'First-stage F' in summary
        assert 'Sargan' in summary

    def test_model_class_interface(self, endogenous_data):
        """Test the model class interface"""
        model = IVRegression(
            formula="y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )
        result = model.fit()

        assert model.is_fitted
        assert result is not None

        predictions = model.predict()
        assert len(predictions) == len(endogenous_data)

    def test_confidence_intervals(self, endogenous_data):
        """Test confidence interval calculation"""
        result = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )
        ci = result.conf_int()

        assert isinstance(ci, pd.DataFrame)
        assert ci.shape[0] == 3

        # CI should contain the point estimate
        for param in result.params.index:
            lower = ci.loc[param].iloc[0]
            upper = ci.loc[param].iloc[1]
            assert lower < result.params[param] < upper

    def test_repr(self, endogenous_data):
        """Test string representation"""
        result = ivreg(
            "y ~ (x_endog ~ z1 + z2) + x_exog",
            data=endogenous_data,
        )
        repr_str = repr(result)
        assert 'IV-2SLS' in repr_str
        assert '3 parameters' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
