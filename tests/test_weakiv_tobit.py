"""
Tests for Weak IV diagnostics and Tobit regression.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.diagnostics.weak_iv import anderson_rubin_test
from statspai.regression.tobit import tobit
from statspai.core.results import CausalResult


# ======================================================================
# Weak IV
# ======================================================================

@pytest.fixture
def strong_iv_data():
    """Strong instruments: parent_edu → education → wage."""
    rng = np.random.default_rng(42)
    n = 1000
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    education = 12 + 0.8 * z1 + 0.6 * z2 + rng.normal(0, 1, n)
    wage = 5 + 2 * education + rng.normal(0, 2, n)
    return pd.DataFrame({
        'wage': wage, 'education': education,
        'z1': z1, 'z2': z2,
        'experience': rng.normal(10, 3, n),
    })


@pytest.fixture
def weak_iv_data():
    """Weak instruments: barely correlated with endogenous variable."""
    rng = np.random.default_rng(42)
    n = 500
    z = rng.normal(0, 1, n)
    education = 12 + 0.05 * z + rng.normal(0, 3, n)  # very weak
    wage = 5 + 2 * education + rng.normal(0, 2, n)
    return pd.DataFrame({'wage': wage, 'education': education, 'z': z})


class TestAndersonRubin:
    def test_basic_run(self, strong_iv_data):
        result = anderson_rubin_test(
            strong_iv_data, y='wage', endog='education',
            instruments=['z1', 'z2'])
        assert 'ar_stat' in result
        assert 'ar_pvalue' in result
        assert 'first_stage_f' in result
        assert 'effective_f' in result

    def test_strong_iv_high_f(self, strong_iv_data):
        """Strong instruments should have high F."""
        result = anderson_rubin_test(
            strong_iv_data, y='wage', endog='education',
            instruments=['z1', 'z2'])
        assert result['first_stage_f'] > 10

    def test_weak_iv_low_f(self, weak_iv_data):
        """Weak instruments should have low F."""
        result = anderson_rubin_test(
            weak_iv_data, y='wage', endog='education',
            instruments=['z'])
        assert result['first_stage_f'] < 10

    def test_ar_rejects_at_zero(self, strong_iv_data):
        """AR test should reject β=0 when true β=2."""
        result = anderson_rubin_test(
            strong_iv_data, y='wage', endog='education',
            instruments=['z1', 'z2'], h0=0)
        assert result['ar_pvalue'] < 0.05

    def test_ar_ci(self, strong_iv_data):
        """AR CI should contain the true value 2.0."""
        result = anderson_rubin_test(
            strong_iv_data, y='wage', endog='education',
            instruments=['z1', 'z2'])
        lo, hi = result['ar_ci']
        assert lo < 2.0 < hi

    def test_with_exog(self, strong_iv_data):
        result = anderson_rubin_test(
            strong_iv_data, y='wage', endog='education',
            instruments=['z1', 'z2'], exog=['experience'])
        assert 'ar_pvalue' in result

    def test_interpretation(self, strong_iv_data):
        result = anderson_rubin_test(
            strong_iv_data, y='wage', endog='education',
            instruments=['z1', 'z2'])
        assert 'interpretation' in result
        assert 'F =' in result['interpretation']


# ======================================================================
# Tobit
# ======================================================================

@pytest.fixture
def tobit_data():
    """Censored data: Y* = 1 + 2X + ε, Y = max(Y*, 0)."""
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(0, 1, n)
    y_star = 1 + 2 * x + rng.normal(0, 1, n)
    y = np.maximum(y_star, 0)  # left-censored at 0
    return pd.DataFrame({'y': y, 'x': x,
                          'z': rng.normal(0, 1, n)})


class TestTobit:
    def test_basic_run(self, tobit_data):
        result = tobit(tobit_data, y='y', x=['x'], ll=0)
        assert isinstance(result, CausalResult)
        assert 'Tobit' in result.method

    def test_coefficient_positive(self, tobit_data):
        """β_x should be positive (true = 2)."""
        result = tobit(tobit_data, y='y', x=['x'], ll=0)
        assert result.estimate > 0

    def test_coefficient_magnitude(self, tobit_data):
        """β_x should be close to 2.0 (not attenuated like OLS)."""
        result = tobit(tobit_data, y='y', x=['x'], ll=0)
        assert abs(result.estimate - 2.0) < 0.5

    def test_sigma_positive(self, tobit_data):
        result = tobit(tobit_data, y='y', x=['x'], ll=0)
        assert result.model_info['sigma'] > 0

    def test_censoring_info(self, tobit_data):
        result = tobit(tobit_data, y='y', x=['x'], ll=0)
        assert result.model_info['n_censored'] > 0
        assert result.model_info['n_uncensored'] > 0
        assert result.model_info['censor_pct'] > 0

    def test_multiple_regressors(self, tobit_data):
        result = tobit(tobit_data, y='y', x=['x', 'z'], ll=0)
        assert len(result.detail) == 4  # const + x + z + sigma

    def test_detail_has_sigma(self, tobit_data):
        result = tobit(tobit_data, y='y', x=['x'], ll=0)
        assert 'sigma' in result.detail['variable'].values

    def test_cite(self, tobit_data):
        result = tobit(tobit_data, y='y', x=['x'], ll=0)
        assert 'tobin1958' in result.cite()

    def test_no_censoring(self):
        """With no censoring, Tobit ≈ OLS."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = 10 + 2 * x + rng.normal(0, 1, n)  # no censoring
        df = pd.DataFrame({'y': y, 'x': x})
        result = tobit(df, y='y', x=['x'], ll=-100)
        assert abs(result.estimate - 2.0) < 0.3


class TestIntegration:
    def test_imports(self):
        import statspai as sp
        assert hasattr(sp, 'anderson_rubin_test')
        assert hasattr(sp, 'tobit')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
