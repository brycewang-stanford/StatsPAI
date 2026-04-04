"""
Tests for quantile regression (qreg, sqreg).
"""

import pytest
import numpy as np
import pandas as pd

from statspai.regression.quantile import qreg, sqreg
from statspai.core.results import CausalResult


@pytest.fixture
def qreg_data():
    """Data with heterogeneous effects across distribution.

    Y = 1 + 2*X + (1 + X)*ε  → effect of X varies by quantile.
    At median: effect ≈ 2. At Q(0.9): effect > 2.
    """
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(0, 1, n)
    eps = rng.normal(0, 1, n)
    y = 1 + 2 * x + (1 + 0.5 * x) * eps
    return pd.DataFrame({'y': y, 'x': x, 'z': rng.normal(0, 1, n)})


class TestQreg:
    def test_basic_median(self, qreg_data):
        result = qreg(qreg_data, y='y', x=['x'], quantile=0.5)
        assert isinstance(result, CausalResult)
        assert '0.5' in result.method

    def test_median_close_to_ols(self, qreg_data):
        """Median regression should be close to OLS for symmetric errors."""
        result = qreg(qreg_data, y='y', x=['x'], quantile=0.5)
        assert abs(result.estimate - 2.0) < 0.5

    def test_high_quantile(self, qreg_data):
        """At Q(0.9), effect should be larger (heteroscedastic DGP)."""
        r50 = qreg(qreg_data, y='y', x=['x'], quantile=0.5)
        r90 = qreg(qreg_data, y='y', x=['x'], quantile=0.9)
        assert r90.estimate > r50.estimate

    def test_low_quantile(self, qreg_data):
        result = qreg(qreg_data, y='y', x=['x'], quantile=0.1)
        assert isinstance(result, CausalResult)

    def test_multiple_regressors(self, qreg_data):
        result = qreg(qreg_data, y='y', x=['x', 'z'], quantile=0.5)
        assert len(result.detail) == 3  # const + x + z

    def test_formula_interface(self, qreg_data):
        result = qreg(qreg_data, formula='y ~ x', quantile=0.5)
        assert isinstance(result, CausalResult)

    def test_detail_table(self, qreg_data):
        result = qreg(qreg_data, y='y', x=['x'], quantile=0.5)
        assert 'variable' in result.detail.columns
        assert 'coefficient' in result.detail.columns

    def test_cite(self, qreg_data):
        result = qreg(qreg_data, y='y', x=['x'], quantile=0.5)
        assert 'koenker' in result.cite().lower()

    def test_invalid_quantile(self, qreg_data):
        with pytest.raises(ValueError, match="quantile"):
            qreg(qreg_data, y='y', x=['x'], quantile=1.5)


class TestSqreg:
    def test_basic(self, qreg_data):
        result = sqreg(qreg_data, y='y', x=['x'])
        assert isinstance(result, pd.DataFrame)
        assert 'Q(0.5)' in result.columns
        assert 'Q(0.1)' in result.columns
        assert 'Q(0.9)' in result.columns

    def test_custom_quantiles(self, qreg_data):
        result = sqreg(qreg_data, y='y', x=['x'],
                       quantiles=[0.25, 0.75])
        assert 'Q(0.25)' in result.columns
        assert 'Q(0.75)' in result.columns

    def test_includes_all_vars(self, qreg_data):
        result = sqreg(qreg_data, y='y', x=['x', 'z'])
        assert len(result) == 3  # const, x, z


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'qreg')
        assert hasattr(sp, 'sqreg')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
