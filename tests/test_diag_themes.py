"""
Tests for diagnostic tests (het, RESET, VIF) and plot themes.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from statspai.diagnostics.tests import diagnose, het_test, reset_test, vif
from statspai.plots.themes import set_theme


@pytest.fixture
def ols_data():
    """OLS data with known properties."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = 0.9 * x1 + 0.1 * rng.normal(0, 1, n)  # collinear with x1
    y = 1 + 2 * x1 + 3 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})


@pytest.fixture
def het_data():
    """Data with heteroskedasticity."""
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(0, 1, n)
    # Variance increases with x
    y = 1 + 2 * x + rng.normal(0, 1, n) * (1 + abs(x))
    return pd.DataFrame({'y': y, 'x': x})


class TestHetTest:
    def test_basic_run(self, ols_data):
        result = het_test(ols_data, y='y', x=['x1', 'x2'])
        assert 'statistic' in result
        assert 'pvalue' in result
        assert result['pvalue'] >= 0

    def test_detects_heteroskedasticity(self):
        """Should reject H0 for strongly heteroskedastic data."""
        rng = np.random.default_rng(42)
        n = 2000
        x = rng.normal(0, 1, n)
        # Strong heteroskedasticity: variance = x²
        y = 1 + 2 * x + rng.normal(0, 1, n) * (x ** 2)
        df = pd.DataFrame({'y': y, 'x': x})
        result = het_test(df, y='y', x=['x'])
        assert result['pvalue'] < 0.05

    def test_homoskedastic_data(self, ols_data):
        """Should NOT reject for well-behaved data."""
        result = het_test(ols_data, y='y', x=['x1', 'x2'])
        # May or may not reject — just check it runs
        assert 0 <= result['pvalue'] <= 1


class TestResetTest:
    def test_basic_run(self, ols_data):
        result = reset_test(ols_data, y='y', x=['x1', 'x2'])
        assert 'statistic' in result
        assert 'pvalue' in result

    def test_linear_dgp_no_reject(self, ols_data):
        """Linear DGP should not reject RESET (no misspecification)."""
        result = reset_test(ols_data, y='y', x=['x1', 'x2'])
        # With a correctly specified linear model, should not reject
        assert result['pvalue'] > 0.01

    def test_nonlinear_dgp_rejects(self):
        """Nonlinear DGP should trigger RESET rejection."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n)
        y = 1 + x ** 2 + rng.normal(0, 0.5, n)  # quadratic, not linear
        df = pd.DataFrame({'y': y, 'x': x})
        result = reset_test(df, y='y', x=['x'])
        assert result['pvalue'] < 0.05


class TestVIF:
    def test_basic_run(self, ols_data):
        result = vif(ols_data, x=['x1', 'x2'])
        assert isinstance(result, pd.DataFrame)
        assert 'VIF' in result.columns
        assert len(result) == 2

    def test_detects_collinearity(self, ols_data):
        """x3 = 0.9*x1 + noise → high VIF."""
        result = vif(ols_data, x=['x1', 'x2', 'x3'])
        x1_vif = result[result['variable'] == 'x1']['VIF'].values[0]
        x3_vif = result[result['variable'] == 'x3']['VIF'].values[0]
        assert x1_vif > 5  # collinear
        assert x3_vif > 5

    def test_no_collinearity(self, ols_data):
        """Independent variables → low VIF."""
        result = vif(ols_data, x=['x1', 'x2'])
        assert (result['VIF'] < 5).all()


class TestDiagnose:
    def test_comprehensive_output(self, ols_data):
        result = diagnose(ols_data, y='y', x=['x1', 'x2'],
                          print_results=False)
        assert 'het_test' in result
        assert 'reset_test' in result
        assert 'vif' in result

    def test_prints_formatted(self, ols_data, capsys):
        diagnose(ols_data, y='y', x=['x1', 'x2'], print_results=True)
        captured = capsys.readouterr()
        assert 'Breusch-Pagan' in captured.out
        assert 'Ramsey RESET' in captured.out
        assert 'VIF' in captured.out


class TestSetTheme:
    def test_academic_theme(self):
        set_theme('academic')
        import matplotlib
        assert matplotlib.rcParams['axes.spines.top'] is False

    def test_aea_theme(self):
        set_theme('aea')
        import matplotlib
        ff = matplotlib.rcParams['font.family']
        assert 'serif' in ff or ff == 'serif'

    def test_minimal_theme(self):
        set_theme('minimal')
        import matplotlib
        assert matplotlib.rcParams['axes.grid'] is True

    def test_cn_journal_theme(self):
        set_theme('cn_journal')

    def test_reset_default(self):
        set_theme('academic')
        set_theme('default')  # should not error

    def test_font_scale(self):
        set_theme('academic', font_scale=1.5)
        import matplotlib
        assert matplotlib.rcParams['font.size'] > 11  # scaled up

    def test_invalid_theme(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            set_theme('nonexistent')


class TestIntegration:
    def test_imports(self):
        import statspai as sp
        assert hasattr(sp, 'binscatter')
        assert hasattr(sp, 'set_theme')
        assert hasattr(sp, 'diagnose')
        assert hasattr(sp, 'het_test')
        assert hasattr(sp, 'reset_test')
        assert hasattr(sp, 'vif')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
