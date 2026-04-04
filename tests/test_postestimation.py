"""
Tests for post-estimation tools: margins, test, lincom, tab.
"""

import pytest
import os
import tempfile
import numpy as np
import pandas as pd
from statspai import regress, margins, marginsplot, test, lincom, tab


@pytest.fixture
def ols_result():
    """OLS result for testing."""
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
    return regress("y ~ x1 + x2", data=df)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        'treatment': rng.binomial(1, 0.5, n),
        'outcome': rng.binomial(1, 0.4, n),
        'group': rng.choice(['A', 'B', 'C'], n),
        'score': rng.normal(50, 10, n),
    })


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ================================================================
# margins
# ================================================================

class TestMargins:

    def test_linear_margins(self, ols_result):
        """For linear models, margins = coefficients."""
        me = margins(ols_result)
        assert isinstance(me, pd.DataFrame)
        assert 'dy/dx' in me.columns
        assert 'se' in me.columns
        assert 'pvalue' in me.columns

        # For linear model, marginal effect of x1 should be ≈ 2.0
        x1_row = me[me['variable'] == 'x1'].iloc[0]
        assert abs(x1_row['dy/dx'] - 2.0) < 0.3

    def test_margins_specific_vars(self, ols_result):
        """Can specify which variables."""
        me = margins(ols_result, variables=['x1'])
        assert len(me) == 1
        assert me.iloc[0]['variable'] == 'x1'

    def test_margins_ci(self, ols_result):
        """CIs should contain the coefficient."""
        me = margins(ols_result)
        for _, row in me.iterrows():
            assert row['ci_lower'] < row['dy/dx'] < row['ci_upper']

    def test_marginsplot(self, ols_result):
        """marginsplot should produce fig, ax."""
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use('Agg')

        me = margins(ols_result)
        fig, ax = marginsplot(me)
        assert fig is not None


# ================================================================
# test (Wald test)
# ================================================================

class TestHypothesisTest:

    def test_single_zero(self, ols_result):
        """Test beta_x1 = 0 should reject (true value is 2)."""
        result = test(ols_result, "x1 = 0")
        assert result['pvalue'] < 0.01
        assert 'statistic' in result

    def test_equality(self, ols_result):
        """Test beta_x1 = beta_x2 should reject (2 ≠ 3)."""
        result = test(ols_result, "x1 = x2")
        assert result['pvalue'] < 0.05

    def test_true_restriction(self):
        """When restriction is true, should not reject."""
        np.random.seed(42)
        n = 500
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 2 * x2 + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
        result_ols = regress("y ~ x1 + x2", data=df)

        t = test(result_ols, "x1 = x2")
        # Should NOT reject (both coefficients ≈ 2)
        assert t['pvalue'] > 0.05


# ================================================================
# lincom
# ================================================================

class TestLincom:

    def test_sum(self, ols_result):
        """x1 + x2 should be ≈ 5.0 (2 + 3)."""
        result = lincom(ols_result, "x1 + x2")
        assert abs(result['estimate'] - 5.0) < 0.5
        assert 'se' in result
        assert 'ci' in result

    def test_difference(self, ols_result):
        """x1 - x2 should be ≈ -1.0 (2 - 3)."""
        result = lincom(ols_result, "x1 - x2")
        assert abs(result['estimate'] - (-1.0)) < 0.5

    def test_ci_contains_estimate(self, ols_result):
        result = lincom(ols_result, "x1 + x2")
        assert result['ci'][0] < result['estimate'] < result['ci'][1]


# ================================================================
# tab (cross-tabulation)
# ================================================================

class TestTab:

    def test_two_way_text(self, sample_df):
        result = tab(sample_df, 'treatment', 'outcome')
        assert isinstance(result, str)
        assert 'chi2' in result

    def test_one_way(self, sample_df):
        result = tab(sample_df, 'group')
        assert isinstance(result, str)
        assert 'Freq' in result

    def test_dataframe(self, sample_df):
        result = tab(sample_df, 'treatment', 'outcome', output='dataframe')
        assert isinstance(result, pd.DataFrame)

    def test_chi2_test(self, sample_df):
        result = tab(sample_df, 'treatment', 'outcome')
        assert 'Pearson chi2' in result

    def test_excel_export(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, 'tab.xlsx')
        tab(sample_df, 'treatment', 'outcome', output=path)
        assert os.path.exists(path)

    def test_word_export(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, 'tab.docx')
        tab(sample_df, 'treatment', 'outcome', output=path)
        assert os.path.exists(path)

    def test_normalize_row(self, sample_df):
        result = tab(sample_df, 'treatment', 'outcome', normalize='row')
        assert '%' in result


# ================================================================
# Integration
# ================================================================

class TestIntegration:

    def test_imports(self):
        import statspai as sp
        assert hasattr(sp, 'margins')
        assert hasattr(sp, 'test')
        assert hasattr(sp, 'lincom')
        assert hasattr(sp, 'tab')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
