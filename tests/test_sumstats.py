"""
Tests for sumstats and balance_table.
"""

import pytest
import os
import tempfile
import numpy as np
import pandas as pd
from statspai import sumstats, balance_table


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        'wage': rng.normal(50000, 10000, n),
        'age': rng.normal(35, 10, n),
        'edu': rng.normal(14, 3, n),
        'female': rng.binomial(1, 0.5, n),
        'treated': rng.binomial(1, 0.4, n),
    })


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ================================================================
# sumstats
# ================================================================

class TestSumstats:

    def test_basic_text(self, sample_df):
        result = sumstats(sample_df, vars=['wage', 'age', 'edu'])
        assert isinstance(result, str)
        assert 'wage' in result
        assert 'Mean' in result

    def test_all_numeric(self, sample_df):
        """Default: all numeric columns."""
        result = sumstats(sample_df)
        assert 'wage' in result
        assert 'age' in result

    def test_by_group(self, sample_df):
        result = sumstats(sample_df, vars=['wage', 'age'], by='female')
        assert isinstance(result, str)

    def test_dataframe_output(self, sample_df):
        df = sumstats(sample_df, vars=['wage'], output='dataframe')
        assert isinstance(df, pd.DataFrame)
        assert 'wage' in df.index

    def test_custom_stats(self, sample_df):
        result = sumstats(sample_df, vars=['wage'],
                          stats=['n', 'mean', 'sd', 'p10', 'p90'])
        assert 'P10' in result
        assert 'P90' in result

    def test_labels(self, sample_df):
        result = sumstats(sample_df, vars=['wage', 'edu'],
                          labels={'wage': 'Annual Wage ($)', 'edu': 'Education (years)'})
        assert 'Annual Wage' in result

    def test_latex(self, sample_df):
        result = sumstats(sample_df, vars=['wage'], output='latex')
        assert '\\begin' in result

    def test_html(self, sample_df):
        result = sumstats(sample_df, vars=['wage'], output='html')
        assert '<table' in result

    def test_excel_export(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, 'stats.xlsx')
        sumstats(sample_df, vars=['wage', 'age'], output=path)
        assert os.path.exists(path)

    def test_word_export(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, 'stats.docx')
        sumstats(sample_df, vars=['wage', 'age'], output=path)
        assert os.path.exists(path)


# ================================================================
# balance_table
# ================================================================

class TestBalanceTable:

    def test_basic(self, sample_df):
        result = balance_table(sample_df, treat='treated',
                               covariates=['wage', 'age', 'edu'])
        assert isinstance(result, str)
        assert 'wage' in result
        assert 'SMD' in result

    def test_dataframe_output(self, sample_df):
        df = balance_table(sample_df, treat='treated',
                           covariates=['wage', 'age'],
                           output='dataframe')
        assert isinstance(df, pd.DataFrame)
        assert 'Treated Mean' in df.columns
        assert 'Control Mean' in df.columns
        assert 'SMD' in df.columns

    def test_pvalue_present(self, sample_df):
        df = balance_table(sample_df, treat='treated',
                           covariates=['wage', 'age'],
                           output='dataframe')
        assert 'p-value' in df.columns

    def test_n_row(self, sample_df):
        df = balance_table(sample_df, treat='treated',
                           covariates=['wage'],
                           output='dataframe')
        assert 'N' in df.index

    def test_labels(self, sample_df):
        result = balance_table(sample_df, treat='treated',
                               covariates=['wage', 'age'],
                               labels={'wage': 'Salary', 'age': 'Age (years)'})
        assert 'Salary' in result

    def test_excel_export(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, 'balance.xlsx')
        balance_table(sample_df, treat='treated',
                      covariates=['wage', 'age'], output=path)
        assert os.path.exists(path)

    def test_word_export(self, sample_df, tmp_dir):
        path = os.path.join(tmp_dir, 'balance.docx')
        balance_table(sample_df, treat='treated',
                      covariates=['wage', 'age'], output=path)
        assert os.path.exists(path)


# ================================================================
# result.to_docx()
# ================================================================

class TestResultToDocx:

    def test_econometric_to_docx(self, tmp_dir):
        from statspai import regress
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.normal(0, 1, 100),
            'x': np.random.normal(0, 1, 100),
        })
        result = regress("y ~ x", data=df)
        path = os.path.join(tmp_dir, 'ols.docx')
        result.to_docx(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000

    def test_causal_to_docx(self, tmp_dir):
        from statspai import rdrobust
        rng = np.random.default_rng(42)
        n = 1000
        X = rng.uniform(-1, 1, n)
        Y = 0.5 * X + 3.0 * (X >= 0) + rng.normal(0, 0.3, n)
        df = pd.DataFrame({'y': Y, 'x': X})
        result = rdrobust(df, y='y', x='x')
        path = os.path.join(tmp_dir, 'rd.docx')
        result.to_docx(path, title='RD Results')
        assert os.path.exists(path)


# ================================================================
# Integration: top-level imports
# ================================================================

class TestIntegration:

    def test_imports(self):
        import statspai as sp
        assert hasattr(sp, 'sumstats')
        assert hasattr(sp, 'balance_table')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
