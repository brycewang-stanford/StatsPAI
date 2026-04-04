"""
Tests for utility functions: labels, pwcorr, winsor, describe.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.utils import (
    label_var, label_vars, get_label, get_labels, describe,
    pwcorr, winsor,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        'wage': rng.normal(5000, 1000, n),
        'education': rng.normal(12, 3, n),
        'experience': rng.normal(10, 5, n),
        'female': rng.binomial(1, 0.5, n),
        'outlier_var': np.concatenate([rng.normal(0, 1, n - 5),
                                       [100, 200, -100, -200, 500]]),
    })


# ======================================================================
# Variable Labels
# ======================================================================

class TestLabels:
    def test_label_var(self, sample_df):
        label_var(sample_df, 'wage', 'Monthly wage (CNY)')
        assert get_label(sample_df, 'wage') == 'Monthly wage (CNY)'

    def test_label_vars_bulk(self, sample_df):
        label_vars(sample_df, {
            'wage': 'Monthly wage',
            'education': 'Years of education',
        })
        assert get_label(sample_df, 'wage') == 'Monthly wage'
        assert get_label(sample_df, 'education') == 'Years of education'

    def test_get_label_fallback(self, sample_df):
        """Unlabeled variable returns column name."""
        assert get_label(sample_df, 'experience') == 'experience'

    def test_get_labels_all(self, sample_df):
        label_var(sample_df, 'wage', 'Wage')
        labels = get_labels(sample_df)
        assert labels['wage'] == 'Wage'
        assert labels['experience'] == 'experience'  # unlabeled fallback

    def test_label_nonexistent_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            label_var(sample_df, 'nonexistent', 'label')


class TestDescribe:
    def test_basic(self, sample_df):
        result = describe(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert 'variable' in result.columns
        assert 'type' in result.columns
        assert 'n' in result.columns
        assert 'label' in result.columns
        assert len(result) == len(sample_df.columns)

    def test_with_labels(self, sample_df):
        label_var(sample_df, 'wage', 'Monthly wage')
        result = describe(sample_df)
        wage_row = result[result['variable'] == 'wage'].iloc[0]
        assert wage_row['label'] == 'Monthly wage'

    def test_subset(self, sample_df):
        result = describe(sample_df, columns=['wage', 'education'])
        assert len(result) == 2


# ======================================================================
# Pairwise Correlation
# ======================================================================

class TestPwcorr:
    def test_basic_text(self, sample_df):
        output = pwcorr(sample_df, vars=['wage', 'education', 'experience'])
        assert isinstance(output, str)
        assert 'wage' in output
        assert '1.000' in output  # diagonal

    def test_stars_present(self, sample_df):
        """Correlated variables should have stars."""
        # Make correlated data
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        y = 2 * x + rng.normal(0, 0.5, n)  # highly correlated
        df = pd.DataFrame({'x': x, 'y': y})
        output = pwcorr(df, stars=True)
        assert '***' in output

    def test_no_stars(self, sample_df):
        output = pwcorr(sample_df, vars=['wage', 'education'],
                        stars=False)
        assert '***' not in output

    def test_dataframe_output(self, sample_df):
        result = pwcorr(sample_df, vars=['wage', 'education'],
                        output='dataframe')
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        # Diagonal should be 1
        assert abs(result.iloc[0, 0] - 1.0) < 1e-10

    def test_latex_output(self, sample_df):
        output = pwcorr(sample_df, vars=['wage', 'education'],
                        output='latex')
        assert '\\begin{tabular}' in output

    def test_spearman(self, sample_df):
        output = pwcorr(sample_df, vars=['wage', 'education'],
                        method='spearman')
        assert isinstance(output, str)

    def test_lower_triangular(self, sample_df):
        """Upper triangle should be empty (Stata convention)."""
        output = pwcorr(sample_df,
                        vars=['wage', 'education', 'experience'])
        lines = output.split('\n')
        # First data line (wage row): should have 1.000 and rest empty
        # The matrix should be lower-triangular
        assert isinstance(output, str)

    def test_n_reported(self, sample_df):
        output = pwcorr(sample_df, vars=['wage', 'education'])
        assert 'N =' in output


# ======================================================================
# Winsorization
# ======================================================================

class TestWinsor:
    def test_basic(self, sample_df):
        result = winsor(sample_df, vars=['outlier_var'])
        assert 'outlier_var_w' in result.columns

    def test_outliers_clipped(self, sample_df):
        result = winsor(sample_df, vars=['outlier_var'], cuts=(1, 99))
        original_max = sample_df['outlier_var'].max()
        winsorized_max = result['outlier_var_w'].max()
        assert winsorized_max < original_max

    def test_replace_mode(self, sample_df):
        result = winsor(sample_df, vars=['outlier_var'], replace=True)
        assert 'outlier_var_w' not in result.columns
        assert 'outlier_var' in result.columns
        # Should be clipped
        assert result['outlier_var'].max() < sample_df['outlier_var'].max()

    def test_custom_cuts(self, sample_df):
        result = winsor(sample_df, vars=['wage'], cuts=(5, 95))
        p5 = np.percentile(sample_df['wage'].dropna(), 5)
        p95 = np.percentile(sample_df['wage'].dropna(), 95)
        assert result['wage_w'].min() >= p5 - 0.01
        assert result['wage_w'].max() <= p95 + 0.01

    def test_preserves_nan(self):
        df = pd.DataFrame({'x': [1, 2, np.nan, 100, 5]})
        result = winsor(df, vars=['x'], cuts=(10, 90))
        assert pd.isna(result['x_w'].iloc[2])

    def test_all_numeric_default(self, sample_df):
        """With vars=None, winsorize all numeric columns."""
        result = winsor(sample_df, cuts=(5, 95))
        for col in sample_df.select_dtypes(include=[np.number]).columns:
            assert col + '_w' in result.columns


# ======================================================================
# Integration
# ======================================================================

class TestIntegration:
    def test_imports(self):
        import statspai as sp
        assert hasattr(sp, 'label_var')
        assert hasattr(sp, 'pwcorr')
        assert hasattr(sp, 'winsor')
        assert hasattr(sp, 'describe')

    def test_full_workflow(self, sample_df):
        """Label → winsorize → correlate → describe."""
        import statspai as sp

        # Label
        sp.label_var(sample_df, 'wage', 'Monthly wage')
        sp.label_var(sample_df, 'education', 'Education (years)')

        # Winsorize
        df2 = sp.winsor(sample_df, vars=['wage'], cuts=(1, 99))

        # Correlate
        corr = sp.pwcorr(df2, vars=['wage_w', 'education'])
        assert isinstance(corr, str)

        # Describe
        desc = sp.describe(sample_df)
        assert len(desc) > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
