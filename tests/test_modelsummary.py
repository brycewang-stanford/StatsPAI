"""
Tests for modelsummary and coefplot.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from statspai import regress, did, rdrobust, modelsummary, coefplot
from statspai.output.modelsummary import modelsummary as ms


@pytest.fixture
def ols_models():
    """Two OLS models for comparison."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 1 + 2 * x1 + 3 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})

    r1 = regress("y ~ x1", data=df)
    r2 = regress("y ~ x1 + x2", data=df)
    r3 = regress("y ~ x1 + x2 + x3", data=df)
    return r1, r2, r3, df


@pytest.fixture
def did_model():
    """A 2x2 DID model."""
    rng = np.random.default_rng(42)
    n = 400
    d = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    y = 1 + 2 * d + 3 * t + 5 * d * t + rng.normal(0, 1, n)
    df = pd.DataFrame({'y': y, 'd': d, 't': t})
    return did(df, y='y', treat='d', time='t')


@pytest.fixture
def rd_model():
    """A sharp RD model."""
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.uniform(-1, 1, n)
    Y = 0.5 * X + 3.0 * (X >= 0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'y': Y, 'x': X})
    return rdrobust(df, y='y', x='x')


class TestModelsummaryText:
    """Test text output format."""

    def test_basic_text(self, ols_models):
        r1, r2, r3, _ = ols_models
        output = modelsummary(r1, r2, r3, output='text')
        assert isinstance(output, str)
        assert '(1)' in output
        assert '(2)' in output
        assert '(3)' in output

    def test_contains_coefficients(self, ols_models):
        r1, r2, r3, _ = ols_models
        output = modelsummary(r1, r2, output='text')
        assert 'x1' in output
        assert 'Intercept' in output

    def test_contains_stars(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2, output='text')
        assert '***' in output  # x1 and x2 should be significant

    def test_no_stars(self, ols_models):
        r1, _, _, _ = ols_models
        output = modelsummary(r1, stars=False, output='text')
        assert '***' not in output

    def test_contains_nobs(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2, output='text')
        assert 'Observations' in output or '500' in output

    def test_custom_model_names(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2, model_names=['Base', 'Full'], output='text')
        assert 'Base' in output
        assert 'Full' in output

    def test_with_title(self, ols_models):
        r1, _, _, _ = ols_models
        output = modelsummary(r1, title='My Results', output='text')
        assert 'My Results' in output

    def test_with_notes(self, ols_models):
        r1, _, _, _ = ols_models
        output = modelsummary(r1, notes=['Robust SEs'], output='text')
        assert 'Robust SEs' in output


class TestModelsummaryFormats:
    """Test different output formats."""

    def test_latex(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2, output='latex')
        assert '\\begin{table}' in output
        assert '\\end{table}' in output
        assert '\\hline' in output

    def test_html(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2, output='html')
        assert '<table' in output
        assert '</table>' in output

    def test_dataframe(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2, output='dataframe')
        assert isinstance(output, pd.DataFrame)
        assert len(output) > 0


class TestModelsummaryMixed:
    """Test mixing different model types."""

    def test_ols_and_did(self, ols_models, did_model):
        r1, _, _, _ = ols_models
        output = modelsummary(r1, did_model, output='text')
        assert isinstance(output, str)
        # Should contain both OLS and DID coefficients
        assert 'x1' in output or 'ATT' in output

    def test_ols_and_rd(self, ols_models, rd_model):
        r1, _, _, _ = ols_models
        output = modelsummary(r1, rd_model, output='text')
        assert isinstance(output, str)

    def test_all_causal(self, did_model, rd_model):
        output = modelsummary(did_model, rd_model,
                              model_names=['DID', 'RD'], output='text')
        assert 'DID' in output
        assert 'RD' in output


class TestModelsummaryOptions:
    """Test customization options."""

    def test_coef_map(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2,
                              coef_map={'x1': 'Education', 'x2': 'Income'},
                              output='text')
        assert 'Education' in output

    def test_add_rows(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2,
                              add_rows={'FE: Year': ['No', 'Yes']},
                              output='text')
        assert 'FE: Year' in output

    def test_show_ci(self, ols_models):
        r1, _, _, _ = ols_models
        output = modelsummary(r1, show_ci=True, output='text')
        assert '[' in output  # CI brackets

    def test_se_brackets(self, ols_models):
        r1, _, _, _ = ols_models
        output = modelsummary(r1, se_type='brackets', output='text')
        # Should have bracket-style SEs
        assert '[' in output

    def test_extra_stats(self, ols_models):
        r1, r2, _, _ = ols_models
        output = modelsummary(r1, r2,
                              stats=['nobs', 'r_squared', 'adj_r_squared'],
                              output='text')
        assert 'R²' in output or 'R-squared' in output


class TestCoefplot:
    """Test coefficient plot."""

    def test_basic_coefplot(self, ols_models):
        r1, r2, _, _ = ols_models
        fig, ax = coefplot(r1, r2)
        assert fig is not None

    def test_custom_names(self, ols_models):
        r1, r2, _, _ = ols_models
        fig, ax = coefplot(r1, r2, model_names=['Base', 'Full'])
        assert fig is not None

    def test_mixed_models(self, ols_models, did_model):
        r1, _, _, _ = ols_models
        fig, ax = coefplot(r1, did_model, model_names=['OLS', 'DID'])
        assert fig is not None


class TestIntegration:
    def test_top_level_import(self):
        import statspai as sp
        assert hasattr(sp, 'modelsummary')
        assert hasattr(sp, 'coefplot')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
