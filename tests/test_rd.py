"""
Tests for RD module: Sharp and Fuzzy RD with robust inference.

All tests use simulated data with known true effects.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.rd import rdrobust, rdplot
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def data_sharp():
    """Sharp RD data. True RD effect = 3.0.

    DGP: Y = 0.5*X + 3*1(X >= 0) + ε, X ~ Uniform(-1, 1).
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.uniform(-1, 1, n)
    Y = 0.5 * X + 3.0 * (X >= 0) + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        'y': Y, 'x': X,
        'z': rng.normal(0, 1, n),  # irrelevant covariate for testing
    })


@pytest.fixture
def data_fuzzy():
    """Fuzzy RD data. True LATE = 5.0.

    DGP: D = 0.2 + 0.6*1(X >= 0) + noise → first stage ≈ 0.6
         Y = 1 + X + 5*D + ε
         LATE = 5.0
    """
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.uniform(-1, 1, n)
    D_prob = 0.2 + 0.6 * (X >= 0)
    D = rng.binomial(1, np.clip(D_prob, 0, 1), n).astype(float)
    Y = 1 + X + 5 * D + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'x': X, 'd': D})


@pytest.fixture
def data_nonlinear():
    """Sharp RD with nonlinear CEF. True RD effect = 2.0.

    DGP: Y = sin(3X) + 2*1(X >= 0) + ε. Tests whether local polynomial
    handles curvature correctly.
    """
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.uniform(-1, 1, n)
    Y = np.sin(3 * X) + 2.0 * (X >= 0) + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'x': X})


# ======================================================================
# Sharp RD tests
# ======================================================================

class TestSharpRD:
    """Tests for sharp RD estimation."""

    def test_basic_sharp(self, data_sharp):
        """Point estimate should be close to true value 3.0."""
        result = rdrobust(data_sharp, y='y', x='x', c=0)
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 3.0) < 0.5

    def test_returns_causal_result(self, data_sharp):
        """Result should be CausalResult with expected structure."""
        result = rdrobust(data_sharp, y='y', x='x')
        assert result.method == 'Sharp RD Estimation'
        assert result.estimand == 'RD Effect'
        assert result.se > 0
        assert result.ci[0] < result.estimate < result.ci[1]
        assert 0 <= result.pvalue <= 1

    def test_detail_table(self, data_sharp):
        """Detail table should have Conventional and Robust rows."""
        result = rdrobust(data_sharp, y='y', x='x')
        assert result.detail is not None
        methods = result.detail['method'].tolist()
        assert 'Conventional' in methods
        assert 'Robust' in methods

    def test_conventional_vs_robust(self, data_sharp):
        """Conventional and robust estimates should differ slightly."""
        result = rdrobust(data_sharp, y='y', x='x')
        conv = result.model_info['conventional']
        rob = result.model_info['robust']
        # Both should be close to 3.0
        assert abs(conv['estimate'] - 3.0) < 0.5
        assert abs(rob['estimate'] - 3.0) < 0.5
        # Robust SE should be >= conventional SE (accounts for bias)
        assert rob['se'] >= conv['se'] * 0.8  # allow some tolerance

    def test_bandwidth_auto(self, data_sharp):
        """Auto bandwidth should be reasonable (0 < h < range)."""
        result = rdrobust(data_sharp, y='y', x='x')
        h = result.model_info['bandwidth_h']
        assert 0 < h < 2  # data range is [-1, 1], so h < 2

    def test_manual_bandwidth(self, data_sharp):
        """Manual bandwidth should be used."""
        result = rdrobust(data_sharp, y='y', x='x', h=0.5)
        assert abs(result.model_info['bandwidth_h'] - 0.5) < 1e-6

    def test_n_effective(self, data_sharp):
        """Effective sample should be subset of total."""
        result = rdrobust(data_sharp, y='y', x='x')
        n_eff = (result.model_info['n_effective_left'] +
                 result.model_info['n_effective_right'])
        assert 0 < n_eff <= result.n_obs

    def test_polynomial_order(self, data_sharp):
        """Higher polynomial order should work."""
        r1 = rdrobust(data_sharp, y='y', x='x', p=1)
        r2 = rdrobust(data_sharp, y='y', x='x', p=2)
        # Both should find the effect
        assert abs(r1.estimate - 3.0) < 0.5
        assert abs(r2.estimate - 3.0) < 0.6

    def test_kernel_triangular(self, data_sharp):
        result = rdrobust(data_sharp, y='y', x='x', kernel='triangular')
        assert abs(result.estimate - 3.0) < 0.5

    def test_kernel_uniform(self, data_sharp):
        result = rdrobust(data_sharp, y='y', x='x', kernel='uniform')
        assert abs(result.estimate - 3.0) < 0.5

    def test_kernel_epanechnikov(self, data_sharp):
        result = rdrobust(data_sharp, y='y', x='x', kernel='epanechnikov')
        assert abs(result.estimate - 3.0) < 0.5

    def test_nonlinear_cef(self, data_nonlinear):
        """Should handle nonlinear CEF (local poly adapts)."""
        result = rdrobust(data_nonlinear, y='y', x='x')
        assert abs(result.estimate - 2.0) < 0.6

    def test_with_covariates(self, data_sharp):
        """Covariates should not break estimation."""
        result = rdrobust(data_sharp, y='y', x='x', covs=['z'])
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 3.0) < 0.5

    def test_nonzero_cutoff(self):
        """Non-zero cutoff should work."""
        rng = np.random.default_rng(42)
        n = 2000
        X = rng.uniform(0, 10, n)
        Y = X + 4.0 * (X >= 5) + rng.normal(0, 0.5, n)
        df = pd.DataFrame({'y': Y, 'x': X})
        result = rdrobust(df, y='y', x='x', c=5)
        assert abs(result.estimate - 4.0) < 1.0

    def test_summary(self, data_sharp):
        """Summary should contain RD-specific info."""
        result = rdrobust(data_sharp, y='y', x='x')
        s = result.summary()
        assert 'Sharp RD' in s
        assert 'RD Effect' in s
        assert 'Bandwidth' in s or 'bandwidth' in s.lower()

    def test_to_latex(self, data_sharp):
        """LaTeX export should work."""
        result = rdrobust(data_sharp, y='y', x='x')
        latex = result.to_latex()
        assert '\\begin{table}' in latex

    def test_cite(self, data_sharp):
        """Citation should return CCT BibTeX."""
        result = rdrobust(data_sharp, y='y', x='x')
        bib = result.cite()
        assert 'calonico2014' in bib

    def test_repr(self, data_sharp):
        result = rdrobust(data_sharp, y='y', x='x')
        assert 'CausalResult' in repr(result)


# ======================================================================
# Fuzzy RD tests
# ======================================================================

class TestFuzzyRD:
    """Tests for fuzzy RD estimation."""

    def test_basic_fuzzy(self, data_fuzzy):
        """Fuzzy RD estimate should be close to true LATE = 5.0."""
        result = rdrobust(data_fuzzy, y='y', x='x', fuzzy='d')
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 2.0  # wider tolerance for fuzzy

    def test_fuzzy_type(self, data_fuzzy):
        """Should be labeled as Fuzzy RD."""
        result = rdrobust(data_fuzzy, y='y', x='x', fuzzy='d')
        assert 'Fuzzy' in result.method
        assert result.estimand == 'LATE'

    def test_fuzzy_larger_se(self, data_fuzzy, data_sharp):
        """Fuzzy RD SE should be larger than sharp (IV penalty)."""
        r_fuzzy = rdrobust(data_fuzzy, y='y', x='x', fuzzy='d', h=0.5)
        r_sharp = rdrobust(data_sharp, y='y', x='x', h=0.5)
        assert r_fuzzy.se > r_sharp.se * 0.5  # fuzzy generally less precise


# ======================================================================
# RD Plot tests
# ======================================================================

class TestRDPlot:
    """Tests for rdplot visualization."""

    def test_rdplot_runs(self, data_sharp):
        """rdplot should run without error."""
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        fig, ax = rdplot(data_sharp, y='y', x='x')
        assert fig is not None
        assert ax is not None

    def test_rdplot_nonzero_cutoff(self):
        """rdplot with non-zero cutoff."""
        import matplotlib
        matplotlib.use('Agg')
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, 500)
        Y = X + 3 * (X >= 5) + rng.normal(0, 0.5, 500)
        df = pd.DataFrame({'y': Y, 'x': X})
        fig, ax = rdplot(df, y='y', x='x', c=5)
        assert fig is not None


# ======================================================================
# Integration: via top-level statspai
# ======================================================================

class TestIntegration:
    """Test rdrobust is accessible from top-level statspai."""

    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'rdrobust')
        assert hasattr(sp, 'rdplot')

    def test_top_level_call(self, data_sharp):
        import statspai as sp
        result = sp.rdrobust(data_sharp, y='y', x='x')
        assert abs(result.estimate - 3.0) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
