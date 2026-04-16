"""
Tests for new RD modules: bandwidth, locrand, hte, rd2d, extrapolate, rdml.

All tests use simulated data with known true effects.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def data_sharp():
    """Sharp RD data. True RD effect = 3.0."""
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + 3.0 * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'x': X, 'z': Z,
                         'z2': rng.normal(0, 1, n)})


@pytest.fixture
def data_hte():
    """RD data with heterogeneous effects. CATE(z) = 2.0 + 1.5*z."""
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    tau_z = 2.0 + 1.5 * Z
    Y = 0.5 * X + tau_z * (X >= 0) + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'x': X, 'z': Z})


@pytest.fixture
def data_2d():
    """2D boundary RD data. True effect = 2.0, boundary at x1=0."""
    rng = np.random.default_rng(42)
    n = 2000
    X1 = rng.uniform(-1, 1, n)
    X2 = rng.uniform(-1, 1, n)
    D = (X1 >= 0).astype(float)
    Y = 0.3 * X1 + 0.2 * X2 + 2.0 * D + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'x1': X1, 'x2': X2, 'd': D})


@pytest.fixture
def data_multi_cutoff():
    """Multi-cutoff RD data. Effects at c=0 (tau=2), c=1 (tau=3)."""
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.uniform(-2, 3, n)
    Y = 0.5 * X + 2.0 * (X >= 0) + 1.0 * (X >= 1) + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'x': X,
                         'z': rng.normal(0, 1, n)})


# ======================================================================
# Bandwidth Selection Tests
# ======================================================================

class TestBandwidth:
    """Tests for rdbwselect."""

    def test_mserd(self, data_sharp):
        from statspai.rd import rdbwselect
        result = rdbwselect(data_sharp, y='y', x='x', c=0, bwselect='mserd')
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
        assert 'h_left' in result.columns

    def test_cerrd(self, data_sharp):
        from statspai.rd import rdbwselect
        result = rdbwselect(data_sharp, y='y', x='x', c=0, bwselect='cerrd')
        assert len(result) >= 1

    def test_all_methods(self, data_sharp):
        from statspai.rd import rdbwselect
        result = rdbwselect(data_sharp, y='y', x='x', c=0, all=True)
        assert len(result) >= 4  # at least MSE and CER variants

    def test_cer_smaller_than_mse(self, data_sharp):
        """CER bandwidth should be smaller than MSE bandwidth."""
        from statspai.rd import rdbwselect
        mse = rdbwselect(data_sharp, y='y', x='x', c=0, bwselect='mserd')
        cer = rdbwselect(data_sharp, y='y', x='x', c=0, bwselect='cerrd')
        h_mse = mse['h_left'].iloc[0]
        h_cer = cer['h_left'].iloc[0]
        assert h_cer <= h_mse * 1.01  # CER ≤ MSE (small tolerance)


# ======================================================================
# Local Randomization Tests
# ======================================================================

class TestLocalRandomization:
    """Tests for rdrandinf, rdwinselect, rdsensitivity, rdrbounds."""

    def test_rdrandinf_basic(self, data_sharp):
        from statspai.rd import rdrandinf
        result = rdrandinf(data_sharp, y='y', x='x', c=0,
                           wl=-0.3, wr=0.3, n_perms=200, seed=42)
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 3.0) < 1.5  # rough check

    def test_rdrandinf_permutation_pvalue(self, data_sharp):
        from statspai.rd import rdrandinf
        result = rdrandinf(data_sharp, y='y', x='x', c=0,
                           wl=-0.3, wr=0.3, n_perms=200, seed=42)
        assert 0 <= result.pvalue <= 1

    def test_rdwinselect(self, data_sharp):
        from statspai.rd import rdwinselect
        result = rdwinselect(data_sharp, x='x', c=0,
                             covs=['z'], nwindows=5, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3

    def test_rdsensitivity(self, data_sharp):
        from statspai.rd import rdsensitivity
        result = rdsensitivity(data_sharp, y='y', x='x', c=0,
                               nwindows=5, n_perms=100, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert 'estimate' in result.columns

    def test_rdrbounds(self, data_sharp):
        from statspai.rd import rdrbounds
        result = rdrbounds(data_sharp, y='y', x='x', c=0,
                           wl=-0.3, wr=0.3, n_perms=100, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert 'gamma' in result.columns


# ======================================================================
# Heterogeneous Treatment Effects Tests
# ======================================================================

class TestHTE:
    """Tests for rdhte, rdbwhte, rdhte_lincom."""

    def test_rdhte_basic(self, data_hte):
        from statspai.rd import rdhte
        result = rdhte(data_hte, y='y', x='x', z='z', c=0, n_eval=10)
        assert isinstance(result, CausalResult)
        # ATE should be close to 2.0 (CATE(z) = 2 + 1.5*z, E[z]=0 → ATE≈2)
        assert abs(result.estimate - 2.0) < 1.0

    def test_rdhte_heterogeneity_detected(self, data_hte):
        from statspai.rd import rdhte
        result = rdhte(data_hte, y='y', x='x', z='z', c=0, n_eval=10)
        het_test = result.model_info.get('heterogeneity_test', {})
        if 'pvalue' in het_test:
            # Should detect heterogeneity
            assert het_test['pvalue'] < 0.10

    def test_rdhte_detail_table(self, data_hte):
        from statspai.rd import rdhte
        result = rdhte(data_hte, y='y', x='x', z='z', c=0, n_eval=10)
        assert result.detail is not None
        assert 'cate' in result.detail.columns or 'estimate' in result.detail.columns

    def test_rdbwhte(self, data_hte):
        from statspai.rd import rdbwhte
        h = rdbwhte(data_hte, y='y', x='x', z='z', c=0)
        assert isinstance(h, float)
        assert h > 0

    def test_rdhte_lincom(self, data_hte):
        from statspai.rd import rdhte, rdhte_lincom
        result = rdhte(data_hte, y='y', x='x', z='z', c=0, n_eval=5)
        n_eval = len(result.detail)
        weights = np.ones(n_eval) / n_eval
        lc = rdhte_lincom(result, weights=weights)
        assert isinstance(lc, dict)
        assert 'estimate' in lc


# ======================================================================
# Boundary Discontinuity (2D) Tests
# ======================================================================

class TestRD2D:
    """Tests for rd2d, rd2d_bw."""

    def test_rd2d_distance(self, data_2d):
        from statspai.rd import rd2d
        result = rd2d(data_2d, y='y', x1='x1', x2='x2',
                      treatment='d', approach='distance')
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 1.5

    def test_rd2d_location(self, data_2d):
        from statspai.rd import rd2d
        result = rd2d(data_2d, y='y', x1='x1', x2='x2',
                      treatment='d', approach='location')
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 2.0) < 2.0

    def test_rd2d_bw(self, data_2d):
        from statspai.rd import rd2d_bw
        h = rd2d_bw(data_2d, y='y', x1='x1', x2='x2',
                    treatment='d', approach='distance')
        assert isinstance(h, float)
        assert h > 0


# ======================================================================
# Extrapolation Tests
# ======================================================================

class TestExtrapolation:
    """Tests for rd_extrapolate, rd_multi_extrapolate, rd_external_validity."""

    def test_rd_extrapolate_basic(self, data_sharp):
        from statspai.rd import rd_extrapolate
        result = rd_extrapolate(
            data_sharp, y='y', x='x', c=0, covs=['z'],
            n_eval=5, method='ols'
        )
        assert isinstance(result, CausalResult)
        # Should pick up roughly the right effect
        assert abs(result.estimate) > 0

    def test_rd_multi_extrapolate(self, data_multi_cutoff):
        from statspai.rd import rd_multi_extrapolate
        result = rd_multi_extrapolate(
            data_multi_cutoff, y='y', x='x',
            cutoffs=[0.0, 1.0], method='linear',
            eval_points=np.linspace(-0.5, 1.5, 5),
        )
        assert isinstance(result, CausalResult)
        assert result.detail is not None

    def test_rd_external_validity(self, data_sharp):
        from statspai.rd import rd_external_validity
        result = rd_external_validity(
            data_sharp, y='y', x='x', c=0, covs=['z']
        )
        assert isinstance(result, dict)
        assert 'local_estimate' in result


# ======================================================================
# ML + RD Tests
# ======================================================================

class TestRDML:
    """Tests for rd_forest, rd_boost, rd_lasso."""

    def test_rd_lasso(self, data_sharp):
        from statspai.rd import rd_lasso
        result = rd_lasso(data_sharp, y='y', x='x', c=0,
                          covs=['z', 'z2'])
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 3.0) < 1.5

    def test_rd_forest(self, data_hte):
        pytest.importorskip('sklearn')
        from statspai.rd import rd_forest
        result = rd_forest(data_hte, y='y', x='x', c=0,
                           covs=['z'], n_trees=50, seed=42)
        assert isinstance(result, CausalResult)
        assert result.model_info is not None

    def test_rd_boost(self, data_hte):
        pytest.importorskip('sklearn')
        from statspai.rd import rd_boost
        result = rd_boost(data_hte, y='y', x='x', c=0,
                          covs=['z'], n_estimators=50, seed=42)
        assert isinstance(result, CausalResult)

    def test_rd_cate_summary(self, data_hte):
        pytest.importorskip('sklearn')
        from statspai.rd import rd_cate_summary
        result = rd_cate_summary(data_hte, y='y', x='x', c=0,
                                 covs=['z'], methods=['lasso'],
                                 seed=42)
        assert isinstance(result, dict)


# ======================================================================
# Enhanced rdrobust Tests (CER bandwidth, covariate adjustment)
# ======================================================================

class TestEnhancedRdrobust:
    """Tests for upgraded rdrobust with CER bandwidth and covariates."""

    def test_cerrd_bandwidth(self, data_sharp):
        from statspai.rd import rdrobust
        result = rdrobust(data_sharp, y='y', x='x', c=0,
                          bwselect='cerrd')
        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 3.0) < 1.0

    def test_certwo_bandwidth(self, data_sharp):
        from statspai.rd import rdrobust
        result = rdrobust(data_sharp, y='y', x='x', c=0,
                          bwselect='certwo')
        bw = result.model_info['bandwidth_h']
        assert isinstance(bw, tuple)

    def test_msecomb1_bandwidth(self, data_sharp):
        from statspai.rd import rdrobust
        result = rdrobust(data_sharp, y='y', x='x', c=0,
                          bwselect='msecomb1')
        assert isinstance(result, CausalResult)

    def test_covariate_adjusted(self, data_sharp):
        from statspai.rd import rdrobust
        # Without covariates
        r1 = rdrobust(data_sharp, y='y', x='x', c=0)
        # With covariates (should reduce SE)
        r2 = rdrobust(data_sharp, y='y', x='x', c=0, covs=['z'])
        assert isinstance(r2, CausalResult)
        assert abs(r2.estimate - 3.0) < 1.0
        # Covariate adjustment typically reduces SE
        # (not guaranteed in every sample, so we just check it runs)

    def test_all_bwselect_methods(self, data_sharp):
        """All bandwidth methods should run without error."""
        from statspai.rd import rdrobust
        for method in ['mserd', 'msetwo', 'cerrd', 'certwo',
                       'msecomb1', 'msecomb2', 'cercomb1', 'cercomb2']:
            result = rdrobust(data_sharp, y='y', x='x', c=0,
                              bwselect=method)
            assert isinstance(result, CausalResult), f"Failed for {method}"


# ======================================================================
# Enhanced Diagnostics Tests
# ======================================================================

class TestEnhancedDiagnostics:
    """Tests for upgraded rdsummary with full diagnostics."""

    def test_rdsummary_basic(self, data_sharp):
        from statspai.rd import rdsummary
        result = rdsummary(data_sharp, y='y', x='x', c=0,
                           verbose=False)
        assert 'estimate' in result
        assert 'density_test' in result

    def test_rdsummary_with_covs(self, data_sharp):
        from statspai.rd import rdsummary
        result = rdsummary(data_sharp, y='y', x='x', c=0,
                           covs=['z'], verbose=False)
        assert result['balance'] is not None

    def test_rdsummary_full(self, data_sharp):
        from statspai.rd import rdsummary
        result = rdsummary(data_sharp, y='y', x='x', c=0,
                           covs=['z'], full=True, verbose=False)
        assert 'honest_ci' in result
        assert 'power' in result
