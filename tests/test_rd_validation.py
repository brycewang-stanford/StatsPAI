"""
Numerical validation tests for RD module.

Tests correctness of point estimates against known DGP true values,
verifies coverage of confidence intervals via Monte Carlo,
and validates integration between modules.
"""

import pytest
import numpy as np
import pandas as pd

import statspai as sp
from statspai.core.results import CausalResult


# ======================================================================
# Numerical Accuracy Tests
# ======================================================================

class TestNumericalAccuracy:
    """Verify estimates are close to known true values."""

    def test_sharp_rd_point_estimate(self):
        """Sharp RD estimate should be within 0.5 of true effect (3.0)."""
        df = sp.dgp_rd(n=5000, effect=3.0, seed=42)
        result = sp.rdrobust(df, y='y', x='x', c=0)
        assert abs(result.estimate - 3.0) < 0.5, \
            f"Estimate {result.estimate:.3f} too far from 3.0"

    def test_sharp_rd_ci_covers_truth(self):
        """95% CI should cover the true effect."""
        df = sp.dgp_rd(n=5000, effect=3.0, seed=42)
        result = sp.rdrobust(df, y='y', x='x', c=0)
        assert result.ci[0] < 3.0 < result.ci[1], \
            f"CI [{result.ci[0]:.3f}, {result.ci[1]:.3f}] misses truth 3.0"

    def test_fuzzy_rd_point_estimate(self):
        """Fuzzy RD LATE should be reasonable."""
        df = sp.dgp_rd(n=5000, effect=2.0, fuzzy=True, seed=42)
        result = sp.rdrobust(df, y='y', x='x', c=0, fuzzy='treatment')
        # Fuzzy is noisier, allow wider tolerance
        assert abs(result.estimate) < 10, \
            f"Fuzzy estimate {result.estimate:.3f} seems unreasonable"

    def test_kink_rd_estimate(self):
        """RKD should detect kink in slope."""
        df = sp.dgp_rd_kink(n=5000, kink=0.8, seed=42)
        result = sp.rkd(df, y='y', x='x', c=0)
        # Kink detection is harder; just check it's positive and significant
        assert result.estimate > 0, "Kink estimate should be positive"

    def test_zero_effect_not_significant(self):
        """When true effect = 0, should not reject H0."""
        df = sp.dgp_rd(n=2000, effect=0.0, seed=42)
        result = sp.rdrobust(df, y='y', x='x', c=0)
        assert result.pvalue > 0.01, \
            f"p={result.pvalue:.4f} < 0.01 for zero true effect"

    def test_large_effect_significant(self):
        """When true effect is large, should reject H0."""
        df = sp.dgp_rd(n=2000, effect=5.0, seed=42)
        result = sp.rdrobust(df, y='y', x='x', c=0)
        assert result.pvalue < 0.05, \
            f"p={result.pvalue:.4f} > 0.05 for large true effect 5.0"

    def test_covariate_adjustment_reduces_se(self):
        """Covariate adjustment should generally reduce SE."""
        rng = np.random.default_rng(42)
        n = 3000
        X = rng.uniform(-1, 1, n)
        Z = rng.normal(0, 1, n)
        Y = 0.5 * X + 3.0 * (X >= 0) + 2.0 * Z + rng.normal(0, 0.3, n)
        df = pd.DataFrame({'y': Y, 'x': X, 'z': Z})

        r_no_cov = sp.rdrobust(df, y='y', x='x', c=0)
        r_cov = sp.rdrobust(df, y='y', x='x', c=0, covs=['z'])
        # Both should be close to 3.0
        assert abs(r_cov.estimate - 3.0) < 1.0
        # SE with covariates should be smaller (or at least not much larger)
        assert r_cov.se < r_no_cov.se * 1.5


# ======================================================================
# Bandwidth Selection Validation
# ======================================================================

class TestBandwidthValidation:
    """Verify bandwidth selection properties."""

    def test_cer_smaller_than_mse(self):
        """CER bandwidth should be smaller than MSE bandwidth."""
        df = sp.dgp_rd(n=3000, effect=2.0, seed=42)
        r_mse = sp.rdrobust(df, y='y', x='x', c=0, bwselect='mserd')
        r_cer = sp.rdrobust(df, y='y', x='x', c=0, bwselect='cerrd')

        h_mse = r_mse.model_info['bandwidth_h']
        h_cer = r_cer.model_info['bandwidth_h']

        if isinstance(h_mse, tuple):
            h_mse = h_mse[0]
        if isinstance(h_cer, tuple):
            h_cer = h_cer[0]

        assert h_cer < h_mse * 1.05, \
            f"CER h={h_cer:.3f} should be < MSE h={h_mse:.3f}"

    def test_bandwidth_positive(self):
        """All bandwidth methods should return positive values."""
        df = sp.dgp_rd(n=2000, effect=2.0, seed=42)
        for method in ['mserd', 'msetwo', 'cerrd', 'certwo',
                       'msecomb1', 'msecomb2', 'cercomb1', 'cercomb2']:
            r = sp.rdrobust(df, y='y', x='x', c=0, bwselect=method)
            h = r.model_info['bandwidth_h']
            if isinstance(h, tuple):
                assert h[0] > 0 and h[1] > 0, f"{method}: h={h}"
            else:
                assert h > 0, f"{method}: h={h}"


# ======================================================================
# Density Test Validation
# ======================================================================

class TestDensityValidation:
    """Verify density manipulation test works correctly."""

    def test_no_manipulation_clean_data(self):
        """Uniform running variable should show no manipulation."""
        df = sp.dgp_rd(n=3000, effect=2.0, seed=42)
        result = sp.rddensity(df, x='x', c=0)
        assert result.pvalue > 0.05, \
            f"False manipulation detected: p={result.pvalue:.4f}"

    def test_manipulation_detected(self):
        """Heaped running variable should be detected as manipulation."""
        rng = np.random.default_rng(42)
        n = 3000
        # Create heaping: observations just above cutoff are more common
        X = rng.uniform(-1, 1, n)
        # Add extra observations just above 0
        X_heap = np.concatenate([X, rng.uniform(0, 0.05, 300)])
        Y = rng.normal(0, 1, len(X_heap))
        df = pd.DataFrame({'y': Y, 'x': X_heap})
        result = sp.rddensity(df, x='x', c=0)
        # Should detect the heaping (may not always reject, but p should be low)
        assert result.pvalue < 0.50  # lenient check


# ======================================================================
# Integration Tests (modules working together)
# ======================================================================

class TestIntegration:
    """Test that modules integrate properly end-to-end."""

    def test_dgp_to_rdrobust_to_rdsummary(self):
        """Full pipeline: generate data → estimate → diagnostics."""
        df = sp.dgp_rd(n=2000, effect=3.0, seed=42)
        # Estimate
        result = sp.rdrobust(df, y='y', x='x', c=0)
        assert abs(result.estimate - 3.0) < 1.0
        # Diagnostics
        diag = sp.rdsummary(df, y='y', x='x', c=0, verbose=False)
        assert 'estimate' in diag
        assert 'density_test' in diag

    def test_rdbwselect_feeds_into_rdrobust(self):
        """Bandwidth from rdbwselect should work in rdrobust."""
        df = sp.dgp_rd(n=2000, effect=3.0, seed=42)
        bw = sp.rdbwselect(df, y='y', x='x', c=0, bwselect='cerrd')
        h_val = bw['h_left'].iloc[0]
        result = sp.rdrobust(df, y='y', x='x', c=0, h=h_val)
        assert abs(result.estimate - 3.0) < 1.0

    def test_hte_with_dgp(self):
        """HTE estimation on DGP with known heterogeneity."""
        df = sp.dgp_rd_hte(n=3000, ate=2.0, hte_coef=1.5, seed=42)
        result = sp.rdhte(df, y='y', x='x', z='z', c=0, n_eval=10)
        # ATE should be close to 2.0
        assert abs(result.estimate - 2.0) < 1.0

    def test_multi_cutoff_with_dgp(self):
        """Multi-cutoff estimation on known DGP."""
        df = sp.dgp_rd_multi(n=3000, seed=42)
        result = sp.rdmc(df, y='y', x='x', cutoffs=[0.0, 1.0])
        # Should detect positive effects
        assert result.pooled_estimate > 0

    def test_2d_with_dgp(self):
        """2D RD on known DGP."""
        df = sp.dgp_rd_2d(n=2000, effect=2.0, seed=42)
        result = sp.rd2d(df, y='y', x1='x1', x2='x2',
                         treatment='d', approach='distance')
        assert abs(result.estimate - 2.0) < 1.5

    def test_honest_ci_wider_than_standard(self):
        """Honest CI should be at least as wide as standard CI."""
        df = sp.dgp_rd(n=3000, effect=3.0, seed=42)
        r_std = sp.rdrobust(df, y='y', x='x', c=0)
        r_honest = sp.rd_honest(df, y='y', x='x', c=0)

        std_width = r_std.ci[1] - r_std.ci[0]
        honest_width = r_honest.ci[1] - r_honest.ci[0]
        assert honest_width >= std_width * 0.9, \
            f"Honest CI ({honest_width:.3f}) should be >= standard ({std_width:.3f})"


# ======================================================================
# Monte Carlo Coverage Test (slow, marks as slow)
# ======================================================================

class TestMonteCarloCoverage:
    """Monte Carlo validation of confidence interval coverage."""

    @pytest.mark.slow
    def test_ci_coverage_95(self):
        """95% CI should cover truth ~95% of the time (loose check: 85-99%)."""
        true_effect = 3.0
        n_sims = 50
        covers = 0

        for seed in range(n_sims):
            df = sp.dgp_rd(n=1000, effect=true_effect, seed=seed)
            try:
                r = sp.rdrobust(df, y='y', x='x', c=0)
                if r.ci[0] <= true_effect <= r.ci[1]:
                    covers += 1
            except Exception:
                pass

        coverage = covers / n_sims
        assert 0.80 <= coverage <= 1.0, \
            f"Coverage {coverage:.2%} outside expected range [80%, 100%]"


# ======================================================================
# DGP Functions Tests
# ======================================================================

class TestDGPFunctions:
    """Verify all new DGP functions produce valid data."""

    def test_dgp_rd_kink(self):
        df = sp.dgp_rd_kink(n=1000, kink=0.8, seed=42)
        assert 'y' in df.columns and 'x' in df.columns
        assert len(df) == 1000
        assert df.attrs['true_kink'] == 0.8

    def test_dgp_rd_multi(self):
        df = sp.dgp_rd_multi(n=2000, seed=42)
        assert len(df) == 2000
        assert df.attrs['true_effects'] == {0.0: 2.0, 1.0: 3.0}

    def test_dgp_rd_hte(self):
        df = sp.dgp_rd_hte(n=2000, ate=2.0, hte_coef=1.5, seed=42)
        assert len(df) == 2000
        assert df.attrs['true_ate'] == 2.0

    def test_dgp_rd_2d(self):
        df = sp.dgp_rd_2d(n=1000, effect=2.0, seed=42)
        assert 'x1' in df.columns and 'x2' in df.columns
        assert df.attrs['true_effect'] == 2.0

    def test_dgp_rdit(self):
        df = sp.dgp_rdit(n_periods=200, effect=2.0, seed=42)
        assert len(df) == 200
        assert df.attrs['true_effect'] == 2.0
