"""
Tests for diagnostics: Oster bounds and McCrary density test.

Oster tests use analytically verifiable examples.
McCrary tests use simulated data with/without manipulation.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.diagnostics import oster_bounds, mccrary_test
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def oster_data():
    """Data where controls reduce the coefficient and increase R².

    DGP: Y = 1 + 2*D + 3*X + ε, where D is correlated with X.
    Short reg (no X): β̊ > 2 (upward biased due to omission)
    Long reg (with X): β̃ ≈ 2 (unbiased)
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.normal(0, 1, n)
    D = 0.5 * X + rng.normal(0, 1, n)  # D correlated with X
    Y = 1 + 2 * D + 3 * X + rng.normal(0, 1, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x': X})


@pytest.fixture
def rd_data_clean():
    """RD data with smooth density (no manipulation). Uniform X."""
    rng = np.random.default_rng(42)
    n = 5000
    X = rng.uniform(-2, 2, n)
    return pd.DataFrame({'x': X})


@pytest.fixture
def rd_data_manipulated():
    """RD data with density bump just above cutoff (manipulation).

    Extra observations bunched slightly above 0.
    """
    rng = np.random.default_rng(42)
    n = 5000
    X = rng.uniform(-2, 2, n)
    # Add 1000 extra observations just above cutoff (manipulation)
    X_extra = rng.uniform(0, 0.3, 1000)
    X_all = np.concatenate([X, X_extra])
    return pd.DataFrame({'x': X_all})


# ======================================================================
# Oster Bounds Tests
# ======================================================================

class TestOsterBounds:
    """Tests for Oster (2019) coefficient stability bounds."""

    def test_from_data(self, oster_data):
        """Should run from data and return valid result."""
        result = oster_bounds(
            oster_data, y='y', treat='d', controls=['x'],
        )
        assert 'delta_for_zero' in result
        assert 'beta_adjusted' in result
        assert 'identified_set' in result
        assert 'robust' in result
        assert isinstance(result['interpretation'], str)

    def test_from_statistics(self):
        """Should run from pre-computed statistics."""
        result = oster_bounds(
            beta_short=3.0, r2_short=0.20,
            beta_long=2.0, r2_long=0.50,
        )
        assert 'delta_for_zero' in result
        assert result['beta_short'] == 3.0
        assert result['beta_long'] == 2.0

    def test_delta_star_formula(self):
        """Verify δ* formula against manual calculation.

        β̊=3, R̊²=0.2, β̃=2, R̃²=0.5, R_max=0.65 (=1.3×0.5)
        δ* = β̃ × (R_max - R̃²) / ((β̊ - β̃) × (R̃² - R̊²))
           = 2 × (0.65 - 0.5) / ((3 - 2) × (0.5 - 0.2))
           = 2 × 0.15 / (1 × 0.3)
           = 0.3 / 0.3
           = 1.0
        """
        result = oster_bounds(
            beta_short=3.0, r2_short=0.2,
            beta_long=2.0, r2_long=0.5,
            r_max=0.65,
        )
        assert abs(result['delta_for_zero'] - 1.0) < 1e-10

    def test_beta_adjusted_formula(self):
        """Verify β*(δ=1) against manual calculation.

        β̊=3, R̊²=0.2, β̃=2, R̃²=0.5, R_max=0.65, δ=1
        β* = β̃ - δ × (β̊ - β̃) × (R̃² - R̊²) / (R_max - R̃²)
           = 2 - 1 × (3 - 2) × (0.5 - 0.2) / (0.65 - 0.5)
           = 2 - 1 × 1 × 0.3 / 0.15
           = 2 - 2 = 0
        """
        result = oster_bounds(
            beta_short=3.0, r2_short=0.2,
            beta_long=2.0, r2_long=0.5,
            r_max=0.65, delta=1.0,
        )
        assert abs(result['beta_adjusted'] - 0.0) < 1e-10

    def test_robust_when_delta_gt_1(self):
        """When δ* > 1, result should be flagged as robust."""
        result = oster_bounds(
            beta_short=3.0, r2_short=0.10,
            beta_long=2.5, r2_long=0.50,
        )
        # Movement is small relative to R² gain → δ* should be > 1
        assert result['delta_for_zero'] > 1
        assert result['robust'] is True

    def test_sensitive_when_delta_lt_1(self):
        """When δ* < 1 and β_adjusted crosses zero, not robust."""
        result = oster_bounds(
            beta_short=5.0, r2_short=0.05,
            beta_long=1.0, r2_long=0.50,
            r_max=0.65, delta=1.0,
        )
        # Large movement + small R² room → adjusted β overshoots zero
        assert result['beta_adjusted'] < 0
        assert result['robust'] is False

    def test_identified_set(self):
        """Identified set should be [min(β_adj, β_long), max(...)]."""
        result = oster_bounds(
            beta_short=3.0, r2_short=0.2,
            beta_long=2.0, r2_long=0.5,
            r_max=0.8, delta=1.0,
        )
        lo, hi = result['identified_set']
        assert lo <= hi
        assert lo <= result['beta_long'] <= hi or lo <= result['beta_adjusted'] <= hi

    def test_custom_r_max(self):
        """Custom R_max should be used."""
        result = oster_bounds(
            beta_short=3.0, r2_short=0.2,
            beta_long=2.0, r2_long=0.5,
            r_max=0.9,
        )
        assert result['r_max'] == 0.9

    def test_controls_reduce_coefficient(self, oster_data):
        """With OVB, β_short > β_long (controls absorb confounding)."""
        result = oster_bounds(
            oster_data, y='y', treat='d', controls=['x'],
        )
        assert result['beta_short'] > result['beta_long']

    def test_data_regression_consistency(self, oster_data):
        """β_long should be close to true value of 2."""
        result = oster_bounds(
            oster_data, y='y', treat='d', controls=['x'],
        )
        assert abs(result['beta_long'] - 2.0) < 0.2

    def test_missing_inputs_raises(self):
        """Should raise error if inputs are incomplete."""
        with pytest.raises(ValueError):
            oster_bounds(beta_short=2.0)  # missing others

    def test_interpretation_string(self):
        result = oster_bounds(
            beta_short=3.0, r2_short=0.10,
            beta_long=2.5, r2_long=0.50,
        )
        assert 'ROBUST' in result['interpretation'] or 'SENSITIVE' in result['interpretation']


# ======================================================================
# McCrary Test
# ======================================================================

class TestMcCraryTest:
    """Tests for McCrary (2008) density discontinuity test."""

    def test_clean_density_no_rejection(self, rd_data_clean):
        """Clean data (uniform) should NOT reject H0."""
        result = mccrary_test(rd_data_clean, x='x', c=0)
        assert isinstance(result, CausalResult)
        assert result.pvalue > 0.01  # should not reject

    def test_manipulated_density_rejection(self, rd_data_manipulated):
        """Manipulated data should reject H0 (density jump)."""
        result = mccrary_test(rd_data_manipulated, x='x', c=0)
        assert isinstance(result, CausalResult)
        # The extra observations above cutoff should create a detectable jump
        assert result.pvalue < 0.1  # should reject at 10%

    def test_returns_causal_result(self, rd_data_clean):
        result = mccrary_test(rd_data_clean, x='x')
        assert result.method == 'McCrary (2008) Density Test'
        assert result.estimand == 'Log Density Ratio'
        assert result.se > 0

    def test_density_estimates(self, rd_data_clean):
        """Density estimates should be positive."""
        result = mccrary_test(rd_data_clean, x='x')
        assert result.model_info['density_left'] > 0
        assert result.model_info['density_right'] > 0

    def test_symmetric_density_near_zero(self, rd_data_clean):
        """For uniform data, log ratio should be near zero."""
        result = mccrary_test(rd_data_clean, x='x', c=0)
        assert abs(result.estimate) < 0.5

    def test_nonzero_cutoff(self):
        """Should work with non-zero cutoff."""
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 10, 3000)
        df = pd.DataFrame({'x': X})
        result = mccrary_test(df, x='x', c=5)
        assert isinstance(result, CausalResult)

    def test_custom_bandwidth(self, rd_data_clean):
        result = mccrary_test(rd_data_clean, x='x', bw=0.5)
        assert abs(result.model_info['bandwidth'] - 0.5) < 1e-6

    def test_custom_bins(self, rd_data_clean):
        result = mccrary_test(rd_data_clean, x='x', n_bins=20)
        assert result.model_info['n_bins'] == 20

    def test_summary(self, rd_data_clean):
        result = mccrary_test(rd_data_clean, x='x')
        s = result.summary()
        assert 'McCrary' in s
        assert 'Log Density' in s

    def test_cite(self, rd_data_clean):
        result = mccrary_test(rd_data_clean, x='x')
        bib = result.cite()
        assert 'mccrary' in bib.lower()


# ======================================================================
# Integration
# ======================================================================

class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'oster_bounds')
        assert hasattr(sp, 'mccrary_test')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
