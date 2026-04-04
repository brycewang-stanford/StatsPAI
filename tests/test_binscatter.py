"""
Tests for binscatter — binned scatter plots with residualization.

Validates both the computation (residualization, binning) and the
plotting output.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from statspai.plots.binscatter import binscatter


@pytest.fixture
def sample_data():
    """Linear relationship with controls. True slope ≈ 2.0."""
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(10, 3, n)
    z1 = rng.normal(0, 1, n)  # confounding control
    z2 = rng.normal(0, 1, n)
    y = 5 + 2.0 * x + 3 * z1 + 1.5 * z2 + rng.normal(0, 2, n)
    return pd.DataFrame({
        'y': y, 'x': x, 'z1': z1, 'z2': z2,
        'female': rng.choice([0, 1], n),
        'firm_id': rng.choice(range(50), n),
        'year': rng.choice([2018, 2019, 2020, 2021], n),
        'weight': rng.uniform(0.5, 2.0, n),
    })


class TestBasicBinscatter:
    """Test core binscatter functionality."""

    def test_basic_plot(self, sample_data):
        """Should produce a figure without error."""
        fig, ax, bins = binscatter(sample_data, y='y', x='x')
        assert fig is not None
        assert ax is not None
        assert len(bins) > 0

    def test_returns_bin_data(self, sample_data):
        """Should return DataFrame with x_mean, y_mean, n."""
        _, _, bins = binscatter(sample_data, y='y', x='x')
        assert 'x_mean' in bins.columns
        assert 'y_mean' in bins.columns
        assert 'n' in bins.columns
        assert bins['n'].sum() == len(sample_data)

    def test_default_n_bins(self, sample_data):
        """Default bins should be reasonable."""
        _, _, bins = binscatter(sample_data, y='y', x='x')
        assert 5 <= len(bins) <= 20

    def test_custom_n_bins(self, sample_data):
        """Custom bin count should be respected."""
        _, _, bins = binscatter(sample_data, y='y', x='x', n_bins=10)
        assert len(bins) == 10

    def test_positive_slope(self, sample_data):
        """Bin means should show positive relationship (true slope=2)."""
        _, _, bins = binscatter(sample_data, y='y', x='x')
        # First bin mean should be lower than last
        assert bins.iloc[0]['y_mean'] < bins.iloc[-1]['y_mean']


class TestResidualizedBinscatter:
    """Test residualization (controlling for variables)."""

    def test_with_controls(self, sample_data):
        """Controls should not break the plot."""
        fig, ax, bins = binscatter(
            sample_data, y='y', x='x', controls=['z1', 'z2'])
        assert fig is not None
        assert len(bins) > 0

    def test_controls_change_slope(self, sample_data):
        """Partialling out confounders may change the visual slope."""
        _, _, bins_raw = binscatter(sample_data, y='y', x='x')
        _, _, bins_ctrl = binscatter(
            sample_data, y='y', x='x', controls=['z1', 'z2'])
        # Both should have data
        assert len(bins_raw) > 0
        assert len(bins_ctrl) > 0

    def test_absorb_fe(self, sample_data):
        """Fixed effects absorption should work."""
        fig, ax, bins = binscatter(
            sample_data, y='y', x='x', absorb=['firm_id'])
        assert fig is not None
        assert len(bins) > 0

    def test_absorb_multiple_fe(self, sample_data):
        """Multiple FE absorption."""
        fig, ax, bins = binscatter(
            sample_data, y='y', x='x',
            absorb=['firm_id', 'year'])
        assert fig is not None

    def test_controls_and_absorb(self, sample_data):
        """Both controls and absorb together."""
        fig, ax, bins = binscatter(
            sample_data, y='y', x='x',
            controls=['z1'], absorb=['firm_id'])
        assert fig is not None


class TestByGroup:
    """Test group-level binscatter (by= parameter)."""

    def test_by_group(self, sample_data):
        """Should produce separate series per group."""
        fig, ax, bins = binscatter(
            sample_data, y='y', x='x', by='female')
        assert fig is not None
        assert 'group' in bins.columns
        assert set(bins['group'].unique()) == {0, 1}

    def test_by_with_controls(self, sample_data):
        fig, ax, bins = binscatter(
            sample_data, y='y', x='x',
            controls=['z1'], by='female')
        assert fig is not None


class TestConfidenceIntervals:
    """Test CI computation."""

    def test_ci_columns(self, sample_data):
        """With ci=True, should have ci_lower/ci_upper."""
        _, _, bins = binscatter(
            sample_data, y='y', x='x', ci=True)
        assert 'ci_lower' in bins.columns
        assert 'ci_upper' in bins.columns

    def test_ci_contains_mean(self, sample_data):
        """CI should contain the bin mean."""
        _, _, bins = binscatter(
            sample_data, y='y', x='x', ci=True)
        for _, row in bins.iterrows():
            assert row['ci_lower'] <= row['y_mean'] <= row['ci_upper']


class TestFitOptions:
    """Test different fit overlay options."""

    def test_linear_fit(self, sample_data):
        fig, ax, _ = binscatter(sample_data, y='y', x='x', fit='linear')
        assert fig is not None

    def test_quadratic_fit(self, sample_data):
        fig, ax, _ = binscatter(sample_data, y='y', x='x', fit='quadratic')
        assert fig is not None

    def test_no_fit(self, sample_data):
        fig, ax, _ = binscatter(sample_data, y='y', x='x', fit='none')
        assert fig is not None

    def test_fit_on_raw(self, sample_data):
        fig, ax, _ = binscatter(
            sample_data, y='y', x='x', fit='linear', fit_on_raw=True)
        assert fig is not None


class TestBinTypes:
    """Test quantile vs equal-width bins."""

    def test_quantile_bins(self, sample_data):
        """Quantile bins should have roughly equal obs per bin."""
        _, _, bins = binscatter(
            sample_data, y='y', x='x', n_bins=10, quantiles=True)
        # Each bin should have ~200 obs (2000/10), allow 50% tolerance
        assert bins['n'].min() > 100

    def test_equal_width_bins(self, sample_data):
        _, _, bins = binscatter(
            sample_data, y='y', x='x', n_bins=10, quantiles=False)
        assert len(bins) > 0


class TestWeights:
    """Test weighted binscatter."""

    def test_with_weights(self, sample_data):
        fig, ax, bins = binscatter(
            sample_data, y='y', x='x', weights='weight')
        assert fig is not None


class TestCustomization:
    """Test visual customization options."""

    def test_custom_labels(self, sample_data):
        fig, ax, _ = binscatter(
            sample_data, y='y', x='x',
            title='Education and Wages',
            x_label='Years of Education',
            y_label='Log Monthly Wage')
        assert ax.get_xlabel() == 'Years of Education'
        assert ax.get_ylabel() == 'Log Monthly Wage'

    def test_custom_figsize(self, sample_data):
        fig, ax, _ = binscatter(
            sample_data, y='y', x='x', figsize=(12, 4))
        assert fig is not None


class TestIntegration:
    def test_top_level_import(self):
        import statspai as sp
        assert hasattr(sp, 'binscatter')

    def test_top_level_call(self, sample_data):
        import statspai as sp
        fig, ax, bins = sp.binscatter(sample_data, y='y', x='x')
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
