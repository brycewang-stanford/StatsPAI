"""
Tests for Causal Impact module.

Uses simulated time-series data with known intervention effects.
"""

import pytest
import numpy as np
import pandas as pd
from statspai.causal_impact import causal_impact, CausalImpactEstimator
from statspai.core.results import CausalResult


@pytest.fixture
def ts_with_effect():
    """
    Time series with intervention at t=50.
    DGP: Y_t = 10 + 2*X_t + eps_t + 5*(t >= 50)
    True average causal effect = 5.0.
    Covariates absorb cross-sectional variation, no time trend.
    """
    rng = np.random.default_rng(42)
    n = 80
    t = np.arange(1, n + 1)
    X = rng.normal(3, 1, n)
    eps = rng.normal(0, 0.5, n)

    Y = 10 + 2 * X + eps
    Y[t >= 50] += 5.0

    return pd.DataFrame({
        'time': t,
        'y': Y,
        'x': X,
    })


@pytest.fixture
def ts_no_effect():
    """Time series with no intervention effect."""
    rng = np.random.default_rng(99)
    n = 80
    t = np.arange(1, n + 1)
    X = rng.normal(2, 1, n)
    Y = 5 + 1.5 * X + rng.normal(0, 0.3, n)

    return pd.DataFrame({'time': t, 'y': Y, 'x': X})


@pytest.fixture
def ts_no_covariates():
    """
    Time series without covariates.
    DGP: Y_t = 20 + eps_t + 3*(t >= 30)
    True effect = 3.0.
    """
    rng = np.random.default_rng(42)
    n = 60
    t = np.arange(1, n + 1)
    eps = rng.normal(0, 0.5, n)
    Y = 20 + eps
    Y[t >= 30] += 3.0

    return pd.DataFrame({'time': t, 'y': Y})


class TestCausalImpactBasic:

    def test_basic_with_covariates(self, ts_with_effect):
        """Should detect intervention effect ≈ 5.0."""
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 5.0) < 2.0, (
            f"Causal Impact estimate = {result.estimate:.2f}, expected ≈ 5.0"
        )

    def test_significance(self, ts_with_effect):
        """Effect should be statistically significant."""
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )
        assert result.pvalue < 0.05

    def test_no_covariates(self, ts_no_covariates):
        """Should work without covariates."""
        result = causal_impact(
            ts_no_covariates, y='y', time='time',
            intervention_time=30,
        )

        assert isinstance(result, CausalResult)
        assert abs(result.estimate - 3.0) < 2.0

    def test_no_effect(self, ts_no_effect):
        """With no real effect, estimate should be near zero."""
        result = causal_impact(
            ts_no_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )

        assert abs(result.estimate) < 3.0, (
            f"Null effect estimate = {result.estimate:.2f}, should be ≈ 0"
        )


class TestCausalImpactOutput:

    def test_detail_table(self, ts_with_effect):
        """Detail table should have full time series."""
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )

        detail = result.detail
        assert detail is not None
        assert len(detail) == 80
        assert 'actual' in detail.columns
        assert 'predicted' in detail.columns
        assert 'effect' in detail.columns
        assert 'post_intervention' in detail.columns

    def test_model_info(self, ts_with_effect):
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )

        info = result.model_info
        assert 'intervention_time' in info
        assert info['intervention_time'] == 50
        assert 'n_pre' in info
        assert 'n_post' in info
        assert 'avg_effect' in info
        assert 'total_effect' in info
        assert 'relative_effect' in info
        assert 'cumulative_effect' in info

    def test_cumulative_effect(self, ts_with_effect):
        """Cumulative effect should grow over post-period."""
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )

        cum = result.model_info['cumulative_effect']
        assert len(cum) == result.model_info['n_post']
        # Cumulative should be monotonically increasing (positive effect)
        assert cum[-1] > cum[0]

    def test_summary(self, ts_with_effect):
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )
        s = result.summary()
        assert 'Causal Impact' in s

    def test_citation(self, ts_with_effect):
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )
        assert 'brodersen' in result.cite().lower()

    def test_repr(self, ts_with_effect):
        result = causal_impact(
            ts_with_effect, y='y', time='time',
            intervention_time=50, covariates=['x'],
        )
        assert 'CausalResult' in repr(result)


class TestCausalImpactErrors:

    def test_missing_column(self, ts_with_effect):
        with pytest.raises(ValueError, match="not found"):
            causal_impact(ts_with_effect, y='nonexistent',
                          time='time', intervention_time=50)

    def test_too_few_pre_periods(self):
        df = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'y': [10, 11, 12, 13, 14],
        })
        with pytest.raises(ValueError, match="pre-intervention"):
            causal_impact(df, y='y', time='time', intervention_time=3)

    def test_no_post_period(self):
        df = pd.DataFrame({
            'time': [1, 2, 3, 4, 5],
            'y': [10, 11, 12, 13, 14],
        })
        with pytest.raises(ValueError):
            causal_impact(df, y='y', time='time', intervention_time=100)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
