"""
Tests for Panel regression module (wrapping linearmodels).
"""

import pytest
import numpy as np
import pandas as pd
from statspai.panel import panel, PanelRegression
from statspai.core.results import EconometricResults


@pytest.fixture
def panel_data():
    """
    Simulated panel: 50 entities, 10 periods.
    DGP: y_it = alpha_i + 2*x1_it + 3*x2_it + eps_it
    """
    rng = np.random.default_rng(42)
    n_entities = 50
    n_periods = 10

    records = []
    for i in range(n_entities):
        alpha_i = rng.normal(5, 2)  # entity fixed effect
        for t in range(1, n_periods + 1):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            eps = rng.normal(0, 0.5)
            y = alpha_i + 2 * x1 + 3 * x2 + eps
            records.append({
                'entity': f'e{i}',
                'time': t,
                'y': y,
                'x1': x1,
                'x2': x2,
            })

    return pd.DataFrame(records)


class TestFixedEffects:

    def test_fe_basic(self, panel_data):
        """FE should recover slopes ≈ 2, 3 (wipes out alpha_i)."""
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='fe')

        assert isinstance(result, EconometricResults)
        assert abs(result.params['x1'] - 2.0) < 0.3
        assert abs(result.params['x2'] - 3.0) < 0.3

    def test_fe_robust(self, panel_data):
        """FE with robust SE should work."""
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time',
                       method='fe', robust='robust')
        assert result is not None

    def test_fe_clustered(self, panel_data):
        """FE with clustered SE by entity."""
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time',
                       method='fe', cluster='entity')
        assert result is not None


class TestRandomEffects:

    def test_re_basic(self, panel_data):
        """RE should also estimate slopes approximately."""
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='re')

        assert isinstance(result, EconometricResults)
        assert abs(result.params['x1'] - 2.0) < 0.5
        assert abs(result.params['x2'] - 3.0) < 0.5


class TestPooledOLS:

    def test_pooled_basic(self, panel_data):
        """Pooled OLS should work."""
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='pooled')

        assert isinstance(result, EconometricResults)
        assert len(result.params) == 3  # const + x1 + x2


class TestFirstDifference:

    def test_fd_basic(self, panel_data):
        """First difference should work."""
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='fd')

        assert isinstance(result, EconometricResults)


class TestBetween:

    def test_between_basic(self, panel_data):
        """Between estimator should work."""
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='be')

        assert isinstance(result, EconometricResults)


class TestPanelGeneral:

    def test_summary(self, panel_data):
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='fe')
        s = result.summary()
        assert isinstance(s, str)
        assert 'Panel FE' in s

    def test_diagnostics(self, panel_data):
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='fe')
        assert 'R-squared' in result.diagnostics

    def test_residuals(self, panel_data):
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='fe')
        resid = result.residuals()
        assert resid is not None
        assert len(resid) == len(panel_data)

    def test_fitted_values(self, panel_data):
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='fe')
        fitted = result.fitted_values()
        assert fitted is not None

    def test_repr(self, panel_data):
        result = panel(panel_data, "y ~ x1 + x2",
                       entity='entity', time='time', method='fe')
        assert 'Panel FE' in repr(result)

    # --- Error handling ---

    def test_invalid_method(self, panel_data):
        with pytest.raises(ValueError, match="method must be"):
            panel(panel_data, "y ~ x1", entity='entity',
                  time='time', method='invalid')

    def test_missing_column(self, panel_data):
        with pytest.raises(ValueError, match="not found"):
            panel(panel_data, "y ~ nonexistent", entity='entity',
                  time='time', method='fe')

    def test_no_tilde(self, panel_data):
        with pytest.raises(ValueError, match="must contain"):
            panel(panel_data, "y x1", entity='entity',
                  time='time', method='fe')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
