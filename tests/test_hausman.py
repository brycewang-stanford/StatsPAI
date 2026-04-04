"""
Tests for Hausman FE vs RE test.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.diagnostics.hausman import hausman_test


@pytest.fixture
def panel_fe_needed():
    """Panel where FE is needed (unit effects correlated with X)."""
    rng = np.random.default_rng(42)
    N, T = 50, 8
    rows = []
    for i in range(N):
        alpha_i = rng.normal(0, 2)
        x_base = alpha_i * 0.5  # X correlated with alpha → RE inconsistent
        for t in range(T):
            x = x_base + rng.normal(0, 1)
            y = 1 + 2 * x + alpha_i + rng.normal(0, 0.5)
            rows.append({'id': i, 'time': t, 'y': y, 'x': x})
    return pd.DataFrame(rows)


@pytest.fixture
def panel_re_ok():
    """Panel where RE is fine (unit effects uncorrelated with X)."""
    rng = np.random.default_rng(42)
    N, T = 50, 8
    rows = []
    for i in range(N):
        alpha_i = rng.normal(0, 1)
        for t in range(T):
            x = rng.normal(0, 1)  # X independent of alpha
            y = 1 + 2 * x + alpha_i + rng.normal(0, 1)
            rows.append({'id': i, 'time': t, 'y': y, 'x': x})
    return pd.DataFrame(rows)


class TestHausman:
    def test_basic_run(self, panel_fe_needed):
        result = hausman_test(panel_fe_needed, y='y', x=['x'],
                              id='id', time='time')
        assert 'statistic' in result
        assert 'pvalue' in result
        assert 'recommendation' in result

    def test_rejects_for_fe(self):
        """When FE is needed, Hausman should reject H0."""
        rng = np.random.default_rng(42)
        N, T = 100, 10
        rows = []
        for i in range(N):
            alpha_i = rng.normal(0, 3)
            for t in range(T):
                x = alpha_i + rng.normal(0, 0.5)  # strongly correlated with FE
                y = 1 + 2 * x + alpha_i + rng.normal(0, 0.5)
                rows.append({'id': i, 'time': t, 'y': y, 'x': x})
        df = pd.DataFrame(rows)
        result = hausman_test(df, y='y', x=['x'], id='id', time='time')
        # H stat should be positive and significant
        assert result['statistic'] > 0
        assert result['pvalue'] < 0.1

    def test_not_reject_for_re(self, panel_re_ok):
        """When RE is OK, Hausman should not reject."""
        result = hausman_test(panel_re_ok, y='y', x=['x'],
                              id='id', time='time')
        assert result['pvalue'] > 0.01
        assert result['recommendation'] == 'RE'

    def test_coefficients(self, panel_fe_needed):
        result = hausman_test(panel_fe_needed, y='y', x=['x'],
                              id='id', time='time')
        assert 'beta_fe' in result
        assert 'beta_re' in result
        # Both should estimate positive coefficient
        assert result['beta_fe']['x'] > 0
        assert result['beta_re']['x'] > 0

    def test_interpretation(self, panel_fe_needed):
        result = hausman_test(panel_fe_needed, y='y', x=['x'],
                              id='id', time='time')
        assert 'chi2' in result['interpretation']


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'hausman_test')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
