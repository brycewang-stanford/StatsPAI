"""
Tests for Heckman (1979) selection model.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.regression.heckman import heckman
from statspai.core.results import CausalResult


@pytest.fixture
def heckman_data():
    """Simulated data with sample selection.

    DGP:
    - Selection: D* = 0.5*Z + u, D = 1(D* > 0)
    - Outcome: Y = 1 + 2*X + ε, observed only if D=1
    - u and ε are correlated (ρ = 0.6) → selection bias
    """
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.normal(0, 1, n)
    Z_excl = rng.normal(0, 1, n)  # exclusion restriction

    # Correlated errors
    rho = 0.6
    u = rng.normal(0, 1, n)
    eps = rho * u + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)

    # Selection
    D_star = 0.3 + 0.5 * X + 0.8 * Z_excl + u
    D = (D_star > 0).astype(int)

    # Outcome (observed only when D=1)
    Y = 1 + 2 * X + eps
    Y_obs = np.where(D == 1, Y, np.nan)

    return pd.DataFrame({
        'y': Y_obs, 'x': X, 'z_excl': Z_excl,
        'selected': D,
    })


class TestHeckman:
    def test_basic_run(self, heckman_data):
        result = heckman(heckman_data, y='y', x=['x'],
                         select='selected', z=['x', 'z_excl'])
        assert isinstance(result, CausalResult)
        assert 'Heckman' in result.method

    def test_x_coefficient(self, heckman_data):
        """β_x should be close to 2.0."""
        result = heckman(heckman_data, y='y', x=['x'],
                         select='selected', z=['x', 'z_excl'])
        assert abs(result.estimate - 2.0) < 0.5

    def test_lambda_significant(self, heckman_data):
        """Lambda (IMR) should be significant (DGP has selection)."""
        result = heckman(heckman_data, y='y', x=['x'],
                         select='selected', z=['x', 'z_excl'])
        assert result.model_info['lambda_pvalue'] < 0.1

    def test_n_selected(self, heckman_data):
        result = heckman(heckman_data, y='y', x=['x'],
                         select='selected', z=['x', 'z_excl'])
        assert result.model_info['n_selected'] > 0
        assert result.model_info['n_censored'] > 0

    def test_detail_table(self, heckman_data):
        result = heckman(heckman_data, y='y', x=['x'],
                         select='selected', z=['x', 'z_excl'])
        assert 'lambda (IMR)' in result.detail['variable'].values

    def test_no_selection_bias(self):
        """When errors are uncorrelated, lambda should be insignificant."""
        rng = np.random.default_rng(42)
        n = 2000
        X = rng.normal(0, 1, n)
        Z_excl = rng.normal(0, 1, n)
        D = (0.5 * Z_excl + rng.normal(0, 1, n) > 0).astype(int)
        Y = 1 + 2 * X + rng.normal(0, 1, n)  # independent errors
        Y_obs = np.where(D == 1, Y, np.nan)
        df = pd.DataFrame({'y': Y_obs, 'x': X, 'z': Z_excl, 'sel': D})

        result = heckman(df, y='y', x=['x'], select='sel', z=['x', 'z'])
        # Lambda should be non-significant
        assert result.model_info['lambda_pvalue'] > 0.01

    def test_cite(self, heckman_data):
        result = heckman(heckman_data, y='y', x=['x'],
                         select='selected', z=['x', 'z_excl'])
        assert 'heckman1979' in result.cite()

    def test_summary(self, heckman_data):
        result = heckman(heckman_data, y='y', x=['x'],
                         select='selected', z=['x', 'z_excl'])
        s = result.summary()
        assert 'Heckman' in s


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'heckman')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
