"""
Tests for Arellano-Bond / Blundell-Bond dynamic panel GMM.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.gmm import xtabond
from statspai.core.results import CausalResult


@pytest.fixture
def dynamic_panel():
    """Simulated dynamic panel: Y_it = 0.5*Y_{i,t-1} + 1.0*X_it + α_i + ε_it.

    50 units × 10 periods. True ρ = 0.5, true β_x = 1.0.
    """
    rng = np.random.default_rng(42)
    N, T = 50, 10
    rho_true = 0.5
    beta_x_true = 1.0
    alpha = rng.normal(0, 1, N)  # unit FE

    rows = []
    for i in range(N):
        y_prev = rng.normal(0, 1)
        for t in range(T):
            x = rng.normal(0, 1)
            eps = rng.normal(0, 0.5)
            y = rho_true * y_prev + beta_x_true * x + alpha[i] + eps
            rows.append({'id': i, 'time': t, 'y': y, 'x': x})
            y_prev = y

    return pd.DataFrame(rows)


class TestArellanoBond:
    def test_basic_run(self, dynamic_panel):
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        assert isinstance(result, CausalResult)
        assert 'Arellano-Bond' in result.method

    def test_rho_positive(self, dynamic_panel):
        """AR coefficient should be positive (true ρ = 0.5)."""
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        assert result.estimate > 0

    def test_rho_magnitude(self, dynamic_panel):
        """ρ̂ should be in reasonable range of 0.5."""
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        assert abs(result.estimate - 0.5) < 0.3

    def test_x_coefficient(self, dynamic_panel):
        """β_x should be positive (true = 1.0)."""
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        x_coef = result.detail[result.detail['variable'] == 'x']['coefficient'].values[0]
        assert x_coef > 0

    def test_ar2_not_reject(self, dynamic_panel):
        """AR(2) test should not reject (DGP has AR(1) only)."""
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        assert result.model_info['ar2_p'] > 0.01

    def test_hansen_test(self, dynamic_panel):
        """Hansen test should be in model_info."""
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        assert 'hansen_p' in result.model_info

    def test_n_instruments(self, dynamic_panel):
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        assert result.model_info['n_instruments'] > 0

    def test_twostep(self, dynamic_panel):
        """Two-step GMM should work."""
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time', twostep=True)
        assert isinstance(result, CausalResult)
        assert result.model_info['twostep'] is True

    def test_system_gmm_not_implemented(self, dynamic_panel):
        """System (Blundell-Bond) GMM is gated until it has a parity ref.

        Proper system GMM needs a stacked level equation; rather than
        return an unvalidated (and previously distorted) estimate, the
        method raises loudly and points at the difference estimator.
        """
        with pytest.raises(NotImplementedError, match="system GMM"):
            xtabond(dynamic_panel, y='y', x=['x'],
                    id='id', time='time', method='system')

    def test_parity_matches_stata_xtabond(self):
        """Difference GMM (one-step robust) must match Stata's xtabond to
        machine precision on the parity DGP (tests/r_parity/50_xtabond.py).

        Stata `xtabond y x, lags(1) vce(robust)`:
            beta_y_lag = 0.39117889 (se 0.04632272)
            beta_x     = 0.21695482 (se 0.04361645)
        """
        rng = np.random.default_rng(42)
        N, T = 100, 8
        rows, y_prev = [], np.zeros(N)
        for t in range(T):
            xx = rng.normal(0, 1, N)
            yy = 0.5 * y_prev + 0.3 * xx + rng.normal(0, 1, N)
            for i in range(N):
                rows.append({'id': i, 'time': t,
                             'y': float(yy[i]), 'x': float(xx[i])})
            y_prev = yy
        df = pd.DataFrame(rows)
        res = xtabond(df, y='y', x=['x'], id='id', time='time', lags=1,
                      gmm_lags=(2, None), method='difference',
                      twostep=False, robust=True)
        d = {r['variable']: (r['coefficient'], r['se'])
             for _, r in res.detail.iterrows()}
        assert abs(d['_y_lag1'][0] - 0.39117889) < 1e-5
        assert abs(d['_y_lag1'][1] - 0.04632272) < 1e-5
        assert abs(d['x'][0] - 0.21695482) < 1e-5
        assert abs(d['x'][1] - 0.04361645) < 1e-5

    def test_no_exogenous(self, dynamic_panel):
        """Should work with only lagged Y (no X)."""
        result = xtabond(dynamic_panel, y='y', id='id', time='time')
        assert isinstance(result, CausalResult)

    def test_detail_table(self, dynamic_panel):
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        assert 'variable' in result.detail.columns
        assert 'coefficient' in result.detail.columns
        assert 'se' in result.detail.columns

    def test_summary(self, dynamic_panel):
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        s = result.summary()
        assert 'Arellano-Bond' in s or 'GMM' in s

    def test_cite(self, dynamic_panel):
        result = xtabond(dynamic_panel, y='y', x=['x'],
                         id='id', time='time')
        bib = result.cite()
        assert 'arellano' in bib.lower()


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'xtabond')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
