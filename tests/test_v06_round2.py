"""
Tests for v0.6 Round 2 modules:
- Interactive FE, Panel Unit Root, Cointegration
- Fractional Response, Beta Regression
- Bivariate Probit, Treatment Effects
- Distributional Treatment Effects
"""

import numpy as np
import pandas as pd
import pytest


class TestInteractiveFE:
    def test_basic(self):
        from statspai.panel.interactive_fe import interactive_fe
        rng = np.random.default_rng(42)
        N, T = 50, 20
        # Generate panel with factor structure
        ids = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        f = rng.normal(0, 1, T)  # common factor
        lam = rng.normal(0, 1, N)  # loadings
        x1 = rng.normal(0, 1, N * T)
        y = 2 * x1 + np.repeat(lam, T) * np.tile(f, N) + rng.normal(0, 0.5, N * T)
        df = pd.DataFrame({'id': ids, 'time': times, 'y': y, 'x1': x1})
        result = interactive_fe(df, y='y', x=['x1'], id='id', time='time', n_factors=1)
        assert result is not None
        # Coefficient on x1 should be close to 2
        assert abs(result.params['x1'] - 2) < 0.5
        s = result.summary()
        assert len(s) > 20


class TestPanelUnitRoot:
    def test_ips(self):
        from statspai.panel.unit_root import panel_unitroot
        rng = np.random.default_rng(42)
        N, T = 20, 50
        ids = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        # Random walk (unit root)
        y = np.zeros(N * T)
        for i in range(N):
            start = i * T
            y[start] = 0
            for t in range(1, T):
                y[start + t] = y[start + t - 1] + rng.normal(0, 1)
        df = pd.DataFrame({'id': ids, 'time': times, 'y': y})
        result = panel_unitroot(df, variable='y', id='id', time='time', test='ips')
        assert result is not None
        s = result.summary()
        assert 'IPS' in s

    def test_fisher(self):
        from statspai.panel.unit_root import panel_unitroot
        rng = np.random.default_rng(42)
        N, T = 20, 50
        ids = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        y = rng.normal(0, 1, N * T)  # stationary
        df = pd.DataFrame({'id': ids, 'time': times, 'y': y})
        result = panel_unitroot(df, variable='y', id='id', time='time', test='fisher')
        assert result is not None
        # Should reject unit root for stationary data
        assert result.p_value < 0.10


class TestCointegration:
    def test_engle_granger(self):
        from statspai.timeseries.cointegration import engle_granger
        rng = np.random.default_rng(42)
        n = 200
        # Cointegrated series: y = 2*x + stationary error
        x = np.cumsum(rng.normal(0, 1, n))
        y = 2 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({'y': y, 'x': x})
        result = engle_granger(df, variables=['y', 'x'])
        assert result is not None
        s = result.summary()
        assert 'Engle-Granger' in s
        # Should detect cointegration
        assert result.rank >= 1

    def test_johansen(self):
        from statspai.timeseries.cointegration import johansen
        rng = np.random.default_rng(42)
        n = 300
        e1 = rng.normal(0, 1, n)
        e2 = rng.normal(0, 1, n)
        x1 = np.cumsum(e1)
        x2 = x1 + rng.normal(0, 0.3, n)  # cointegrated with x1
        x3 = np.cumsum(e2)  # independent random walk
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        result = johansen(df, variables=['x1', 'x2', 'x3'], lags=2)
        assert result is not None
        s = result.summary()
        assert 'Johansen' in s


class TestFractionalResponse:
    def test_fracreg(self):
        from statspai.regression.fracreg import fracreg
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        xb = -0.5 + 0.3 * x1 + 0.2 * x2
        y = 1 / (1 + np.exp(-xb)) + rng.normal(0, 0.1, n)
        y = np.clip(y, 0, 1)
        df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})
        result = fracreg(data=df, y='y', x=['x1', 'x2'])
        assert result is not None
        assert result.params['x1'] > 0

    def test_betareg(self):
        from statspai.regression.fracreg import betareg
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(0, 1, n)
        xb = 0.5 + 0.3 * x1
        mu = 1 / (1 + np.exp(-xb))
        phi = 10
        a = mu * phi
        b = (1 - mu) * phi
        y = rng.beta(a, b)
        df = pd.DataFrame({'y': y, 'x1': x1})
        result = betareg(data=df, y='y', x=['x1'])
        assert result is not None
        s = result.summary()
        assert len(s) > 20


class TestSelectionModels:
    def test_biprobit(self):
        from statspai.regression.selection import biprobit
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        u1 = rng.normal(0, 1, n)
        u2 = 0.5 * u1 + rng.normal(0, np.sqrt(0.75), n)  # correlated
        y1 = (0.5 * x1 + u1 > 0).astype(int)
        y2 = (0.3 * x2 + u2 > 0).astype(int)
        df = pd.DataFrame({'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2})
        result = biprobit(df, y1='y1', y2='y2', x1=['x1'], x2=['x2'])
        assert result is not None
        assert 'rho' in result.model_info

    def test_etregress(self):
        from statspai.regression.selection import etregress
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        x = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        v = 0.5 * u + rng.normal(0, 0.87, n)
        D = (0.5 * z + v > 0).astype(int)
        y = 1 + 0.5 * x + 2 * D + u
        df = pd.DataFrame({'y': y, 'x': x, 'D': D, 'z': z})
        result = etregress(df, y='y', x=['x'], treatment='D', z=['z'])
        assert result is not None
        # Treatment effect should be positive
        assert result.diagnostics['ate'] > 0


class TestDistributionalTE:
    def test_dte(self):
        from statspai.qte.distributional import distributional_te
        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        treatment = rng.binomial(1, 0.5, n)
        y = 2 + 0.5 * x + 1.5 * treatment + rng.normal(0, 1, n)
        df = pd.DataFrame({'y': y, 'treatment': treatment, 'x': x})
        result = distributional_te(df, y='y', treatment='treatment', x=['x'],
                                    n_boot=50, seed=42)
        assert result is not None
        s = result.summary()
        assert 'Distributional' in s or 'DTE' in s or len(s) > 20
