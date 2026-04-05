"""
Tests for v0.6 Round 3 modules:
- Truncated Regression, SUR, 3SLS
- Panel Logit/Probit, Panel FGLS
- Mixed Effects, Stochastic Frontier, General GMM
"""

import numpy as np
import pandas as pd
import pytest


class TestTruncReg:
    def test_left_truncated(self):
        from statspai.regression.truncreg import truncreg
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n)
        y_latent = 2 + 0.5 * x + rng.normal(0, 1, n)
        # Left truncation at 0
        mask = y_latent > 0
        df = pd.DataFrame({'y': y_latent[mask], 'x': x[mask]})
        result = truncreg(data=df, y='y', x=['x'], ll=0)
        assert result is not None
        # Should recover approximately β ≈ 0.5
        assert abs(result.params['x'] - 0.5) < 0.3


class TestSUR:
    def test_sureg(self):
        from statspai.regression.sur import sureg
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n)
        y1 = 1 + 0.5 * x1 + e[:, 0]
        y2 = 2 + 0.3 * x2 + e[:, 1]
        df = pd.DataFrame({'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2})
        result = sureg(
            equations={'eq1': ('y1', ['x1']), 'eq2': ('y2', ['x2'])},
            data=df,
        )
        assert result is not None
        s = result.summary()
        assert 'SUR' in s or 'Seemingly' in s
        # Check equation results exist
        assert 'eq1' in result.equations
        assert 'eq2' in result.equations

    def test_three_sls(self):
        from statspai.regression.sur import three_sls
        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(0, 1, n)
        x = rng.normal(0, 1, n)
        e = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n)
        y1 = 1 + 0.5 * x + 0.3 * z + e[:, 0]
        y2 = 2 + 0.4 * x + e[:, 1]
        df = pd.DataFrame({'y1': y1, 'y2': y2, 'x': x, 'z': z})
        result = three_sls(
            equations={
                'eq1': ('y1', ['x', 'z'], []),
                'eq2': ('y2', ['x'], []),
            },
            data=df,
            instruments=['x', 'z'],
        )
        assert result is not None
        assert result.method == '3SLS'


class TestPanelBinary:
    def test_panel_logit_fe(self):
        from statspai.panel.panel_binary import panel_logit
        rng = np.random.default_rng(42)
        N, T = 100, 10
        ids = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        alpha = np.repeat(rng.normal(0, 1, N), T)
        x = rng.normal(0, 1, N * T)
        latent = alpha + 0.5 * x + rng.logistic(0, 1, N * T)
        y = (latent > 0).astype(int)
        df = pd.DataFrame({'id': ids, 'time': times, 'y': y, 'x': x})
        result = panel_logit(df, y='y', x=['x'], id='id', time='time', method='fe')
        assert result is not None

    def test_panel_logit_re(self):
        from statspai.panel.panel_binary import panel_logit
        rng = np.random.default_rng(42)
        N, T = 50, 8
        ids = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        x = rng.normal(0, 1, N * T)
        latent = 0.5 * x + rng.logistic(0, 1, N * T)
        y = (latent > 0).astype(int)
        df = pd.DataFrame({'id': ids, 'time': times, 'y': y, 'x': x})
        result = panel_logit(df, y='y', x=['x'], id='id', time='time', method='re')
        assert result is not None


class TestPanelFGLS:
    def test_basic(self):
        from statspai.panel.panel_fgls import panel_fgls
        rng = np.random.default_rng(42)
        N, T = 30, 20
        ids = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        x = rng.normal(0, 1, N * T)
        # Heteroskedastic errors
        sigma_i = np.repeat(rng.exponential(1, N), T)
        y = 2 + 0.5 * x + sigma_i * rng.normal(0, 1, N * T)
        df = pd.DataFrame({'id': ids, 'time': times, 'y': y, 'x': x})
        result = panel_fgls(df, y='y', x=['x'], id='id', time='time',
                            panels='heteroskedastic')
        assert result is not None
        assert abs(result.params['x'] - 0.5) < 0.3


class TestMixed:
    def test_random_intercept(self):
        from statspai.multilevel.mixed import mixed
        rng = np.random.default_rng(42)
        n_groups = 30
        n_per = 10
        group = np.repeat(np.arange(n_groups), n_per)
        u = np.repeat(rng.normal(0, 1, n_groups), n_per)  # random intercepts
        x = rng.normal(0, 1, n_groups * n_per)
        y = 2 + 0.5 * x + u + rng.normal(0, 0.5, n_groups * n_per)
        df = pd.DataFrame({'y': y, 'x': x, 'group': group})
        result = mixed(df, y='y', x_fixed=['x'], group='group')
        assert result is not None
        s = result.summary()
        assert 'Mixed' in s or 'mixed' in s or len(s) > 20


class TestFrontier:
    def test_production_frontier(self):
        from statspai.frontier.sfa import frontier
        rng = np.random.default_rng(42)
        n = 300
        log_labor = rng.normal(2, 0.5, n)
        log_capital = rng.normal(3, 0.5, n)
        v = rng.normal(0, 0.3, n)  # noise
        u = np.abs(rng.normal(0, 0.5, n))  # inefficiency
        log_output = 1 + 0.6 * log_labor + 0.4 * log_capital + v - u
        df = pd.DataFrame({
            'log_output': log_output,
            'log_labor': log_labor,
            'log_capital': log_capital,
        })
        result = frontier(df, y='log_output', x=['log_labor', 'log_capital'])
        assert result is not None
        eff = result.efficiency()
        assert len(eff) == n
        assert eff.mean() < 1  # should be less than 1
        assert eff.mean() > 0.3  # should be reasonable
        s = result.summary()
        assert 'Frontier' in s


class TestGeneralGMM:
    def test_iv_gmm(self):
        from statspai.gmm.general_gmm import gmm
        rng = np.random.default_rng(42)
        n = 500
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        x = 0.5 * z1 + 0.3 * z2 + 0.5 * u
        y = 1 + 0.8 * x + u

        df = pd.DataFrame({'y': y, 'x': x, 'z1': z1, 'z2': z2})

        def moment_fn(theta, data):
            y_ = data['y'].values
            x_ = data['x'].values
            z1_ = data['z1'].values
            z2_ = data['z2'].values
            resid = y_ - theta[0] - theta[1] * x_
            return np.column_stack([resid, resid * z1_, resid * z2_])

        result = gmm(moment_fn, theta0=np.zeros(2), data=df,
                     param_names=['_cons', 'x'])
        assert result is not None
        # Should recover β ≈ 0.8
        assert abs(result.params['x'] - 0.8) < 0.5
        # J-test should exist
        assert 'J_stat' in result.diagnostics
