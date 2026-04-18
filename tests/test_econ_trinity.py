"""
Tests for the ``feat/econ-trinity`` P0 trio:

    1. DML-PLIV   (Partially Linear IV, Chernozhukov et al. 2018)
    2. Mixed Logit (Simulated MLE, Train 2009)
    3. IV-QR       (Inverse-QR, Chernozhukov-Hansen 2006/2008)

Each test validates against a controlled DGP — estimates must land within
reasonable bands of the true parameters (looser than vs. reference packages
because these are Monte-Carlo style tests, not bit-matched benchmarks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# =============================================================================
# DML-PLIV
# =============================================================================

class TestDMLPLIV:

    @pytest.fixture(scope='class')
    def dgp_pliv(self):
        rng = np.random.default_rng(42)
        n = 2000
        X = rng.normal(size=(n, 5))
        Z = 0.8 * X[:, 0] + rng.normal(size=n)
        u = rng.normal(size=n)
        D = 0.5 * Z + 0.3 * X[:, 0] ** 2 + 0.4 * u + 0.3 * rng.normal(size=n)
        Y = 1.5 * D + np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 2.0 * u
        df = pd.DataFrame({
            'y': Y, 'd': D, 'z': Z,
            **{f'x{i}': X[:, i] for i in range(5)},
        })
        return df, ['x0', 'x1', 'x2', 'x3', 'x4']

    def test_pliv_recovers_true_theta(self, dgp_pliv):
        df, X_cols = dgp_pliv
        r = sp.dml(df, y='y', treat='d', covariates=X_cols,
                   model='pliv', instrument='z', n_folds=5)
        assert abs(r.estimate - 1.5) < 0.2
        assert r.se > 0 and r.se < 0.2
        assert r.estimand == 'LATE'
        assert r.model_info['dml_model'] == 'PLIV'

    def test_pliv_ci_covers_truth(self, dgp_pliv):
        df, X_cols = dgp_pliv
        r = sp.dml(df, y='y', treat='d', covariates=X_cols,
                   model='pliv', instrument='z', n_folds=5)
        lo, hi = r.ci
        assert lo <= 1.5 <= hi

    def test_pliv_requires_instrument(self, dgp_pliv):
        df, X_cols = dgp_pliv
        with pytest.raises(ValueError, match='pliv'):
            sp.dml(df, y='y', treat='d', covariates=X_cols, model='pliv')

    def test_instrument_rejected_for_plr(self, dgp_pliv):
        df, X_cols = dgp_pliv
        with pytest.raises(ValueError):
            sp.dml(df, y='y', treat='d', covariates=X_cols,
                   model='plr', instrument='z')


# =============================================================================
# Mixed Logit
# =============================================================================

class TestMixedLogit:

    @pytest.fixture(scope='class')
    def panel_choice_data(self):
        rng = np.random.default_rng(0)
        N, T, J = 500, 4, 3
        beta_price = -1.0
        mean_q, sd_q = 1.2, 0.6
        beta_q_ind = rng.normal(mean_q, sd_q, size=N)
        rows = []
        for n in range(N):
            for t in range(T):
                prices = rng.uniform(0.5, 2.0, J)
                quality = rng.uniform(0, 3, J)
                u = (beta_price * prices
                     + beta_q_ind[n] * quality
                     + rng.gumbel(0, 1, J))
                chosen = int(np.argmax(u))
                for j in range(J):
                    rows.append({
                        'pid': n, 'chid': n * T + t, 'alt': j,
                        'price': prices[j], 'quality': quality[j],
                        'y': 1.0 if j == chosen else 0.0,
                    })
        return pd.DataFrame(rows), beta_price, mean_q, sd_q

    def test_mixlogit_recovers_true_params(self, panel_choice_data):
        df, true_p, true_m, true_s = panel_choice_data
        res = sp.mixlogit(
            df, y='y', alt='alt', chid='chid',
            x_fixed=['price'], x_random=['quality'],
            panel_id='pid', n_draws=300, maxiter=80,
        )
        p = res.params
        assert abs(p['price'] - true_p) < 0.25
        assert abs(p['mean_quality'] - true_m) < 0.30
        # sd_quality has a sign ambiguity (identified up to absolute value)
        assert abs(abs(p['sd_quality']) - true_s) < 0.30

    def test_mixlogit_model_metadata(self, panel_choice_data):
        df, *_ = panel_choice_data
        res = sp.mixlogit(
            df, y='y', alt='alt', chid='chid',
            x_random=['price', 'quality'],
            panel_id='pid', n_draws=150, maxiter=50,
        )
        info = res.model_info
        assert info['model_type'] == 'Mixed Logit'
        assert info['n_draws'] == 150
        assert info['converged'] in (True, False)

    def test_mixlogit_requires_random(self, panel_choice_data):
        df, *_ = panel_choice_data
        with pytest.raises(ValueError):
            sp.mixlogit(df, y='y', alt='alt', chid='chid', x_fixed=['price'])


# =============================================================================
# IV-QR
# =============================================================================

class TestIVQR:

    @pytest.fixture(scope='class')
    def dgp_ivqr(self):
        rng = np.random.default_rng(0)
        n = 1200
        Z = rng.normal(size=n)
        X = rng.normal(size=n)
        u = rng.normal(size=n)
        D = 0.8 * Z + 0.3 * X + 0.5 * u + 0.2 * rng.normal(size=n)
        # Pure location model → quantile effects constant at 1.5
        Y = 1.5 * D + 0.5 * X + u
        return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': X})

    def test_ivqr_median(self, dgp_ivqr):
        r = sp.ivqreg(dgp_ivqr, y='y', endog='d',
                      instruments='z', exog=['x'], tau=0.5,
                      n_grid=31, bootstrap=0)
        assert abs(r.params['d'] - 1.5) < 0.15
        assert abs(r.params['x'] - 0.5) < 0.15

    def test_ivqr_multiple_taus(self, dgp_ivqr):
        out = sp.ivqreg(dgp_ivqr, y='y', endog='d',
                        instruments='z', exog=['x'],
                        tau=[0.25, 0.5, 0.75], n_grid=21, bootstrap=0)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 3
        # All three τ should land near 1.5 in a location-shift DGP
        assert (out['d_coef'] - 1.5).abs().max() < 0.30

    def test_ivqr_raises_on_missing_instrument_count(self, dgp_ivqr):
        with pytest.raises(ValueError):
            sp.ivqreg(dgp_ivqr, y='y',
                      endog=['d', 'x'], instruments='z',
                      tau=0.5, bootstrap=0)
