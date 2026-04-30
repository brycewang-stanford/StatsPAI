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


# =============================================================================
# Adversarial / regression tests (bugs caught in code review)
# =============================================================================

class TestAdversarial:
    """Tests that target specific bugs discovered during the first review."""

    def test_mixlogit_string_chid_matches_numeric(self):
        """
        B1/B2 regression: Mixed Logit must produce identical results whether
        chid values sort lexicographically or numerically. Prior to the fix,
        using string chids like 'c0','c1','c10','c100',... would silently
        misalign situation blocks and biased all parameters.
        """
        import numpy as np
        rng = np.random.default_rng(0)
        N, T, J = 150, 3, 3
        beta_price, mean_q, sd_q = -1.0, 1.5, 0.6
        beta_q_ind = rng.normal(mean_q, sd_q, size=N)
        rows_num, rows_str = [], []
        for n in range(N):
            for t in range(T):
                prices = rng.uniform(0.5, 2.0, J)
                quality = rng.uniform(0, 3, J)
                u = (beta_price * prices + beta_q_ind[n] * quality
                     + rng.gumbel(0, 1, J))
                chosen = int(np.argmax(u))
                for j in range(J):
                    base = {
                        'price': prices[j], 'quality': quality[j],
                        'y': 1.0 if j == chosen else 0.0,
                    }
                    rows_num.append({'pid': n, 'chid': n * T + t, **base})
                    rows_str.append({'pid': f'p{n:03d}',
                                     'chid': f'c{n * T + t}', **base})
        df_num = pd.DataFrame(rows_num)
        df_str = pd.DataFrame(rows_str)

        r_num = sp.mixlogit(df_num, y='y', chid='chid',
                            x_fixed=['price'], x_random=['quality'],
                            panel_id='pid', n_draws=100, maxiter=30)
        r_str = sp.mixlogit(df_str, y='y', chid='chid',
                            x_fixed=['price'], x_random=['quality'],
                            panel_id='pid', n_draws=100, maxiter=30)
        # Bit-identical parameters (tolerance for float noise)
        for name in ['price', 'mean_quality', 'sd_quality']:
            assert abs(r_num.params[name] - r_str.params[name]) < 1e-8, (
                f"{name}: numeric vs string chid differ by "
                f"{abs(r_num.params[name] - r_str.params[name]):.2e}"
            )

    def test_mixlogit_missing_chosen_alternative_raises(self):
        """B-extra: situations without exactly one y==1 must raise."""
        df = pd.DataFrame({
            'chid': [1, 1, 1, 2, 2, 2],
            'y':    [0, 0, 0, 1, 0, 0],  # chid=1 has no chosen
            'x':    [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        })
        with pytest.raises(ValueError, match='chosen alternative'):
            sp.mixlogit(df, y='y', chid='chid', x_random=['x'],
                        n_draws=50, maxiter=10)

    def test_mixlogit_correlated_rejects_non_normal(self):
        """B3-related: correlated=True must reject lognormal/triangular."""
        df = pd.DataFrame({
            'chid': [0, 0, 1, 1],
            'y': [1, 0, 0, 1],
            'x1': [1.0, 2.0, 1.5, 2.5],
            'x2': [0.5, 1.5, 0.8, 1.8],
        })
        with pytest.raises(ValueError, match='correlated=True'):
            sp.mixlogit(df, y='y', chid='chid',
                        x_random=['x1', 'x2'],
                        random_dist={'x1': 'lognormal'},
                        correlated=True,
                        n_draws=20, maxiter=5)

    def test_ivqreg_multidim_warns_when_bootstrap_zero(self):
        """B5: multi-dim endog without bootstrap must warn about NaN SEs."""
        rng = np.random.default_rng(0)
        n = 300
        Z1 = rng.normal(size=n); Z2 = rng.normal(size=n)
        X = rng.normal(size=n);  u = rng.normal(size=n)
        D1 = 0.6 * Z1 + 0.2 * X + 0.3 * u + 0.3 * rng.normal(size=n)
        D2 = 0.5 * Z2 + 0.2 * X + 0.3 * u + 0.3 * rng.normal(size=n)
        Y = 1.0 * D1 + 0.5 * D2 + 0.3 * X + u
        df = pd.DataFrame({'y': Y, 'd1': D1, 'd2': D2,
                           'z1': Z1, 'z2': Z2, 'x': X})
        with pytest.warns(UserWarning, match='bootstrap=0'):
            sp.ivqreg(df, y='y', endog=['d1', 'd2'],
                      instruments=['z1', 'z2'], exog=['x'],
                      tau=0.5, bootstrap=0)

    def test_dml_pliv_rejects_multi_instrument_list(self):
        """H7: passing 2+ instruments must raise (no silent truncation)."""
        rng = np.random.default_rng(0)
        n = 400
        X = rng.normal(size=(n, 3))
        Z1 = 0.5 * X[:, 0] + rng.normal(size=n)
        Z2 = 0.5 * X[:, 1] + rng.normal(size=n)
        u = rng.normal(size=n)
        D = 0.4 * Z1 + 0.3 * Z2 + 0.5 * u + rng.normal(size=n) * 0.3
        Y = 1.5 * D + X[:, 0] + 0.5 * u
        df = pd.DataFrame({
            'y': Y, 'd': D, 'z1': Z1, 'z2': Z2,
            **{f'x{i}': X[:, i] for i in range(3)},
        })
        with pytest.raises(ValueError, match='single scalar instrument'):
            sp.dml(df, y='y', treat='d',
                   covariates=['x0', 'x1', 'x2'],
                   model='pliv', instrument=['z1', 'z2'])

    def test_dml_pliv_degenerate_instrument_raises(self):
        """
        M6: the weak-instrument guard is scale-aware and catches
        *degenerate* instruments (near-zero covariance after residualizing
        on X). This test uses Z = X[:,0] exactly — a perfectly collinear
        (irrelevant) instrument whose residualization on X leaves pure
        ML-fit noise.
        """
        rng = np.random.default_rng(0)
        n = 800
        X = rng.normal(size=(n, 3))
        # Z is literally a linear function of X → residualization kills it
        # (up to ML approximation error; the scale-aware guard triggers
        # when the ratio is below 1e-6, which requires the ML predictor
        # to be near-exact on X → deliberately use a linear learner that
        # can recover X[:,0] exactly).
        Z = X[:, 0].copy()
        D = 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.5 * rng.normal(size=n)
        Y = 1.0 * D + X[:, 0] + rng.normal(size=n)
        df = pd.DataFrame({
            'y': Y, 'd': D, 'z': Z,
            **{f'x{i}': X[:, i] for i in range(3)},
        })
        from sklearn.linear_model import LinearRegression
        # A near-perfect linear fit of Z on X makes z_resid ≈ 0 →
        # |E[z̃·d̃]| / scale ≈ 0 → guard trips.
        # In v1.12 the PLIV first-stage guard message changed to
        # "Weak / degenerate PLIV first stage" and the threshold tightened
        # from 1e-6 to 1e-3 on the partial correlation. The match here is
        # case-insensitive and tolerates either the old or new wording.
        with pytest.raises(RuntimeError, match=r'(?i)(weak|degenerate).*PLIV'):
            sp.dml(df, y='y', treat='d',
                   covariates=['x0', 'x1', 'x2'],
                   model='pliv', instrument='z',
                   ml_r=LinearRegression(),
                   n_folds=5)
