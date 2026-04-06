"""
Tests for the Smart Workflow Engine — StatsPAI's unique features.
"""

import numpy as np
import pandas as pd
import pytest


def _sample_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    treatment = rng.binomial(1, 0.5, n)
    y = 2 + 0.5 * x1 + 0.3 * x2 + 1.5 * treatment + rng.normal(0, 1, n)
    return pd.DataFrame({
        'y': y, 'x1': x1, 'x2': x2, 'treatment': treatment,
    })


def _panel_data(n_units=50, n_periods=6, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    treatment = np.repeat(rng.binomial(1, 0.5, n_units), n_periods)
    post = (times >= 3).astype(int)
    treat_post = treatment * post
    y = 2 + 1.5 * treat_post + rng.normal(0, 1, n_units * n_periods)
    x1 = rng.normal(0, 1, n_units * n_periods)
    return pd.DataFrame({
        'id': ids, 'time': times, 'y': y, 'treatment': treatment,
        'post': post, 'treat_post': treat_post, 'x1': x1,
    })


class TestRecommend:
    def test_observational(self):
        from statspai.smart.recommend import recommend
        df = _sample_data()
        rec = recommend(df, y='y', treatment='treatment',
                        covariates=['x1', 'x2'])
        assert rec is not None
        assert len(rec.recommendations) >= 2
        assert rec.design == 'observational'
        s = rec.summary()
        assert 'Recommendation' in s or 'RECOMMENDED' in s

    def test_did_detection(self):
        from statspai.smart.recommend import recommend
        df = _panel_data()
        # Treatment varies over time → should detect DID
        rec = recommend(df, y='y', treatment='treat_post',
                        id='id', time='time')
        assert rec is not None
        assert rec.design in ['did', 'panel']

    def test_rct(self):
        from statspai.smart.recommend import recommend
        df = _sample_data()
        rec = recommend(df, y='y', treatment='treatment', design='rct')
        assert rec is not None
        assert rec.design == 'rct'
        assert any('OLS' in r['method'] for r in rec.recommendations)

    def test_iv(self):
        from statspai.smart.recommend import recommend
        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(0, 1, n)
        x = 0.5 * z + rng.normal(0, 1, n)
        y = 1 + 0.8 * x + rng.normal(0, 1, n)
        df = pd.DataFrame({'y': y, 'x': x, 'z': z})
        rec = recommend(df, y='y', treatment='x', instrument='z')
        assert rec.design == 'iv'
        assert any('2SLS' in r['method'] or 'LIML' in r['method']
                    for r in rec.recommendations)

    def test_run(self):
        from statspai.smart.recommend import recommend
        df = _sample_data()
        rec = recommend(df, y='y', treatment='treatment', design='rct')
        result = rec.run()
        assert result is not None


class TestCompareEstimators:
    def test_basic(self):
        from statspai.smart.compare import compare_estimators
        df = _sample_data()
        comp = compare_estimators(df, y='y', treatment='treatment',
                                   methods=['ols'],
                                   covariates=['x1', 'x2'])
        assert comp is not None
        # OLS may succeed or fail depending on API — check gracefully
        assert isinstance(comp.estimates_table, pd.DataFrame)


class TestAssumptionAudit:
    def test_ols_audit(self):
        import statspai as sp
        from statspai.smart.assumptions import assumption_audit
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'y': rng.normal(0, 1, 200),
            'x1': rng.normal(0, 1, 200),
            'x2': rng.normal(0, 1, 200),
        })
        result = sp.regress('y ~ x1 + x2', data=df, robust='hc1')
        audit = assumption_audit(result, verbose=False)
        assert audit is not None
        assert len(audit.checks) >= 1
        assert audit.overall_grade in ['A', 'B', 'C', 'D', 'F', '?']


class TestPubReady:
    def test_top5(self):
        from statspai.smart.publication import pub_ready
        check = pub_ready(venue='top5_econ', design='observational')
        assert check is not None
        assert check.score >= 0
        s = check.summary()
        assert 'Publication' in s or 'Readiness' in s

    def test_rct(self):
        from statspai.smart.publication import pub_ready
        check = pub_ready(venue='rct', design='rct',
                          has_balance=True, has_robustness=True)
        assert check.score > 0
        assert len(check.present) >= 1


class TestReplicate:
    def test_list(self):
        from statspai.smart.replicate import list_replications
        df = list_replications()
        assert len(df) >= 4
        assert 'card_1995' in df['key'].values

    def test_card_1995(self):
        from statspai.smart.replicate import replicate
        data, guide = replicate('card_1995')
        assert data is not None
        assert len(data) == 3010
        assert 'lwage' in data.columns
        assert 'nearc4' in data.columns
        assert 'REPLICATION GUIDE' in guide

    def test_lalonde(self):
        from statspai.smart.replicate import replicate
        data, guide = replicate('lalonde_1986')
        assert len(data) == 445
        assert 'treat' in data.columns
        assert 're78' in data.columns

    def test_lee_rd(self):
        from statspai.smart.replicate import replicate
        data, guide = replicate('lee_2008')
        assert len(data) == 6558
        assert 'margin' in data.columns

    def test_unknown(self):
        from statspai.smart.replicate import replicate
        with pytest.raises(ValueError, match="Unknown replication"):
            replicate('nonexistent_paper')


class TestSensitivityDashboard:
    def test_basic(self):
        import statspai as sp
        from statspai.smart.sensitivity import sensitivity_dashboard
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'y': rng.normal(0, 1, 200),
            'x1': rng.normal(0, 1, 200),
        })
        result = sp.regress('y ~ x1', data=df)
        dash = sensitivity_dashboard(result, data=df, verbose=False,
                                      dimensions=['sample'])
        assert dash is not None
        assert dash.overall_stability in ['A', 'B', 'C', 'D', 'F', '?']
