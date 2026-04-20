"""Tests for broom-style ``tidy()`` and ``glance()`` methods.

Validates that every major result class exposes the unified broom-style
interface:

- ``tidy()`` -> long-format DataFrame with required columns
- ``glance()`` -> 1-row DataFrame with model-level stats

Required columns on ``tidy()``:
    term, estimate, std_error, statistic, p_value, conf_low, conf_high

Required columns on ``glance()``:
    method, nobs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


TIDY_REQUIRED = {'term', 'estimate', 'std_error', 'statistic',
                 'p_value', 'conf_low', 'conf_high'}

GLANCE_REQUIRED = {'method', 'nobs'}


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

@pytest.fixture
def ols_result():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        'x1': rng.normal(size=300),
        'x2': rng.normal(size=300),
    })
    df['y'] = 1.5 + 2 * df['x1'] - 0.5 * df['x2'] + rng.normal(size=300)
    return sp.regress('y ~ x1 + x2', data=df)


class TestOLSTidyGlance:
    def test_tidy_returns_dataframe(self, ols_result):
        t = ols_result.tidy()
        assert isinstance(t, pd.DataFrame)

    def test_tidy_has_required_columns(self, ols_result):
        t = ols_result.tidy()
        missing = TIDY_REQUIRED - set(t.columns)
        assert not missing, f"tidy() missing columns: {missing}"

    def test_tidy_correct_number_of_rows(self, ols_result):
        t = ols_result.tidy()
        assert len(t) == 3  # Intercept, x1, x2

    def test_tidy_estimates_match_params(self, ols_result):
        t = ols_result.tidy().set_index('term')
        for name in ['Intercept', 'x1', 'x2']:
            assert t.loc[name, 'estimate'] == pytest.approx(
                ols_result.params[name]
            )

    def test_tidy_conf_level_affects_interval(self, ols_result):
        t95 = ols_result.tidy(conf_level=0.95).set_index('term')
        t90 = ols_result.tidy(conf_level=0.90).set_index('term')
        # 90% CI must be strictly narrower than 95%
        w95 = (t95['conf_high'] - t95['conf_low'])
        w90 = (t90['conf_high'] - t90['conf_low'])
        assert (w90 < w95).all(), f"90% CI wider than 95%: {w90} vs {w95}"

    def test_glance_returns_one_row(self, ols_result):
        g = ols_result.glance()
        assert isinstance(g, pd.DataFrame)
        assert len(g) == 1

    def test_glance_has_required_columns(self, ols_result):
        g = ols_result.glance()
        missing = GLANCE_REQUIRED - set(g.columns)
        assert not missing, f"glance() missing: {missing}"

    def test_glance_nobs_matches_data(self, ols_result):
        g = ols_result.glance()
        assert g['nobs'].iloc[0] == 300

    def test_glance_has_r_squared(self, ols_result):
        g = ols_result.glance()
        assert 'r_squared' in g.columns
        assert 0 < g['r_squared'].iloc[0] <= 1


# ---------------------------------------------------------------------------
# DID (Callaway-Sant'Anna)
# ---------------------------------------------------------------------------

@pytest.fixture
def cs_result():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(200):
        g = [3, 5, 7, 0][i % 4]
        for t in range(1, 9):
            te = max(0, t - g + 1) * 1.5 if g > 0 else 0
            rows.append({'i': i, 't': t, 'g': g,
                         'y': 0.2 * t + te + rng.normal()})
    df = pd.DataFrame(rows)
    return sp.callaway_santanna(df, y='y', g='g', t='t', i='i',
                                estimator='reg')


class TestCSTidyGlance:
    def test_tidy_main_row_exists(self, cs_result):
        t = cs_result.tidy()
        main = t[t['type'] == 'main']
        assert len(main) == 1
        assert main.iloc[0]['term'] == 'ATT'
        assert main.iloc[0]['estimate'] == pytest.approx(cs_result.estimate)

    def test_tidy_includes_group_time_rows(self, cs_result):
        t = cs_result.tidy()
        gt_rows = t[t['type'] == 'group_time']
        assert len(gt_rows) > 0, "Expected group-time rows"
        # Each group-time term should look like 'att(g=X,t=Y)'
        for term in gt_rows['term']:
            assert term.startswith('att(g=')

    def test_tidy_includes_event_study_rows(self, cs_result):
        t = cs_result.tidy()
        es_rows = t[t['type'] == 'event_study']
        # Event study not always present, but when present:
        if len(es_rows) > 0:
            for term in es_rows['term']:
                assert term.startswith('event_')

    def test_tidy_required_columns(self, cs_result):
        t = cs_result.tidy()
        missing = TIDY_REQUIRED - set(t.columns)
        assert not missing

    def test_glance_has_method_and_estimand(self, cs_result):
        g = cs_result.glance()
        assert 'method' in g.columns
        assert 'estimand' in g.columns
        assert g['estimand'].iloc[0] == 'ATT'

    def test_glance_nobs(self, cs_result):
        g = cs_result.glance()
        assert g['nobs'].iloc[0] > 0


# ---------------------------------------------------------------------------
# RD
# ---------------------------------------------------------------------------

@pytest.fixture
def rd_result():
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(-1, 1, n)
    y = 2 + 3*x + x**2 + 1.0 * (x >= 0).astype(int) + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({'y': y, 'x': x})
    return sp.rdrobust(df, y='y', x='x', c=0.0)


class TestRDTidyGlance:
    def test_tidy_returns_dataframe(self, rd_result):
        t = rd_result.tidy()
        assert isinstance(t, pd.DataFrame)
        assert TIDY_REQUIRED.issubset(set(t.columns))

    def test_tidy_main_row_is_estimand(self, rd_result):
        t = rd_result.tidy()
        main = t[t['type'] == 'main'].iloc[0]
        assert main['estimate'] == pytest.approx(rd_result.estimate)

    def test_glance_has_nobs(self, rd_result):
        g = rd_result.glance()
        assert 'nobs' in g.columns
        assert g['nobs'].iloc[0] > 0


# ---------------------------------------------------------------------------
# Synth
# ---------------------------------------------------------------------------

def test_synth_tidy_glance(synth_factor_model_data):
    """Check that synth results expose tidy/glance interface."""
    r = sp.synth(synth_factor_model_data, outcome='y', unit='unit',
                 time='year',
                 treated_unit=synth_factor_model_data.attrs['treated_unit'],
                 treatment_time=synth_factor_model_data.attrs['treatment_year'],
                 method='classic', placebo=False)
    t = r.tidy()
    g = r.glance()
    assert TIDY_REQUIRED.issubset(set(t.columns))
    assert GLANCE_REQUIRED.issubset(set(g.columns))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_matching_tidy_glance(matching_cia_data):
    r = sp.match(matching_cia_data, y='y', treat='d',
                 covariates=['X1', 'X2', 'X3'], estimand='ATT')
    t = r.tidy()
    g = r.glance()
    assert TIDY_REQUIRED.issubset(set(t.columns))
    assert GLANCE_REQUIRED.issubset(set(g.columns))


# ---------------------------------------------------------------------------
# Shared fixtures from reference_parity
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_factor_model_data():
    rng = np.random.default_rng(1999)
    T, T0, n_controls = 30, 20, 10
    f1 = np.cumsum(rng.normal(scale=0.3, size=T)) + 10
    f2 = np.cumsum(rng.normal(scale=0.3, size=T)) + 5
    treated_loadings = np.array([0.6, 0.4])
    control_loadings = rng.dirichlet([1, 1], size=n_controls)
    control_loadings[0] = np.array([0.8, 0.2])
    control_loadings[1] = np.array([0.4, 0.6])
    rows = []
    for unit in range(n_controls + 1):
        is_t = (unit == n_controls)
        lam = treated_loadings if is_t else control_loadings[unit]
        y_base = lam[0] * f1 + lam[1] * f2
        y = y_base + rng.normal(scale=0.2, size=T)
        if is_t:
            y[T0:] += -5.0
        for t in range(T):
            rows.append({'unit': unit, 'year': 2000 + t, 'y': y[t],
                         'treat': int(is_t and t >= T0),
                         'is_treated_unit': int(is_t)})
    df = pd.DataFrame(rows)
    df.attrs['true_effect'] = -5.0
    df.attrs['treatment_year'] = 2020
    df.attrs['treated_unit'] = n_controls
    return df


@pytest.fixture
def matching_cia_data():
    rng = np.random.default_rng(55)
    n = 1000
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    X3 = rng.binomial(1, 0.4, size=n)
    lin = -0.3 + 0.5 * X1 - 0.3 * X2 + 0.4 * X3
    p = 1 / (1 + np.exp(-lin))
    d = (rng.uniform(0, 1, n) < p).astype(int)
    y0 = 1.0 + 1.5 * X1 - 0.8 * X2 + 0.6 * X3 + rng.normal(scale=0.8, size=n)
    y = y0 + 2.0 * d
    df = pd.DataFrame({'y': y, 'd': d, 'X1': X1, 'X2': X2, 'X3': X3})
    df.attrs['true_effect'] = 2.0
    return df
