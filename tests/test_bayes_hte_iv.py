"""Tests for ``sp.bayes_hte_iv`` — heterogeneous-effect Bayesian IV."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianHTEIVResult, bayes_hte_iv

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _hte_iv_dgp(n, tau_intercept, tau_slope, seed):
    """Linear first stage + linear CATE on a single modifier."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    X_mod = rng.normal(size=n)
    U = rng.normal(size=n)
    D = 0.9 * Z + 0.4 * U + rng.normal(size=n)
    tau_i = tau_intercept + tau_slope * X_mod
    Y = tau_i * D + U + rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x_mod': X_mod})


@pytest.fixture
def hte_iv_data():
    return _hte_iv_dgp(800, tau_intercept=1.0, tau_slope=0.6, seed=301)


@pytest.fixture
def homo_iv_data():
    return _hte_iv_dgp(800, tau_intercept=1.0, tau_slope=0.0, seed=302)


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_bayes_hte_iv_returns_extended_result(hte_iv_data):
    r = bayes_hte_iv(hte_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=['x_mod'],
                     draws=300, tune=300, chains=2, progressbar=False)
    assert isinstance(r, BayesianHTEIVResult)
    assert r.estimand == 'LATE (avg)'
    assert 'Bayesian HTE-IV' in r.method
    assert isinstance(r.cate_slopes, pd.DataFrame)
    assert len(r.cate_slopes) == 1
    assert r.cate_slopes.iloc[0]['term'] == 'x_mod'


def test_bayes_hte_iv_top_level_export():
    assert sp.bayes_hte_iv is bayes_hte_iv
    assert sp.BayesianHTEIVResult is BayesianHTEIVResult


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_bayes_hte_iv_average_late_recovered(hte_iv_data):
    r = bayes_hte_iv(hte_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=['x_mod'],
                     draws=500, tune=500, chains=2, progressbar=False,
                     random_state=21)
    # avg LATE at mean(x_mod)=0 is 1.0
    assert r.hdi_lower < 1.0 < r.hdi_upper, (
        f"True avg LATE 1.0 not in 95% HDI "
        f"[{r.hdi_lower:.3f}, {r.hdi_upper:.3f}]"
    )


def test_bayes_hte_iv_slope_recovered_on_heterogeneous_dgp(hte_iv_data):
    r = bayes_hte_iv(hte_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=['x_mod'],
                     draws=500, tune=500, chains=2, progressbar=False,
                     random_state=22)
    row = r.cate_slopes.iloc[0]
    assert row['hdi_low'] < 0.6 < row['hdi_high'], (
        f"True slope 0.6 not in 95% HDI "
        f"[{row['hdi_low']:.3f}, {row['hdi_high']:.3f}]"
    )
    # With clear heterogeneity the slope posterior should be strongly positive.
    assert row['prob_positive'] > 0.9


def test_bayes_hte_iv_null_slope_on_homogeneous_dgp(homo_iv_data):
    r = bayes_hte_iv(homo_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=['x_mod'],
                     draws=500, tune=500, chains=2, progressbar=False,
                     random_state=23)
    row = r.cate_slopes.iloc[0]
    # Null slope should have HDI covering 0, prob_positive near 0.5
    assert row['hdi_low'] < 0 < row['hdi_high'], (
        f"Null slope HDI [{row['hdi_low']:.3f}, {row['hdi_high']:.3f}] "
        "should straddle 0"
    )
    assert 0.1 < row['prob_positive'] < 0.9


# ---------------------------------------------------------------------------
# predict_cate helper
# ---------------------------------------------------------------------------


def test_bayes_hte_iv_predict_cate_returns_summary(hte_iv_data):
    r = bayes_hte_iv(hte_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=['x_mod'],
                     draws=400, tune=400, chains=2, progressbar=False,
                     random_state=24)
    p = r.predict_cate({'x_mod': 1.0})
    for key in ('mean', 'median', 'sd', 'hdi_low', 'hdi_high', 'prob_positive'):
        assert key in p
    # True CATE at x_mod=1.0 is 1.0 + 0.6*1.0 = 1.6
    assert abs(p['mean'] - 1.6) < 0.3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_bayes_hte_iv_missing_modifier_raises(hte_iv_data):
    with pytest.raises(ValueError, match='not found'):
        bayes_hte_iv(hte_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=['nope'],
                     draws=50, tune=50, chains=1, progressbar=False)


def test_bayes_hte_iv_empty_modifiers_raises(hte_iv_data):
    with pytest.raises(ValueError, match='effect_modifiers'):
        bayes_hte_iv(hte_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=[],
                     draws=50, tune=50, chains=1, progressbar=False)


def test_bayes_hte_iv_modifier_covariate_overlap_raises(hte_iv_data):
    """A column in both effect_modifiers and covariates would enter
    the design matrix twice asymmetrically (via M in the structural
    equation, via X in the first stage) — the user should pick one
    role explicitly."""
    with pytest.raises(ValueError, match='overlap'):
        bayes_hte_iv(hte_iv_data, y='y', treat='d', instrument='z',
                     effect_modifiers=['x_mod'], covariates=['x_mod'],
                     draws=50, tune=50, chains=1, progressbar=False)


# ---------------------------------------------------------------------------
# Multi-modifier smoke
# ---------------------------------------------------------------------------


def test_bayes_hte_iv_multiple_modifiers():
    rng = np.random.default_rng(30)
    n = 600
    Z = rng.normal(size=n)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    U = rng.normal(size=n)
    D = 0.8 * Z + 0.3 * U + rng.normal(size=n)
    tau_i = 1.0 + 0.5 * X1 + 0.2 * X2
    Y = tau_i * D + U + rng.normal(size=n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x1': X1, 'x2': X2})
    r = bayes_hte_iv(df, y='y', treat='d', instrument='z',
                     effect_modifiers=['x1', 'x2'],
                     draws=300, tune=300, chains=2, progressbar=False)
    assert len(r.cate_slopes) == 2
    assert set(r.cate_slopes['term']) == {'x1', 'x2'}
