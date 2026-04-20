"""Tests for ``sp.bayes_iv`` — Bayesian linear IV via control function."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianCausalResult, bayes_iv

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixtures with known LATE
# ---------------------------------------------------------------------------


def _iv_dgp(n, strength, rho=0.5, seed=0, true_late=1.5):
    """Linear IV DGP with tunable first-stage strength ``pi_Z`` and
    confounding correlation ``rho`` between first-stage and structural
    residuals."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U = rng.normal(size=n)
    v = rng.normal(size=n)
    # Endogeneity: D absorbs fraction `rho` of U
    D = strength * Z + rho * U + v
    Y = true_late * D + U + rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z})


@pytest.fixture
def strong_iv_data():
    return _iv_dgp(600, strength=0.9, rho=0.5, seed=101, true_late=1.5)


@pytest.fixture
def weak_iv_data():
    return _iv_dgp(600, strength=0.05, rho=0.5, seed=102, true_late=1.5)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_bayes_iv_returns_result(strong_iv_data):
    r = bayes_iv(strong_iv_data, y='y', treat='d', instrument='z',
                 draws=300, tune=300, chains=2, progressbar=False)
    assert isinstance(r, BayesianCausalResult)
    assert r.estimand == 'LATE'
    assert r.method.startswith('Bayesian IV')


def test_bayes_iv_top_level_export():
    assert sp.bayes_iv is bayes_iv


# ---------------------------------------------------------------------------
# Recovery / identification
# ---------------------------------------------------------------------------


def test_bayes_iv_strong_instrument_recovers_late(strong_iv_data):
    r = bayes_iv(strong_iv_data, y='y', treat='d', instrument='z',
                 draws=500, tune=500, chains=2, progressbar=False,
                 random_state=7)
    assert r.hdi_lower < 1.5 < r.hdi_upper, (
        f"True LATE=1.5 not covered by 95% HDI "
        f"[{r.hdi_lower:.3f}, {r.hdi_upper:.3f}] "
        f"(mean {r.posterior_mean:.3f})"
    )


def test_bayes_iv_weak_instrument_widens_hdi(strong_iv_data, weak_iv_data):
    """Weak Z should produce a substantially wider posterior than strong Z."""
    strong = bayes_iv(strong_iv_data, y='y', treat='d', instrument='z',
                      draws=400, tune=400, chains=2, progressbar=False,
                      random_state=11)
    weak = bayes_iv(weak_iv_data, y='y', treat='d', instrument='z',
                    draws=400, tune=400, chains=2, progressbar=False,
                    random_state=11)
    strong_width = strong.hdi_upper - strong.hdi_lower
    weak_width = weak.hdi_upper - weak.hdi_lower
    assert weak_width > 1.5 * strong_width, (
        f"Weak-IV HDI width {weak_width:.3f} should be >> strong "
        f"{strong_width:.3f}"
    )


# ---------------------------------------------------------------------------
# Multiple instruments / covariates
# ---------------------------------------------------------------------------


def test_bayes_iv_multiple_instruments():
    rng = np.random.default_rng(13)
    n = 600
    Z1 = rng.normal(size=n)
    Z2 = rng.normal(size=n)
    U = rng.normal(size=n)
    D = 0.5 * Z1 + 0.4 * Z2 + 0.3 * U + rng.normal(size=n)
    Y = 1.5 * D + U + rng.normal(size=n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z1': Z1, 'z2': Z2})
    r = bayes_iv(df, y='y', treat='d', instrument=['z1', 'z2'],
                 draws=300, tune=300, chains=2, progressbar=False)
    assert r.model_info['n_instruments'] == 2
    assert np.isfinite(r.posterior_mean)


def test_bayes_iv_covariates_run():
    rng = np.random.default_rng(14)
    n = 500
    Z = rng.normal(size=n)
    X = rng.normal(size=n)
    U = rng.normal(size=n)
    D = 0.7 * Z + 0.4 * X + 0.3 * U + rng.normal(size=n)
    Y = 1.5 * D + 0.5 * X + U + rng.normal(size=n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': X})
    r = bayes_iv(df, y='y', treat='d', instrument='z', covariates=['x'],
                 draws=300, tune=300, chains=2, progressbar=False)
    assert r.model_info['covariates'] == ['x']


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_bayes_iv_missing_column_raises(strong_iv_data):
    with pytest.raises(ValueError, match='not found'):
        bayes_iv(strong_iv_data, y='nope', treat='d', instrument='z',
                 draws=50, tune=50, chains=1, progressbar=False)


def test_bayes_iv_tidy_and_glance(strong_iv_data):
    r = bayes_iv(strong_iv_data, y='y', treat='d', instrument='z',
                 draws=300, tune=300, chains=2, progressbar=False)
    assert len(r.tidy()) == 1
    assert 'late' in r.tidy()['term'].iloc[0]
    assert len(r.glance()) == 1
