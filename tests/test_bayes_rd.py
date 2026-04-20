"""Tests for ``sp.bayes_rd`` — Bayesian sharp regression discontinuity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianCausalResult, bayes_rd

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sharp_rd_data():
    """n=600, cutoff=0, true LATE = 2.0."""
    rng = np.random.default_rng(201)
    n = 600
    x = rng.uniform(-1, 1, n)
    treated = (x >= 0).astype(int)
    y = (0.5 + 1.0 * x + 2.0 * treated + 0.3 * treated * x
         + rng.normal(0, 0.5, n))
    return pd.DataFrame({'y': y, 'x': x})


@pytest.fixture
def null_rd_data():
    """No discontinuity at cutoff."""
    rng = np.random.default_rng(202)
    n = 600
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': y, 'x': x})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_bayes_rd_returns_result(sharp_rd_data):
    r = bayes_rd(sharp_rd_data, y='y', running='x', cutoff=0.0,
                 draws=300, tune=300, chains=2, progressbar=False)
    assert isinstance(r, BayesianCausalResult)
    assert r.estimand == 'LATE'
    assert r.method.startswith('Bayesian sharp RD')


def test_bayes_rd_top_level_export():
    assert sp.bayes_rd is bayes_rd


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def test_bayes_rd_recovers_sharp_late(sharp_rd_data):
    r = bayes_rd(sharp_rd_data, y='y', running='x', cutoff=0.0,
                 bandwidth=0.5,
                 draws=500, tune=500, chains=2, progressbar=False,
                 random_state=7)
    assert r.hdi_lower < 2.0 < r.hdi_upper, (
        f"True LATE 2.0 not covered by 95% HDI "
        f"[{r.hdi_lower:.3f}, {r.hdi_upper:.3f}]; "
        f"mean {r.posterior_mean:.3f}."
    )


def test_bayes_rd_null_effect_hdi_straddles_zero(null_rd_data):
    r = bayes_rd(null_rd_data, y='y', running='x', cutoff=0.0,
                 bandwidth=0.5,
                 draws=500, tune=500, chains=2, progressbar=False,
                 random_state=8)
    assert r.hdi_lower < 0 < r.hdi_upper, (
        f"Null effect: HDI [{r.hdi_lower:.3f}, {r.hdi_upper:.3f}] "
        "should straddle 0."
    )


# ---------------------------------------------------------------------------
# Bandwidth / polynomial
# ---------------------------------------------------------------------------

def test_bayes_rd_bandwidth_shrinks_local_sample(sharp_rd_data):
    wide = bayes_rd(sharp_rd_data, y='y', running='x', cutoff=0.0,
                    bandwidth=1.0,
                    draws=200, tune=200, chains=2, progressbar=False)
    narrow = bayes_rd(sharp_rd_data, y='y', running='x', cutoff=0.0,
                      bandwidth=0.3,
                      draws=200, tune=200, chains=2, progressbar=False)
    assert narrow.n_obs < wide.n_obs


def test_bayes_rd_poly2_runs(sharp_rd_data):
    r = bayes_rd(sharp_rd_data, y='y', running='x', cutoff=0.0,
                 bandwidth=0.5, poly=2,
                 draws=300, tune=300, chains=2, progressbar=False)
    assert r.model_info['poly'] == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_bayes_rd_bad_bandwidth_raises(sharp_rd_data):
    with pytest.raises(ValueError, match="[Bb]andwidth|observations"):
        bayes_rd(sharp_rd_data, y='y', running='x', cutoff=5.0,
                 bandwidth=0.01,
                 draws=50, tune=50, chains=1, progressbar=False)


def test_bayes_rd_poly_zero_raises(sharp_rd_data):
    with pytest.raises(ValueError, match="poly"):
        bayes_rd(sharp_rd_data, y='y', running='x', cutoff=0.0,
                 poly=0,
                 draws=50, tune=50, chains=1, progressbar=False)
