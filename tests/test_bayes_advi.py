"""Tests for the ``inference='advi'`` option on all Bayesian estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.bayes import (
    bayes_did, bayes_rd, bayes_iv, bayes_fuzzy_rd, bayes_hte_iv,
)

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# Small ADVI budget — just smoke + sanity.
_ADVI_ITS = 5000
_DRAWS = 400


def _did_data():
    rng = np.random.default_rng(401)
    rows = []
    for u in range(40):
        treated = u < 20
        for t in range(2):
            post = (t == 1)
            y = (1.0 + 0.5 * treated + 0.3 * post
                 + 1.5 * (treated and post)
                 + rng.normal(0, 0.5))
            rows.append({'y': y, 'treat': int(treated), 'post': int(post)})
    return pd.DataFrame(rows)


def _rd_data():
    rng = np.random.default_rng(402)
    n = 500
    x = rng.uniform(-1, 1, n)
    t = (x >= 0).astype(int)
    y = 0.5 + 1.0 * x + 2.0 * t + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': y, 'x': x})


def _iv_data():
    rng = np.random.default_rng(403)
    n = 500
    Z = rng.normal(size=n)
    U = rng.normal(size=n)
    D = 0.9 * Z + 0.3 * U + rng.normal(size=n)
    Y = 1.5 * D + U + rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z})


def _fuzzy_data():
    rng = np.random.default_rng(404)
    n = 600
    x = rng.uniform(-1, 1, n)
    side = (x >= 0).astype(int)
    p = np.where(side == 1, 0.8, 0.1)
    D = rng.binomial(1, p).astype(float)
    Y = 0.5 + 1.0 * x + 2.0 * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x': x})


def _hte_iv_data():
    rng = np.random.default_rng(405)
    n = 500
    Z = rng.normal(size=n)
    X = rng.normal(size=n)
    U = rng.normal(size=n)
    D = 0.9 * Z + 0.3 * U + rng.normal(size=n)
    tau_i = 1.0 + 0.5 * X
    Y = tau_i * D + U + rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x_mod': X})


# ---------------------------------------------------------------------------
# Each estimator with ADVI must return a usable result
# ---------------------------------------------------------------------------


def test_bayes_did_advi_runs():
    df = _did_data()
    r = bayes_did(df, y='y', treat='treat', post='post',
                  inference='advi', advi_iterations=_ADVI_ITS,
                  draws=_DRAWS, progressbar=False)
    assert r.model_info['inference'] == 'advi'
    assert np.isfinite(r.posterior_mean)


def test_bayes_rd_advi_runs():
    df = _rd_data()
    r = bayes_rd(df, y='y', running='x', cutoff=0.0, bandwidth=0.5,
                 inference='advi', advi_iterations=_ADVI_ITS,
                 draws=_DRAWS, progressbar=False)
    assert r.model_info['inference'] == 'advi'
    assert np.isfinite(r.posterior_mean)


def test_bayes_iv_advi_runs():
    df = _iv_data()
    r = bayes_iv(df, y='y', treat='d', instrument='z',
                 inference='advi', advi_iterations=_ADVI_ITS,
                 draws=_DRAWS, progressbar=False)
    assert r.model_info['inference'] == 'advi'
    assert np.isfinite(r.posterior_mean)


def test_bayes_fuzzy_rd_advi_runs():
    df = _fuzzy_data()
    r = bayes_fuzzy_rd(df, y='y', treat='d', running='x', cutoff=0.0,
                       bandwidth=0.5,
                       inference='advi', advi_iterations=_ADVI_ITS,
                       draws=_DRAWS, progressbar=False)
    assert r.model_info['inference'] == 'advi'
    assert np.isfinite(r.posterior_mean)


def test_bayes_hte_iv_advi_runs():
    df = _hte_iv_data()
    r = bayes_hte_iv(df, y='y', treat='d', instrument='z',
                     effect_modifiers=['x_mod'],
                     inference='advi', advi_iterations=_ADVI_ITS,
                     draws=_DRAWS, progressbar=False)
    assert r.model_info['inference'] == 'advi'
    assert np.isfinite(r.posterior_mean)
    assert len(r.cate_slopes) == 1


# ---------------------------------------------------------------------------
# Invalid inference mode
# ---------------------------------------------------------------------------


def test_bayes_did_advi_summary_flags_rhat():
    """ADVI traces have 1 chain so R-hat is NaN. summary() must
    communicate that convergence is undiagnosable rather than
    printing a bare 'nan' that can be mistaken for a model failure."""
    df = _did_data()
    r = bayes_did(df, y='y', treat='treat', post='post',
                  inference='advi', advi_iterations=_ADVI_ITS,
                  draws=_DRAWS, progressbar=False)
    s = r.summary()
    assert 'ADVI' in s
    assert 'mean-field' in s or 'variational' in s
    assert 'Inference:   advi' in s


@pytest.mark.parametrize("fn,kwargs", [
    (bayes_did, {'y': 'y', 'treat': 'treat', 'post': 'post'}),
    (bayes_rd, {'y': 'y', 'running': 'x', 'cutoff': 0.0, 'bandwidth': 0.5}),
    (bayes_iv, {'y': 'y', 'treat': 'd', 'instrument': 'z'}),
    (bayes_fuzzy_rd, {'y': 'y', 'treat': 'd', 'running': 'x',
                      'cutoff': 0.0, 'bandwidth': 0.5}),
    (bayes_hte_iv, {'y': 'y', 'treat': 'd', 'instrument': 'z',
                    'effect_modifiers': ['x_mod']}),
])
def test_invalid_inference_raises(fn, kwargs):
    """All Bayesian estimators must reject unknown inference modes."""
    if fn is bayes_did:
        df = _did_data()
    elif fn is bayes_rd:
        df = _rd_data()
    elif fn is bayes_iv:
        df = _iv_data()
    elif fn is bayes_fuzzy_rd:
        df = _fuzzy_data()
    else:
        df = _hte_iv_data()

    with pytest.raises(ValueError, match='inference'):
        fn(df, inference='laplace',
           draws=50, tune=50, chains=1, progressbar=False,
           **kwargs)
