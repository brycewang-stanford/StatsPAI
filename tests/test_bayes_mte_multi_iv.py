"""Tests for ``sp.bayes_mte(instrument=list)`` — multi-instrument MTE."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianMTEResult, bayes_mte

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _multi_iv_dgp(n, seed, n_instruments=2):
    rng = np.random.default_rng(seed)
    Zs = [rng.normal(size=n) for _ in range(n_instruments)]
    U = rng.normal(size=n)
    pi = [0.6, 0.4, -0.3, 0.2][:n_instruments]
    linpred = sum(c * Z for c, Z in zip(pi, Zs)) + 0.3 * U
    D = (linpred > 0).astype(float)
    Y = 1.0 + 1.5 * D + 0.3 * rng.normal(size=n)
    df = pd.DataFrame({'y': Y, 'd': D})
    for i, Z in enumerate(Zs):
        df[f'z{i+1}'] = Z
    return df


@pytest.fixture
def two_iv_data():
    return _multi_iv_dgp(500, seed=901, n_instruments=2)


@pytest.fixture
def three_iv_data():
    return _multi_iv_dgp(500, seed=902, n_instruments=3)


# ---------------------------------------------------------------------------
# Scalar back-compat — single string instrument still works
# ---------------------------------------------------------------------------


def test_scalar_instrument_still_accepted(two_iv_data):
    r = bayes_mte(two_iv_data, y='y', treat='d', instrument='z1',
                  draws=300, tune=300, chains=2, progressbar=False)
    assert isinstance(r, BayesianMTEResult)
    assert r.model_info['n_instruments'] == 1
    # `instruments` is always a list — downstream code doesn't branch on type
    assert r.model_info['instruments'] == ['z1']
    # Verify the legacy typed `instrument` key is NOT stored (dropped
    # in v0.9.11 per round-B review — type varied between str/list).
    assert 'instrument' not in r.model_info


def test_single_element_list_equivalent_to_scalar(two_iv_data):
    """Passing ['z1'] should produce a posterior statistically
    indistinguishable from passing 'z1'. Under shared random_state,
    NUTS still sees slightly different graph shapes (scalar pi_Z vs
    shape-1 pi_Z) which can nudge adaptive step sizes. We therefore
    assert the difference is small *relative to the posterior SD*
    rather than an absolute tolerance, which would pass even if the
    posterior drifted by a full SD (cf. v0.9.11 round-B review)."""
    r_scalar = bayes_mte(two_iv_data, y='y', treat='d', instrument='z1',
                         draws=400, tune=400, chains=2, progressbar=False,
                         random_state=101)
    r_list = bayes_mte(two_iv_data, y='y', treat='d', instrument=['z1'],
                       draws=400, tune=400, chains=2, progressbar=False,
                       random_state=101)
    ate_sd = max(r_scalar.posterior_sd, 1e-6)
    relative_diff = abs(r_scalar.ate - r_list.ate) / ate_sd
    assert relative_diff < 0.5, (
        f"scalar ATE {r_scalar.ate:.3f} vs list ATE {r_list.ate:.3f}; "
        f"|Δ| / SD = {relative_diff:.3f} should be << 1"
    )


# ---------------------------------------------------------------------------
# 2- and 3-instrument fits
# ---------------------------------------------------------------------------


def test_two_instrument_fit(two_iv_data):
    r = bayes_mte(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                  draws=300, tune=300, chains=2, progressbar=False,
                  random_state=13)
    assert r.model_info['n_instruments'] == 2
    assert r.model_info['instruments'] == ['z1', 'z2']
    # True ATE=1.5; recovery should be reasonable on n=500
    assert abs(r.ate - 1.5) < 0.8


def test_three_instrument_fit(three_iv_data):
    r = bayes_mte(three_iv_data, y='y', treat='d',
                  instrument=['z1', 'z2', 'z3'],
                  draws=300, tune=300, chains=2, progressbar=False,
                  random_state=14)
    assert r.model_info['n_instruments'] == 3


def test_joint_first_stage_multi_iv(two_iv_data):
    r = bayes_mte(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                  first_stage='joint',
                  draws=250, tune=250, chains=2, progressbar=False,
                  random_state=15)
    # pi_Z should be shape (2,) in the trace
    pi_Z_shape = r.trace.posterior['pi_Z'].values.shape
    assert pi_Z_shape[-1] == 2


def test_hv_latent_multi_iv(two_iv_data):
    r = bayes_mte(two_iv_data, y='y', treat='d', instrument=['z1', 'z2'],
                  mte_method='hv_latent',
                  draws=250, tune=250, chains=2, progressbar=False,
                  random_state=16)
    assert r.model_info['mte_method'] == 'hv_latent'
    assert r.model_info['n_instruments'] == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_instrument_list_raises(two_iv_data):
    with pytest.raises(ValueError, match='at least one'):
        bayes_mte(two_iv_data, y='y', treat='d', instrument=[],
                  draws=50, tune=50, chains=1, progressbar=False)


def test_missing_instrument_column_raises(two_iv_data):
    with pytest.raises(ValueError, match='not found'):
        bayes_mte(two_iv_data, y='y', treat='d',
                  instrument=['z1', 'not_a_column'],
                  draws=50, tune=50, chains=1, progressbar=False)
