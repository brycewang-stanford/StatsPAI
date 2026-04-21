"""Smoke tests for v0.10 RDD frontier estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def rd_data():
    rng = np.random.default_rng(0)
    n = 400
    R = rng.uniform(-1, 1, size=n)
    Rn = R + 0.3 * rng.standard_normal(n)
    treat = (R >= 0).astype(int)
    Y = 0.5 * R + 1.5 * treat + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({
        'y': Y, 'r': R, 'rn': Rn,
        'x1': rng.standard_normal(n), 'x2': rng.standard_normal(n),
    })
    return df


@pytest.fixture
def multi_score_data():
    rng = np.random.default_rng(2)
    n = 500
    r1 = rng.uniform(-1, 1, size=n)
    r2 = rng.uniform(-1, 1, size=n)
    treat = ((r1 >= 0) & (r2 >= 0)).astype(int)
    Y = 0.3 * r1 + 0.3 * r2 + 2.0 * treat + rng.standard_normal(n) * 0.4
    return pd.DataFrame({'y': Y, 'r1': r1, 'r2': r2})


def test_rd_interference(rd_data):
    res = sp.rd_interference(
        rd_data, y='y', running='r', neighbour_running='rn',
        cutoff=0.0, bandwidth=0.6,
    )
    assert isinstance(res, sp.RDInterferenceResult)
    # True direct effect = 1.5
    assert 0.5 < res.direct_effect < 3.0


def test_rd_multi_score(multi_score_data):
    res = sp.rd_multi_score(
        multi_score_data, y='y', running_vars=['r1', 'r2'],
        cutoffs=[0.0, 0.0], bandwidth=0.6,
    )
    assert isinstance(res, sp.MultiScoreRDResult)
    assert 0 <= res.boundary_share <= 1


def test_rd_distribution(rd_data):
    res = sp.rd_distribution(
        rd_data, y='y', running='r', cutoff=0.0,
        quantiles=np.array([0.25, 0.5, 0.75]), bandwidth=0.6,
    )
    assert isinstance(res, sp.DistRDResult)
    assert len(res.qte) == 3


def test_rd_bayes_hte(rd_data):
    res = sp.rd_bayes_hte(
        rd_data, y='y', running='r', covariates=['x1', 'x2'],
        cutoff=0.0, bandwidth=0.6, n_draws=200,
    )
    assert isinstance(res, sp.BayesRDHTEResult)
    assert len(res.cate) == len(rd_data)
    assert 0.5 < res.posterior_mean < 3.0


def test_rd_distributional_design(rd_data):
    res = sp.rd_distributional_design(
        rd_data, y='y', running='r', cutoff=0.0,
        quantiles=np.array([0.25, 0.5, 0.75]), bandwidth=0.6,
    )
    assert isinstance(res, sp.DDDResult)
    assert len(res.rdd_effect) == 3
    assert len(res.rkd_effect) == 3


def test_multi_score_invalid_lengths():
    df = pd.DataFrame({
        'y': np.random.randn(20), 'r1': np.random.randn(20),
    })
    with pytest.raises(ValueError, match="len"):
        sp.rd_multi_score(df, y='y', running_vars=['r1'], cutoffs=[0.0, 0.5])
