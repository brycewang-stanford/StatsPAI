"""Smoke tests for v0.10 conformal frontier estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def cf_data():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 3))
    D = (rng.uniform(size=n) < 0.4).astype(int)
    Y = X[:, 0] + X[:, 1] + 2.0 * D + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({
        'y': Y, 'treat': D,
        'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2],
        'group': np.where(X[:, 2] > 0, 'A', 'B'),
    })
    return df


def test_conformal_density(cf_data):
    res = sp.conformal_density_ite(
        cf_data, y='y', treat='treat',
        covariates=['x1', 'x2', 'x3'], alpha=0.1,
    )
    assert res.intervals.shape == (len(cf_data), 2)
    assert (res.intervals[:, 1] >= res.intervals[:, 0]).all()
    # Mean ITE should be near 2.0
    assert 1.0 < res.point_estimate.mean() < 3.0


def test_conformal_multidp(cf_data):
    df = cf_data.copy()
    df['y2'] = df['y'] + np.random.default_rng(1).standard_normal(len(df))
    df['d2'] = (np.random.default_rng(2).uniform(size=len(df)) < 0.5).astype(int)
    res = sp.conformal_ite_multidp(
        df, y_per_stage=['y', 'y2'],
        treat_per_stage=['treat', 'd2'],
        history_per_stage=[['x1', 'x2'], ['x1', 'x2', 'x3']],
        alpha=0.1,
    )
    assert res.n_stages == 2
    assert len(res.intervals_per_stage) == 2
    assert res.cumulative_interval.shape == (len(df), 2)


def test_conformal_debiased(cf_data):
    res = sp.conformal_debiased_ml(
        cf_data, y='y', treat='treat',
        covariates=['x1', 'x2', 'x3'], alpha=0.1, n_folds=3,
    )
    assert res.intervals.shape == (len(cf_data), 2)
    assert 1.0 < res.point_estimate.mean() < 3.0


def test_conformal_fair(cf_data):
    res = sp.conformal_fair_ite(
        cf_data, y='y', treat='treat',
        covariates=['x1', 'x2', 'x3', 'group'],
        protected='group', alpha=0.1,
    )
    assert res.intervals.shape == (len(cf_data), 2)
    assert set(res.group_widths.keys()) == {'A', 'B'}


def test_conformal_density_invalid_treat(cf_data):
    df = cf_data.copy()
    df['treat'] = np.random.uniform(size=len(df))  # continuous, not binary
    with pytest.raises(ValueError, match="binary"):
        sp.conformal_density_ite(
            df, y='y', treat='treat', covariates=['x1', 'x2', 'x3']
        )
