"""Smoke tests for FOCaL + Cluster-CATE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def functional_data():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 3))
    D = (rng.uniform(size=n) < 0.5).astype(int)
    # Functional outcome at t = 0, 1, ..., 4
    y_cols = []
    df = pd.DataFrame({
        'd': D,
        'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2],
    })
    for t in range(5):
        col = f'y_t{t}'
        df[col] = (
            X[:, 0] + X[:, 1] + (1.0 + 0.2 * t) * D
            + 0.4 * rng.standard_normal(n)
        )
        y_cols.append(col)
    return df, y_cols


@pytest.fixture
def cluster_cate_data():
    rng = np.random.default_rng(1)
    n = 400
    X = rng.standard_normal((n, 4))
    D = (rng.uniform(size=n) < 0.5).astype(int)
    # CATE depends on x1: τ = 0.5 + x1
    cate = 0.5 + X[:, 0]
    Y = X[:, 1] + cate * D + rng.standard_normal(n) * 0.5
    return pd.DataFrame({
        'y': Y, 'treat': D,
        'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2], 'x4': X[:, 3],
    })


def test_focal_cate(functional_data):
    df, y_cols = functional_data
    res = sp.focal_cate(
        df, y_columns=y_cols, treat='d', covariates=['x1', 'x2', 'x3'],
    )
    assert isinstance(res, sp.FunctionalCATEResult)
    assert res.cate_grid.shape == (len(df), 5)
    # Mean CATE should be around (1.0 + 0.2*2) ≈ 1.4
    assert 0.5 < res.cate_grid.mean() < 2.5


def test_cluster_cate(cluster_cate_data):
    res = sp.cluster_cate(
        cluster_cate_data, y='y', treat='treat',
        covariates=['x1', 'x2', 'x3', 'x4'], n_clusters=4,
    )
    assert isinstance(res, sp.ClusterCATEResult)
    assert res.n_clusters > 0
    assert 'cate' in res.cluster_table.columns
    # CATE should vary across clusters (because true CATE = 0.5 + x1)
    assert res.cluster_table['cate'].std() > 0


def test_cluster_cate_invalid_treat():
    df = pd.DataFrame({
        'y': np.random.randn(20), 'treat': np.random.randn(20),
        'x': np.random.randn(20),
    })
    with pytest.raises(ValueError, match="binary"):
        sp.cluster_cate(df, y='y', treat='treat', covariates=['x'])
