"""Smoke tests for v0.10 Cluster RCT × Interference suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def matched_pair_data():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(20):
        # Each pair: 2 clusters, 1 treated
        for cl_idx, treated in enumerate([0, 1]):
            cluster_id = f'c{p * 2 + cl_idx}'
            for i in range(15):
                y = 0.3 * p + 1.0 * treated + rng.standard_normal()
                rows.append({
                    'y': y, 'cluster': cluster_id,
                    'treat': treated, 'pair': f'p{p}',
                })
    return pd.DataFrame(rows)


@pytest.fixture
def cross_cluster_data():
    rng = np.random.default_rng(1)
    rows = []
    n_clusters = 30
    for c in range(n_clusters):
        treated = int(rng.uniform() < 0.5)
        nshare = float(rng.uniform())  # share of treated neighbours
        for i in range(20):
            y = 0.5 * treated + 0.3 * nshare + rng.standard_normal()
            rows.append({
                'y': y, 'cluster': f'c{c}', 'treat': treated,
                'nshare': nshare,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_cluster_data():
    rng = np.random.default_rng(2)
    rows = []
    for c in range(15):
        first_t = 0 if c < 5 else (3 if c < 10 else 5)
        for t in range(8):
            post = first_t > 0 and t >= first_t
            y = 0.1 * c + 0.2 * t + (1.0 if post else 0.0) + rng.standard_normal()
            rows.append({
                'y': y, 'cluster': f'c{c}', 'time': t, 'first_treat': first_t,
            })
    return pd.DataFrame(rows)


def test_cluster_matched_pair(matched_pair_data):
    res = sp.cluster_matched_pair(
        matched_pair_data, y='y', cluster='cluster',
        treat='treat', pair='pair',
    )
    assert isinstance(res, sp.MatchedPairResult)
    # True effect ≈ 1.0
    assert 0.0 < res.estimate < 2.0


def test_cluster_cross_interference(cross_cluster_data):
    res = sp.cluster_cross_interference(
        cross_cluster_data, y='y', cluster='cluster',
        treat='treat', neighbour_treat_share='nshare',
    )
    assert isinstance(res, sp.CrossClusterRCTResult)
    # True direct ≈ 0.5
    assert -0.5 < res.direct_effect < 1.5


def test_cluster_staggered_rollout(staggered_cluster_data):
    res = sp.cluster_staggered_rollout(
        staggered_cluster_data, y='y', cluster='cluster',
        time='time', first_treat='first_treat',
        leads=2, lags=2,
    )
    assert isinstance(res, sp.StaggeredClusterRCTResult)
    # True post-ATT ≈ 1.0
    assert -0.5 < res.overall_att < 2.5


def test_dnc_gnn_did(staggered_cluster_data):
    df = staggered_cluster_data.copy()
    rng = np.random.default_rng(3)
    df['nc_y1'] = df['y'] * 0.3 + rng.standard_normal(len(df))
    df['nc_x1'] = rng.standard_normal(len(df))
    res = sp.dnc_gnn_did(
        df, y='y', treat='first_treat', time='time', id='cluster',
        nc_outcome=['nc_y1'], nc_exposure=['nc_x1'], n_boot=20,
    )
    assert isinstance(res, sp.DNCGNNDiDResult)
    assert np.isfinite(res.estimate)


def test_matched_pair_too_few_pairs():
    df = pd.DataFrame({
        'y': [1.0, 2.0], 'cluster': ['c1', 'c2'],
        'treat': [0, 1], 'pair': ['p1', 'p1'],
    })
    with pytest.raises(ValueError, match="at least 2"):
        sp.cluster_matched_pair(df, y='y', cluster='cluster',
                                  treat='treat', pair='pair')
