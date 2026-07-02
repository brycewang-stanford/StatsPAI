"""Analytical parity: network centrality/clustering closed-form identities.

Freeman degree, betweenness, and local clustering have exact values on
canonical graphs (star, triangle, path). These match the graph-theoretic
closed forms to machine precision — e.g. normalized degree = deg_i / (n-1),
local clustering = 2 * triangles_i / (deg_i (deg_i - 1)). Breaks the network
family's zero-parity row.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

# star K1,4: hub 0 connected to leaves 1..4
STAR = np.array(
    [
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
)
TRIANGLE = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
PATH = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # 0-1-2


def test_degree_centrality_is_normalized_degree():
    d = sp.degree_centrality(STAR)
    # hub degree 4/(5-1)=1.0; each leaf 1/4
    assert float(d.iloc[0]) == pytest.approx(1.0, abs=1e-12)
    for i in range(1, 5):
        assert float(d.iloc[i]) == pytest.approx(0.25, abs=1e-12)
    # triangle: every node degree 2/(3-1)=1.0
    dt = sp.degree_centrality(TRIANGLE)
    assert np.allclose(dt.to_numpy(dtype=float), 1.0, atol=1e-12)


def test_betweenness_star_hub_on_all_paths():
    b = sp.betweenness_centrality(STAR)
    assert float(b.iloc[0]) == pytest.approx(1.0, abs=1e-12)  # hub on every path
    assert np.allclose(b.to_numpy(dtype=float)[1:], 0.0, atol=1e-12)  # leaves none
    # path 0-1-2: middle node mediates the single 0<->2 pair -> 1.0 normalized
    bp = sp.betweenness_centrality(PATH)
    assert float(bp.iloc[1]) == pytest.approx(1.0, abs=1e-12)
    assert float(bp.iloc[0]) == pytest.approx(0.0, abs=1e-12)


def test_local_clustering_closed_form():
    # triangle: every node's neighbours are connected -> clustering 1
    ct = sp.clustering(TRIANGLE)
    assert np.allclose(ct.to_numpy(dtype=float), 1.0, atol=1e-12)
    # star / path: no triangles -> clustering 0 everywhere
    assert np.allclose(sp.clustering(STAR).to_numpy(dtype=float), 0.0, atol=1e-12)
    assert np.allclose(sp.clustering(PATH).to_numpy(dtype=float), 0.0, atol=1e-12)
