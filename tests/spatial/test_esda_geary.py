import numpy as np
from statspai.spatial.esda import geary
from statspai.spatial.weights import knn_weights


def test_geary_smooth_data_c_below_one():
    rng = np.random.default_rng(42)
    coords = rng.uniform(size=(100, 2))
    y = np.linalg.norm(coords - 0.5, axis=1) + 0.05 * rng.standard_normal(100)
    w = knn_weights(coords, k=6); w.transform = "R"
    res = geary(y, w, permutations=499, seed=42)
    assert res.C < 0.85
    assert res.p_sim < 0.05


def test_geary_iid_near_one():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(100, 2))
    y = rng.standard_normal(100)
    w = knn_weights(coords, k=4); w.transform = "R"
    res = geary(y, w, permutations=499, seed=0)
    assert abs(res.C - 1.0) < 0.25
