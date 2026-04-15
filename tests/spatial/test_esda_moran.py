import numpy as np
from statspai.spatial.esda import moran, moran_local
from statspai.spatial.weights import knn_weights


def test_moran_positive_autocorrelation():
    rng = np.random.default_rng(42)
    coords = rng.uniform(size=(100, 2))
    y = np.linalg.norm(coords - 0.5, axis=1) + 0.05 * rng.standard_normal(100)
    w = knn_weights(coords, k=6); w.transform = "R"
    res = moran(y, w, permutations=499, seed=42)
    assert res.I > 0.3
    assert res.p_sim < 0.05


def test_moran_iid_near_zero():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(100, 2))
    y = rng.standard_normal(100)
    w = knn_weights(coords, k=4); w.transform = "R"
    res = moran(y, w, permutations=499, seed=0)
    assert (res.p_sim > 0.05) or (abs(res.I) < 0.1)


def test_moran_summary_prints():
    rng = np.random.default_rng(1)
    coords = rng.uniform(size=(30, 2))
    y = rng.standard_normal(30)
    w = knn_weights(coords, k=3); w.transform = "R"
    res = moran(y, w, permutations=99, seed=1)
    assert "Moran" in res.summary()
