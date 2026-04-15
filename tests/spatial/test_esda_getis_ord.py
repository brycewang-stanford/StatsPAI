import numpy as np
from statspai.spatial.esda import getis_ord_g, getis_ord_local
from statspai.spatial.weights import knn_weights


def test_getis_ord_g_hotspot():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(80, 2))
    # one cluster at origin
    y = np.where(np.linalg.norm(coords, axis=1) < 0.3, 10.0, 1.0)
    w = knn_weights(coords, k=5); w.transform = "R"
    res = getis_ord_g(y, w, permutations=499, seed=0)
    assert res.p_sim < 0.1


def test_getis_ord_local_returns_array():
    rng = np.random.default_rng(1)
    coords = rng.uniform(size=(30, 2))
    y = rng.uniform(size=30)
    w = knn_weights(coords, k=3); w.transform = "R"
    out = getis_ord_local(y, w, star=True)
    assert out["Gs"].shape == (30,)
