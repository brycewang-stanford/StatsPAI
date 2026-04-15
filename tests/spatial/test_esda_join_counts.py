import numpy as np
from statspai.spatial.esda import join_counts
from statspai.spatial.weights import knn_weights


def test_join_counts_clustered_bb_high():
    # cluster of 1s in one half, 0s in other — many BB joins
    rng = np.random.default_rng(0)
    coords_a = rng.uniform(low=0, high=0.4, size=(40, 2))
    coords_b = rng.uniform(low=0.6, high=1.0, size=(40, 2))
    coords = np.vstack([coords_a, coords_b])
    y = np.array([1] * 40 + [0] * 40)
    w = knn_weights(coords, k=4); w.transform = "B"
    out = join_counts(y, w, permutations=299, seed=0)
    assert out["BB"] > 0
    assert out["p_sim_BB"] < 0.1


def test_join_counts_rejects_non_binary():
    import pytest
    w_coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    w = knn_weights(w_coords, k=1)
    with pytest.raises(ValueError):
        join_counts(np.array([0, 1, 2]), w)
