import numpy as np
import pytest
from statspai.spatial.weights.distance import (
    knn_weights, distance_band, kernel_weights,
)


def test_knn_basic():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    w = knn_weights(coords, k=1)
    assert w.neighbors[0] == [1]
    assert w.neighbors[1] in ([0], [2])
    assert w.neighbors[3] == [2]


def test_distance_band_binary():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    w = distance_band(coords, threshold=1.5, binary=True)
    assert sorted(w.neighbors[0]) == [1]
    assert sorted(w.neighbors[2]) == []


def test_kernel_gaussian_weights_decay():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    w = kernel_weights(coords, bandwidth=1.0, kernel="gaussian", fixed=True)
    w0 = dict(zip(w.neighbors[0], w._weights[0]))
    assert w0[1] > w0[2]


def test_knn_rejects_bad_k():
    coords = np.zeros((3, 2))
    with pytest.raises(ValueError):
        knn_weights(coords, k=5)
