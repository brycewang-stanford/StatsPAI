import numpy as np
import pytest
from scipy import sparse
from statspai.spatial.weights.core import W


def test_w_from_neighbors_dict():
    neighbors = {0: [1, 2], 1: [0], 2: [0]}
    w = W(neighbors)
    assert w.n == 3
    assert w.sparse.shape == (3, 3)
    assert w.sparse[0, 1] == 1.0
    assert w.sparse[1, 2] == 0.0


def test_w_row_standardise():
    w = W({0: [1, 2], 1: [0], 2: [0]})
    w.transform = "R"
    row_sums = np.asarray(w.sparse.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0])


def test_w_islands_detected():
    w = W({0: [1], 1: [0], 2: []})
    assert w.islands == [2]


def test_w_bad_input_raises():
    with pytest.raises(TypeError):
        W("not a dict")
