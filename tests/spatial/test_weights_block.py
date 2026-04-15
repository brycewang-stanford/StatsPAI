import numpy as np
from statspai.spatial.weights.block import block_weights


def test_block_weights_same_regime_are_neighbours():
    regimes = np.array(["A", "A", "B", "B", "B"])
    w = block_weights(regimes)
    assert sorted(w.neighbors[0]) == [1]
    assert sorted(w.neighbors[2]) == [3, 4]


def test_block_weights_singleton_isolated():
    regimes = np.array(["A", "B", "C"])
    w = block_weights(regimes)
    assert w.islands == [0, 1, 2]
