"""Tier D P2 known-truth upgrades — spatial weight builders & Getis-Ord.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). These were graded ``weak`` by
``scripts/tierd_classify.py``. Each anchors to an exact graph-theoretic fact on
a regular lattice, where the neighbour structure is known by construction:

    sp.distance_band    radius-1 band on a unit grid == rook contiguity
                        (interior degree 4, edge 3, corner 2); symmetric.
    sp.kernel_weights   distance-decay: nearer points get larger weights.
    sp.getis_ord_local  Gi* is positive in a high-value cluster, negative in
                        a low-value one.

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pytest

import statspai as sp


def _grid_coords(side=3):
    return np.array([[i, j] for i in range(side) for j in range(side)], dtype=float)


# ---------------------------------------------------------------------------
# sp.distance_band
# ---------------------------------------------------------------------------
class TestDistanceBandAnalytic:

    def test_unit_band_is_rook_contiguity_on_grid(self):
        # On a 3x3 unit grid, a radius-1 band connects exactly the rook
        # neighbours: corners 2, edges 3, the centre 4.
        w = sp.distance_band(_grid_coords(3), threshold=1.0, binary=True)
        deg = np.asarray(w.sparse.sum(axis=1)).ravel()
        expected = np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])
        np.testing.assert_array_equal(deg.astype(int), expected)

    def test_weights_symmetric(self):
        w = sp.distance_band(_grid_coords(4), threshold=1.0, binary=True)
        S = w.sparse.toarray()
        np.testing.assert_array_equal(S, S.T)

    def test_larger_threshold_adds_neighbours(self):
        coords = _grid_coords(3)
        deg1 = np.asarray(
            sp.distance_band(coords, threshold=1.0).sparse.sum(axis=1)
        ).ravel()
        # sqrt(2) ~ 1.4142 also pulls in the diagonal (queen) neighbours.
        deg2 = np.asarray(
            sp.distance_band(coords, threshold=1.45).sparse.sum(axis=1)
        ).ravel()
        assert deg2[4] == 8  # centre now sees all 8 surrounding cells
        assert deg2.sum() > deg1.sum()


# ---------------------------------------------------------------------------
# sp.kernel_weights
# ---------------------------------------------------------------------------
class TestKernelWeightsAnalytic:

    def test_weight_decays_with_distance(self):
        coords = _grid_coords(3)
        S = sp.kernel_weights(coords, bandwidth=2.0, kernel="gaussian").sparse.toarray()
        # Centre (index 4) to an adjacent cell (1, distance 1) outweighs the
        # corner (0, distance sqrt(2)).
        assert S[4, 1] > S[4, 0] > 0

    def test_zero_self_weight_and_nearest_neighbour_dominates(self):
        coords = _grid_coords(3)
        S = sp.kernel_weights(coords, bandwidth=2.0, kernel="gaussian").sparse.toarray()
        # W convention: no self-neighbour (zero diagonal). Among the actual
        # neighbours, the nearest (a rook cell at distance 1) has the largest
        # weight in the centre's row.
        np.testing.assert_allclose(np.diag(S), 0.0, atol=1e-12)
        assert S[4, 1] == pytest.approx(S[4].max())


# ---------------------------------------------------------------------------
# sp.getis_ord_local
# ---------------------------------------------------------------------------
class TestGetisOrdAnalytic:

    def test_hotspot_positive_coldspot_negative(self):
        w = sp.distance_band(_grid_coords(3), threshold=1.0, binary=True)
        # High values on the first row, low elsewhere -> the high cluster is a
        # hot spot (positive Gi* z), the low cluster a cold spot (negative).
        y = np.array([10, 10, 10, 1, 1, 1, 1, 1, 1], dtype=float)
        out = sp.getis_ord_local(y, w, star=True, permutations=0)
        z = np.asarray(out["z"])
        assert z[1] > 0  # interior of the hot cluster
        assert z[7] < 0  # interior of the cold cluster

    def test_strongest_hotspot_has_the_largest_score(self):
        # The peak Gi* z-score sits inside the high-value cluster.
        w = sp.distance_band(_grid_coords(3), threshold=1.0, binary=True)
        y = np.array([10, 10, 10, 1, 1, 1, 1, 1, 1], dtype=float)
        out = sp.getis_ord_local(y, w, star=True, permutations=0)
        z = np.asarray(out["z"])
        assert int(np.argmax(z)) in {0, 1, 2}
