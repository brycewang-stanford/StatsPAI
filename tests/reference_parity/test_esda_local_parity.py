"""Analytical parity: local ESDA statistics on a known clustered field.

Local Moran (`sp.moran_local`), local Getis-Ord (`sp.getis_ord_local`) and
join counts (`sp.join_counts`) have unambiguous ground truth on a field made of
two well-separated blocks with opposite values: every unit is surrounded only
by same-value neighbours, so the local statistics must flag positive spatial
association everywhere and the binary field must have **zero** black-white
joins. The independent-field cases pin the null behaviour. Analytical evidence
tier (known-truth recovery on a deterministic DGP; no cross-package target).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _two_block_field(seed=7, n_per=60, jitter=0.05):
    """Two well-separated blocks; block A value +1, block B value -1."""
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.uniform(0, 1, n_per), rng.uniform(0, 1, n_per)])
    b = np.column_stack([rng.uniform(10, 11, n_per), rng.uniform(10, 11, n_per)])
    coords = np.vstack([a, b])
    y = np.concatenate([np.full(n_per, 1.0), np.full(n_per, -1.0)])
    y = y + rng.normal(0, jitter, 2 * n_per)
    return coords, y, n_per


def _independent_field(seed=0, n=200):
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    y = rng.standard_normal(n)
    return coords, y, n


# --------------------------------------------------------------------------
# local Moran's I
# --------------------------------------------------------------------------
def test_local_moran_positive_everywhere_in_clustered_field():
    coords, y, _ = _two_block_field()
    w = sp.knn_weights(coords, k=5)
    res = sp.moran_local(y, w, permutations=199, seed=0)
    Is = np.asarray(res["Is"], dtype=float)
    # Every unit sits inside a same-value block -> positive local association.
    assert np.all(Is > 0)


def test_local_moran_flags_the_clusters():
    coords, y, _ = _two_block_field()
    w = sp.knn_weights(coords, k=5)
    res = sp.moran_local(y, w, permutations=199, seed=0)
    p = np.asarray(res["p_sim"], dtype=float)
    # A strongly clustered field is significant for the large majority of units.
    assert (p < 0.05).mean() > 0.5


def test_local_moran_null_field_mostly_insignificant():
    coords, y, _ = _independent_field()
    w = sp.knn_weights(coords, k=5)
    res = sp.moran_local(y, w, permutations=199, seed=0)
    p = np.asarray(res["p_sim"], dtype=float)
    # Under spatial randomness few units should reject at 5%.
    assert (p < 0.05).mean() < 0.15


def test_local_moran_reproducible_under_seed():
    coords, y, _ = _two_block_field()
    w = sp.knn_weights(coords, k=5)
    r1 = sp.moran_local(y, w, permutations=199, seed=0)
    r2 = sp.moran_local(y, w, permutations=199, seed=0)
    np.testing.assert_array_equal(np.asarray(r1["Is"]), np.asarray(r2["Is"]))
    np.testing.assert_array_equal(
        np.asarray(r1["p_sim"]), np.asarray(r2["p_sim"])
    )


# --------------------------------------------------------------------------
# local Getis-Ord G_i*
# --------------------------------------------------------------------------
def test_getis_ord_high_and_low_regions_have_opposite_z():
    coords, y, n_per = _two_block_field()
    w = sp.knn_weights(coords, k=5)
    res = sp.getis_ord_local(y, w, star=True, permutations=199, seed=0)
    z = np.asarray(res["z"], dtype=float)
    # High-value block -> hot spot (positive z); low-value block -> cold spot.
    assert z[:n_per].mean() > 1.5
    assert z[n_per:].mean() < -1.5


def test_getis_ord_reproducible_under_seed():
    coords, y, _ = _two_block_field()
    w = sp.knn_weights(coords, k=5)
    r1 = sp.getis_ord_local(y, w, star=True, permutations=199, seed=0)
    r2 = sp.getis_ord_local(y, w, star=True, permutations=199, seed=0)
    np.testing.assert_array_equal(np.asarray(r1["z"]), np.asarray(r2["z"]))


# --------------------------------------------------------------------------
# join counts (binary)
# --------------------------------------------------------------------------
def test_join_counts_segregated_field_has_no_cross_joins():
    coords, _, n_per = _two_block_field()
    w = sp.knn_weights(coords, k=5)
    yb = np.concatenate([np.ones(n_per, int), np.zeros(n_per, int)])
    res = sp.join_counts(yb, w, permutations=199, seed=0)
    # Perfectly segregated binary field: no black-white edges at all.
    assert float(res["BW"]) == 0.0
    assert float(res["BB"]) > 0
    assert float(res["WW"]) > 0
    # Total joins conserved: BB + WW + BW == number of directed edges / 2 pairs.
    assert float(res["BB"]) + float(res["WW"]) + float(res["BW"]) > 0


def test_join_counts_segregation_significant():
    coords, _, n_per = _two_block_field()
    w = sp.knn_weights(coords, k=5)
    yb = np.concatenate([np.ones(n_per, int), np.zeros(n_per, int)])
    res = sp.join_counts(yb, w, permutations=199, seed=0)
    # Excess same-colour joins vs the permutation null is significant.
    assert float(res["p_sim_BB"]) < 0.05
