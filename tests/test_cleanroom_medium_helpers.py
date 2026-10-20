"""Clean-room helpers of the MEDIUM provenance rows (docs/dev/cleanroom/).

Each test asserts a documented rule, an algebraic identity or a published
formula on a hand-computable case.
"""

from __future__ import annotations

import numpy as np
import pytest

from statspai.forest._grf_family import observation_weights as family_weights
from statspai.forest._grf_fit import observation_weights as fit_weights
from statspai.inference.jackknife import _matrix_power_sym
from statspai.mendelian.extras import _mbe_bandwidth

# --------------------------------------------------------------- grf weights
CLUSTERS = np.array(["a", "a", "a", "b", "c", "c"])


def test_equalized_clusters_get_equal_total_weight():
    # grf ?causal_forest: "each observation is given weight 1/cluster size,
    # so that the total weight of each cluster is the same".
    w = fit_weights(CLUSTERS, True, CLUSTERS.size)
    totals = {c: w[CLUSTERS == c].sum() for c in np.unique(CLUSTERS)}
    assert w.sum() == pytest.approx(1.0, rel=1e-15)
    assert all(t == pytest.approx(1.0 / 3.0, rel=1e-15) for t in totals.values())


def test_without_equalizing_every_observation_weighs_the_same():
    w = fit_weights(CLUSTERS, False, CLUSTERS.size)
    np.testing.assert_allclose(w, np.full(6, 1.0 / 6.0), rtol=1e-15)


def test_sample_weights_are_used_as_given_and_take_precedence():
    sw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
    for equalize in (False, True):
        np.testing.assert_allclose(
            family_weights(CLUSTERS, equalize, sw, 6), sw / sw.sum(), rtol=1e-15
        )


# ------------------------------------------------------- CR2 matrix power
def test_inverse_square_root_on_a_singular_block_is_a_generalized_inverse():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(5, 3))
    A = B @ B.T  # rank 3
    half_inv = _matrix_power_sym(A, -0.5)
    # A^{-1/2} A A^{-1/2} is the orthogonal projector onto range(A).
    P = half_inv @ A @ half_inv
    Q, _ = np.linalg.qr(B)
    np.testing.assert_allclose(P, Q @ Q.T, atol=1e-12)
    # and (A^{1/2})^2 = A
    root = _matrix_power_sym(A, 0.5)
    np.testing.assert_allclose(root @ root, A, atol=1e-12)


def test_eigenvalues_at_or_below_the_threshold_are_dropped():
    D = np.diag([4.0, 1e-13, 1e-11])
    np.testing.assert_allclose(
        np.diag(_matrix_power_sym(D, -1.0)), [0.25, 0.0, 1e11], rtol=1e-14
    )
    np.testing.assert_allclose(
        np.diag(_matrix_power_sym(D, -1.0, tol=-10.0)), [0.25, 0.0, 0.0], rtol=1e-14
    )


# ------------------------------------------------- mode-based bandwidth
def test_mbe_bandwidth_is_hartwig_eq_7():
    # ratios 1, 2, 3, 4, 10: sd = sqrt(50/4) = 3.5355..., MAD = median|r - 3|
    # = 1, so 1.4826 * MAD < sd and s = 0.9 * 1.4826 / 5^(1/5).
    r = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    expected = 0.5 * 0.9 * 1.4826 / 5**0.2
    assert _mbe_bandwidth(r, 0.5) == pytest.approx(expected, rel=1e-15)
    # with the SD binding: ratios 0, 0, 1, 1 (sd = sqrt(1/3), MAD = 0.5)
    r2 = np.array([0.0, 0.0, 1.0, 1.0])
    assert _mbe_bandwidth(r2, 1.0) == pytest.approx(
        0.9 * np.sqrt(1.0 / 3.0) / 4**0.2, rel=1e-15
    )
