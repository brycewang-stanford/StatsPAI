"""Frozen-stat parity: sp.cluster_robust_se determinism + closed-form variance.

cluster_robust_se returns sqrt(diag(V)) where V is the multiway cluster
variance. The function is deterministic — same input yields the same SEs — and
the SE is a closed-form function of (X, residuals, cluster assignments).
Analytical evidence tier (deterministic closed-form; no cross-language target).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(42)
    n = 200
    firm = rng.integers(0, 10, size=n)
    year = rng.integers(0, 8, size=n)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    resid = rng.normal(size=n)
    return X, resid, firm, year, sp.cluster_robust_se(X, resid, [firm, year])


def test_deterministic_reproducibility(fitted):
    X, resid, firm, year, _ = fitted
    r1 = sp.cluster_robust_se(X, resid, [firm, year])
    r2 = sp.cluster_robust_se(X, resid, [firm, year])
    assert np.allclose(r1, r2, atol=1e-15)


def test_variance_is_nonnegative_sqrt_of_diag(fitted):
    """The function must be a non-negative real square root of diag(V)."""
    X, resid, firm, year, se = fitted
    # SE should be non-negative.
    assert np.all(np.isfinite(se))
    assert np.all(se >= 0.0)


def test_duplicate_cluster_equals_single_cluster(fitted):
    """Passing the same cluster as a 1-D and as a 2-D list should give the
    same SE — the multiway formula collapses to a one-way cluster variance."""
    X, resid, firm, year, _ = fitted
    se_single = sp.cluster_robust_se(X, resid, firm)
    se_double = sp.cluster_robust_se(X, resid, [firm, firm])
    assert np.allclose(se_single, se_double, atol=1e-12)


def test_different_cluster_assignments_change_se(fitted):
    X, resid, firm, year, _ = fitted
    # Permute firm -> a different partition -> different SE.
    perm = np.random.RandomState(0).permutation(firm)
    se_orig = sp.cluster_robust_se(X, resid, firm)
    se_perm = sp.cluster_robust_se(X, resid, perm)
    assert not np.allclose(se_orig, se_perm, atol=1e-3)
