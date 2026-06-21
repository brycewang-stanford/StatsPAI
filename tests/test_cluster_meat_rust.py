"""Parity tests for the Rust ``cluster_meat`` kernel vs the numba reference.

Skips automatically when ``statspai_hdfe`` is not built into the
current venv (i.e. ``maturin develop --release`` has not been run).
"""

from __future__ import annotations

import numpy as np
import pytest

from statspai.core._numba_kernels import (
    _HAS_RUST_CLUSTER,
    _cluster_meat_sorted,
    cluster_meat,
)

try:
    import statspai_hdfe  # noqa: F401
except ImportError:  # pragma: no cover
    statspai_hdfe = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(
    not _HAS_RUST_CLUSTER,
    reason="statspai_hdfe (Rust) not built; run "
    "`pip install maturin && cd rust/statspai_hdfe && maturin develop --release`",
)

# Floats from a Rayon-parallel reduction can differ from a sequential
# numba sum at the last few ulp. ``atol=1e-10`` matches the contract
# the rest of the Rust crate ships under (see ``rust/statspai_hdfe/README.md``).
_ATOL = 1e-10


def _sorted_inputs(X, residuals, cluster_ids):
    order = np.argsort(cluster_ids, kind="mergesort")
    X_s = np.ascontiguousarray(X[order], dtype=np.float64)
    r_s = np.ascontiguousarray(residuals[order], dtype=np.float64)
    c_s = cluster_ids[order]
    change = np.empty(len(c_s), dtype=np.bool_)
    change[0] = True
    change[1:] = c_s[1:] != c_s[:-1]
    starts = np.where(change)[0].astype(np.intp)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(c_s)
    return X_s, r_s, starts, ends


@pytest.mark.parametrize(
    "n,k,n_clusters,seed",
    [
        (200, 3, 30, 0),
        (1_000, 5, 50, 1),
        (5_000, 8, 200, 2),
        (10_000, 12, 1_000, 3),
    ],
)
def test_rust_matches_numba(n, k, n_clusters, seed):
    """Rust + numba kernels must agree to ``_ATOL`` on standard DGPs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    residuals = rng.normal(size=n) * 0.7
    cluster_ids = rng.integers(0, n_clusters, size=n)

    rust_meat = cluster_meat(X, residuals, cluster_ids)

    X_s, r_s, starts, ends = _sorted_inputs(X, residuals, cluster_ids)
    numba_meat = _cluster_meat_sorted(X_s, r_s, starts, ends)

    np.testing.assert_allclose(rust_meat, numba_meat, atol=_ATOL, rtol=0)


def test_rust_returns_symmetric_matrix():
    rng = np.random.default_rng(42)
    n, k, G = 500, 4, 25
    X = rng.normal(size=(n, k))
    residuals = rng.normal(size=n)
    cluster_ids = rng.integers(0, G, size=n)
    meat = cluster_meat(X, residuals, cluster_ids)
    np.testing.assert_allclose(meat, meat.T, atol=1e-14)


def test_rust_handles_singleton_clusters():
    """Each row in its own cluster: meat = sum_i (x_i x_i.T) * r_i^2."""
    rng = np.random.default_rng(7)
    n, k = 100, 3
    X = rng.normal(size=(n, k))
    residuals = rng.normal(size=n)
    cluster_ids = np.arange(n)  # singleton clusters

    rust_meat = cluster_meat(X, residuals, cluster_ids)
    expected = (X * residuals[:, None]).T @ (X * residuals[:, None])
    np.testing.assert_allclose(rust_meat, expected, atol=_ATOL)


def test_rust_handles_one_giant_cluster():
    """All rows in one cluster: meat = (X.T r)(X.T r).T."""
    rng = np.random.default_rng(11)
    n, k = 1_000, 5
    X = rng.normal(size=(n, k))
    residuals = rng.normal(size=n)
    cluster_ids = np.zeros(n, dtype=np.int64)

    rust_meat = cluster_meat(X, residuals, cluster_ids)
    s = X.T @ residuals
    expected = np.outer(s, s)
    np.testing.assert_allclose(rust_meat, expected, atol=_ATOL)


def test_rust_kernel_rejects_non_c_contiguous():
    """Direct kernel call should raise on Fortran-order ``X``."""
    rng = np.random.default_rng(0)
    X_f = np.asfortranarray(rng.normal(size=(20, 3)))
    r = rng.normal(size=20)
    starts = np.array([0, 5, 12], dtype=np.int64)
    ends = np.array([5, 12, 20], dtype=np.int64)
    with pytest.raises(ValueError, match="C-contiguous"):
        statspai_hdfe.cluster_meat(X_f, r, starts, ends)


def test_rust_kernel_validates_cluster_bounds():
    """Out-of-range cluster bounds should raise rather than panic."""
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(rng.normal(size=(20, 3)), dtype=np.float64)
    r = np.ascontiguousarray(rng.normal(size=20), dtype=np.float64)
    bad_starts = np.array([0, 5], dtype=np.int64)
    bad_ends = np.array([5, 999], dtype=np.int64)  # 999 > n=20
    with pytest.raises(ValueError, match="invalid range"):
        statspai_hdfe.cluster_meat(X, r, bad_starts, bad_ends)
