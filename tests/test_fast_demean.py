"""Tests for ``sp.fast.demean`` — multi-way HDFE within-transform.

These tests exercise both the Rust kernel (when the compiled extension
is available) and the NumPy fallback. They are the Phase 1 acceptance
gate for the new HDFE backend.

References (cross-checked against existing implementations):

* Rust ↔ NumPy: bit-equivalent algorithm, expect agreement to 1e-12.
* sp.fast.demean ↔ sp.demean (Absorber): same algorithm, ditto.
* K-way ↔ dummy-variable LSQR: orthogonal projection onto FE complement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

def _make_panel(n_units=100, n_periods=20, seed=0):
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    n = i.size
    x = rng.normal(size=n)
    a = rng.normal(0, 0.5, size=n_units)[i]
    g = rng.normal(0, 0.3, size=n_periods)[t]
    y = 1.0 + 0.3 * x + a + g + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "i": i, "t": t})


def _dummy_ols_residuals(y: np.ndarray, codes_list, atol: float = 1e-12) -> np.ndarray:
    """Reference: project y onto orthogonal complement of FE design matrix
    via sparse LSQR. Used to validate AP convergence."""
    from scipy import sparse
    from scipy.sparse.linalg import lsqr

    n = y.size
    rows = np.arange(n)
    blocks = []
    for codes in codes_list:
        codes = np.asarray(codes, dtype=np.int64)
        G = int(codes.max()) + 1
        D = sparse.csr_matrix((np.ones(n), (rows, codes)), shape=(n, G))
        blocks.append(D)
    D = sparse.hstack(blocks).tocsr()
    beta = lsqr(D, y, atol=atol, btol=atol, iter_lim=20_000)[0]
    return y - D @ beta


# ---------------------------------------------------------------------------
# Correctness vs analytical / brute-force references
# ---------------------------------------------------------------------------

def test_oneway_matches_groupby_mean():
    """Single-FE demean equals y - groupby mean exactly (closed form)."""
    df = _make_panel(seed=1)
    y = df["y"].to_numpy()
    fe = df[["i"]].to_numpy()
    y_dem, info = sp.fast.demean(y, fe, drop_singletons=False)
    expected = y - df.groupby("i")["y"].transform("mean").to_numpy()
    assert np.allclose(y_dem, expected, atol=1e-12)
    assert info.iters[0] == 1
    assert info.converged[0]


def test_twoway_converges_against_existing_demean():
    """Two-way AP must agree with sp.demean (Absorber) to floating-point."""
    df = _make_panel(seed=2)
    y = df["y"].to_numpy()
    fe = df[["i", "t"]].to_numpy()
    y_new, info = sp.fast.demean(y, fe, drop_singletons=False)
    y_old, _ = sp.demean(y, fe, drop_singletons=False)
    assert np.allclose(y_new, y_old, atol=1e-9)
    assert info.converged[0]


def test_threeway_matches_lsqr_reference():
    """Three-way demean = orthogonal projection; agrees with LSQR residuals."""
    rng = np.random.default_rng(3)
    n = 800
    i = rng.integers(0, 30, size=n)
    j = rng.integers(0, 20, size=n)
    k = rng.integers(0, 15, size=n)
    y = rng.normal(size=n)
    y_dem, info = sp.fast.demean(
        y, [i, j, k], drop_singletons=False, max_iter=5_000, tol=1e-12
    )
    y_ref = _dummy_ols_residuals(y, [i, j, k], atol=1e-12)
    # AP and LSQR converge to the same projection up to their own tolerance
    assert np.allclose(y_dem, y_ref, atol=1e-7)


# ---------------------------------------------------------------------------
# Backend cross-check (Rust ↔ NumPy)
# ---------------------------------------------------------------------------

def test_backend_rust_equals_numpy():
    df = _make_panel(seed=4)
    y = df["y"].to_numpy()
    fe = df[["i", "t"]].to_numpy()

    y_np, info_np = sp.fast.demean(y, fe, backend="numpy", drop_singletons=False)
    try:
        y_rs, info_rs = sp.fast.demean(y, fe, backend="rust", drop_singletons=False)
    except RuntimeError:
        pytest.skip("Rust extension unavailable")

    # Bit-equivalent algorithm => identical to floating-point precision
    assert np.allclose(y_np, y_rs, atol=1e-12)
    assert info_rs.backend == "rust"
    assert info_np.backend == "numpy"


def test_2d_input_block_processed_per_column():
    """Demeaning a 2-column matrix matches per-column results."""
    df = _make_panel(seed=5)
    fe = df[["i", "t"]].to_numpy()
    X = df[["x", "y"]].to_numpy()

    Xd, info = sp.fast.demean(X, fe, drop_singletons=False)
    yd, _ = sp.fast.demean(df["y"].to_numpy(), fe, drop_singletons=False)
    xd, _ = sp.fast.demean(df["x"].to_numpy(), fe, drop_singletons=False)

    assert np.allclose(Xd[:, 0], xd, atol=1e-12)
    assert np.allclose(Xd[:, 1], yd, atol=1e-12)
    assert len(info.iters) == 2
    assert all(info.converged)


# ---------------------------------------------------------------------------
# Singleton handling
# ---------------------------------------------------------------------------

def test_singleton_drop_default_on():
    """A row whose FE level appears once is dropped by default."""
    rng = np.random.default_rng(6)
    n = 200
    i = rng.integers(0, 30, size=n).astype(np.int64)
    # Force a unique FE level on the last row
    i[-1] = 999
    y = rng.normal(size=n)
    _, info = sp.fast.demean(y, [i], drop_singletons=True)
    assert info.n_dropped >= 1
    assert info.keep_mask[-1] is np.False_  # the singleton row was dropped
    assert info.n_kept == n - info.n_dropped


def test_singleton_drop_off_keeps_all_rows():
    rng = np.random.default_rng(7)
    n = 50
    i = np.arange(n).astype(np.int64)  # every row is its own singleton
    y = rng.normal(size=n)
    y_dem, info = sp.fast.demean(y, [i], drop_singletons=False)
    assert info.n_dropped == 0
    assert info.n_kept == n
    # With every row a singleton, demeaning yields exactly zero everywhere.
    assert np.allclose(y_dem, 0.0, atol=1e-12)


def test_cascading_singleton_drop():
    """Dropping a singleton in dim 1 may create one in dim 2; both removed."""
    i = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    j = np.array([0, 1, 0, 1, 99], dtype=np.int64)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _, info = sp.fast.demean(y, [i, j], drop_singletons=True)
    assert info.n_dropped == 1
    assert info.keep_mask[-1] == False  # noqa: E712


# ---------------------------------------------------------------------------
# DataFrame / Series / list inputs
# ---------------------------------------------------------------------------

def test_dataframe_fe_input():
    df = _make_panel(seed=8)
    y = df["y"].to_numpy()
    y1, _ = sp.fast.demean(y, df[["i", "t"]], drop_singletons=False)
    y2, _ = sp.fast.demean(y, df[["i", "t"]].to_numpy(), drop_singletons=False)
    y3, _ = sp.fast.demean(y, [df["i"].to_numpy(), df["t"].to_numpy()],
                           drop_singletons=False)
    assert np.allclose(y1, y2, atol=1e-12)
    assert np.allclose(y1, y3, atol=1e-12)


def test_string_fe_factorisation():
    """Non-numeric FE labels go through pd.factorize cleanly."""
    n = 100
    rng = np.random.default_rng(9)
    fe = rng.choice(["a", "b", "c", "d"], size=n)
    y = rng.normal(size=n)
    y_dem, info = sp.fast.demean(y, [fe], drop_singletons=False)
    assert info.converged[0]
    # Mean of demeaned y within each FE level should be ~0
    df = pd.DataFrame({"y": y_dem, "fe": fe})
    means = df.groupby("fe")["y"].mean()
    assert np.allclose(means.values, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_nan_in_X_raises():
    n = 50
    fe = np.zeros(n, dtype=np.int64)
    fe[: n // 2] = 1
    y = np.arange(n, dtype=float)
    y[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sp.fast.demean(y, [fe])


def test_nan_in_FE_raises():
    n = 50
    fe = np.array([1.0, np.nan] * (n // 2))
    y = np.arange(n, dtype=float)
    with pytest.raises(ValueError, match="NaN"):
        sp.fast.demean(y, [fe])


def test_unknown_accel_rejected():
    n = 20
    fe = np.zeros(n, dtype=np.int64)
    y = np.arange(n, dtype=float)
    with pytest.raises(ValueError, match="accel"):
        sp.fast.demean(y, [fe], accel="bogus")


def test_rust_backend_when_unavailable():
    """If the user forces backend='rust' on a no-Rust install, we must
    raise — never silently fall back without telling them."""
    # Reach the submodule explicitly (the package re-exports the function
    # under the same attribute name, so attribute lookup gives us the
    # function not the submodule — go through sys.modules instead).
    import sys
    demean_mod = sys.modules["statspai.fast.demean"]

    if demean_mod._HAS_RUST:
        pytest.skip("Rust extension is installed — cannot test missing-Rust path")
    with pytest.raises(RuntimeError, match="rust"):
        sp.fast.demean(np.zeros(10), [np.zeros(10, dtype=np.int64)], backend="rust")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_repeated_calls_deterministic():
    df = _make_panel(seed=10)
    y = df["y"].to_numpy()
    fe = df[["i", "t"]].to_numpy()
    a, _ = sp.fast.demean(y, fe, drop_singletons=False)
    b, _ = sp.fast.demean(y, fe, drop_singletons=False)
    assert np.array_equal(a, b)
