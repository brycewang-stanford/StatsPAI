"""Tests for the Phase 7 JAX backend in ``sp.fast.demean``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

jax = pytest.importorskip("jax")


def _panel(seed=0, n_units=80, n_periods=15):
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    n = i.size
    x = rng.normal(size=n)
    a = rng.normal(0, 0.5, size=n_units)[i]
    g = rng.normal(0, 0.3, size=n_periods)[t]
    y = 1.0 + 0.3 * x + a + g + rng.normal(size=n)
    return pd.DataFrame({
        "y": y, "x": x,
        "i": i.astype(np.int32), "t": t.astype(np.int32),
    })


def test_jax_device_info_string():
    s = sp.fast.jax_device_info()
    assert "jax:" in s


def test_jax_demean_matches_rust():
    df = _panel(seed=1)
    y = df["y"].to_numpy()
    fe = df[["i", "t"]].to_numpy()

    y_rs, _ = sp.fast.demean(y, fe, backend="rust", drop_singletons=False,
                              tol=1e-12)
    y_jx, _ = sp.fast.demean(y, fe, backend="jax", drop_singletons=False,
                              tol=1e-10)
    # JAX uses jit + accumulator order may differ slightly from Rust;
    # 1e-9 is well below any practical use case.
    assert np.allclose(y_rs, y_jx, atol=1e-9)


def test_jax_demean_2d_input():
    df = _panel(seed=2)
    fe = df[["i", "t"]].to_numpy()
    X = df[["x", "y"]].to_numpy()
    Xd, info = sp.fast.demean(X, fe, backend="jax", drop_singletons=False)
    assert info.backend == "jax"
    assert Xd.shape == (len(df), 2)


def test_jax_oneway_exact():
    """K=1 → closed-form group demean."""
    rng = np.random.default_rng(3)
    n = 200
    g = rng.integers(0, 10, size=n).astype(np.int64)
    y = rng.normal(size=n)
    yd, info = sp.fast.demean(y, [g], backend="jax", drop_singletons=False)
    assert info.backend == "jax"
    expected = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    assert np.allclose(yd, expected, atol=1e-9)


def test_jax_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        sp.fast.demean(np.zeros(10), [np.zeros(10, dtype=np.int64)],
                        backend="cupy")


def test_jax_when_unavailable_raises():
    """If backend='jax' is forced and jax cannot be imported, we must raise."""
    import statspai.fast.jax_backend as jb_mod
    if jb_mod._HAS_JAX:
        pytest.skip("jax is installed — cannot test missing-jax path")
    with pytest.raises(RuntimeError, match="jax"):
        sp.fast.demean(np.zeros(10), [np.zeros(10, dtype=np.int64)],
                        backend="jax")
