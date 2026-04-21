"""Tests for Kernel IV with uniform inference (Lob et al. 2025)."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def _iv_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    v = rng.normal(size=n)
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    D = 0.8 * Z + v
    Y = 1.0 + 2.0 * D - 0.3 * D ** 2 + u
    return pd.DataFrame({"y": Y, "d": D, "z": Z})


def test_kernel_iv_returns_valid_result():
    df = _iv_data()
    r = sp.kernel_iv(df, y="y", treat="d", instrument="z", n_boot=30, seed=0)
    assert r.n_obs == len(df)
    assert r.bandwidth > 0
    assert r.grid.shape == r.h_hat.shape == r.ci_low.shape == r.ci_high.shape
    assert np.all(r.ci_low <= r.ci_high + 1e-10)


def test_kernel_iv_nontrivial_structural_shape():
    df = _iv_data(n=800, seed=1)
    r = sp.kernel_iv(df, y="y", treat="d", instrument="z", n_boot=40, seed=1)
    # Function should be monotonic-ish in the interior (h is quadratic,
    # so the slope around 0 should be positive).
    idx0 = np.argmin(np.abs(r.grid))
    idx1 = np.argmin(np.abs(r.grid - 1))
    # h(1) > h(0) — slope is positive in the bulk
    assert r.h_hat[idx1] > r.h_hat[idx0]


def test_kernel_iv_registry():
    fns = sp.list_functions()
    assert "kernel_iv" in fns


def test_kernel_iv_custom_grid_and_bw():
    df = _iv_data(seed=2)
    grid = np.linspace(-1.0, 1.0, 11)
    r = sp.kernel_iv(df, y="y", treat="d", instrument="z",
                      grid=grid, bandwidth=0.5, n_boot=20, seed=2)
    np.testing.assert_array_equal(r.grid, grid)
    assert abs(r.bandwidth - 0.5) < 1e-9


def test_kernel_iv_summary_is_string():
    df = _iv_data(n=200, seed=3)
    r = sp.kernel_iv(df, y="y", treat="d", instrument="z", n_boot=10, seed=3)
    s = r.summary()
    assert "Kernel IV" in s
    assert "N = 200" in s
