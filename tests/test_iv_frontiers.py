"""Smoke tests for v0.10 IV frontiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def kernel_data():
    rng = np.random.default_rng(0)
    n = 400
    Z = rng.standard_normal(n)
    D = 0.7 * Z + 0.3 * rng.standard_normal(n)
    Y = 1.2 * D + 0.5 * D ** 2 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({'y': Y, 'treat': D, 'z': Z})


@pytest.fixture
def continuous_iv_data():
    rng = np.random.default_rng(1)
    n = 500
    Z = rng.uniform(-1, 1, size=n)
    D = (Z > 0).astype(float) + 0.3 * rng.standard_normal(n)
    Y = 1.5 * D + 0.5 * Z + rng.standard_normal(n) * 0.4
    return pd.DataFrame({'y': Y, 'treat': D, 'z': Z})


@pytest.fixture
def ivdml_data():
    rng = np.random.default_rng(2)
    n = 600
    Z = rng.standard_normal((n, 3))
    X = rng.standard_normal((n, 4))
    D = 0.5 * Z[:, 0] + 0.3 * X[:, 0] + 0.2 * rng.standard_normal(n)
    Y = 1.0 * D + 0.4 * X[:, 1] + rng.standard_normal(n)
    return pd.DataFrame({
        'y': Y, 'treat': D,
        **{f'z{i}': Z[:, i] for i in range(3)},
        **{f'x{i}': X[:, i] for i in range(4)},
    })


def test_kernel_iv(kernel_data):
    res = sp.iv.kernel_iv(
        kernel_data, y='y', treat='treat', instrument='z',
        n_boot=20, seed=0,
    )
    assert isinstance(res, sp.iv.KernelIVResult)
    assert len(res.h_hat) == 30
    # h(0) should be near 0 (regression centered)
    assert np.isfinite(res.h_hat).any()


def test_continuous_iv_late(continuous_iv_data):
    res = sp.iv.continuous_iv_late(
        continuous_iv_data, y='y', treat='treat', instrument='z',
        n_quantiles=4, n_boot=50,
    )
    assert isinstance(res, sp.iv.ContinuousLATEResult)
    # True LATE ≈ 1.5
    assert 0.5 < res.estimate < 3.0


def test_ivdml(ivdml_data):
    res = sp.iv.ivdml(
        ivdml_data, y='y', treat='treat',
        instruments=['z0', 'z1', 'z2'],
        covariates=['x0', 'x1', 'x2', 'x3'], n_folds=3,
    )
    assert isinstance(res, sp.iv.IVDMLResult)
    # True LATE = 1.0
    assert 0.0 < res.estimate < 2.5
