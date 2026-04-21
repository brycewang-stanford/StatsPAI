"""Smoke tests for v0.10 distributional IV / panel QTE frontier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def iv_data():
    rng = np.random.default_rng(0)
    n = 600
    Z = (rng.uniform(size=n) < 0.5).astype(int)
    D = ((rng.uniform(size=n) < 0.4 + 0.4 * Z)).astype(int)
    Y = 1.5 * D + 0.5 * Z + rng.standard_normal(n)
    return pd.DataFrame({'y': Y, 'treat': D, 'z': Z})


@pytest.fixture
def panel_qte_data():
    rng = np.random.default_rng(1)
    n_units, n_t = 30, 4
    rows = []
    for u in range(n_units):
        for t in range(n_t):
            d = int(u >= 15 and t >= 2)
            x = rng.standard_normal(5)
            y = u * 0.05 + t * 0.1 + 1.0 * d + 0.5 * x[0] + rng.standard_normal()
            rows.append({
                'y': y, 'treat': d, 'unit': f'u{u}', 'time': t,
                **{f'x{i}': x[i] for i in range(5)},
            })
    return pd.DataFrame(rows)


def test_dist_iv(iv_data):
    res = sp.dist_iv(
        iv_data, y='y', treat='treat', instrument='z',
        quantiles=np.array([0.25, 0.5, 0.75]), n_boot=50,
    )
    assert isinstance(res, sp.DistIVResult)
    assert len(res.late_q) == 3
    # True LATE on compliers ≈ 1.5
    median_late = res.late_q[1]
    assert 0.5 < median_late < 3.5


def test_kan_dlate(iv_data):
    res = sp.kan_dlate(
        iv_data, y='y', treat='treat', instrument='z',
        quantiles=np.array([0.5]), n_boot=30,
    )
    assert isinstance(res, sp.DistIVResult)


def test_qte_hd_panel(panel_qte_data):
    res = sp.qte_hd_panel(
        panel_qte_data, y='y', treat='treat', unit='unit', time='time',
        covariates=[f'x{i}' for i in range(5)],
        quantiles=np.array([0.25, 0.5, 0.75]),
    )
    assert isinstance(res, sp.HDPanelQTEResult)
    assert len(res.qte) == 3


def test_beyond_average(iv_data):
    res = sp.beyond_average_late(
        iv_data, y='y', treat='treat', instrument='z',
        quantiles=np.array([0.25, 0.5, 0.75]), n_boot=50,
    )
    assert isinstance(res, sp.BeyondAverageResult)
    assert 0 < res.complier_share < 1


def test_beyond_average_invalid_instrument():
    df = pd.DataFrame({
        'y': np.random.randn(50), 'treat': np.random.randint(0, 2, 50),
        'z': np.random.randn(50),  # continuous, should fail
    })
    with pytest.raises(ValueError, match="binary"):
        sp.beyond_average_late(df, y='y', treat='treat', instrument='z')
