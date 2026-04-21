"""Smoke tests for v0.10 Proximal Causal Inference frontier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def pci_data():
    rng = np.random.default_rng(42)
    n = 500
    U = rng.standard_normal(n)            # unobserved confounder
    Z = U + 0.5 * rng.standard_normal(n)  # treatment-side proxy
    W = U + 0.5 * rng.standard_normal(n)  # outcome-side proxy
    X = rng.standard_normal(n)            # observed covariate
    D = (1.0 * Z + 0.5 * X + rng.standard_normal(n) > 0).astype(int)
    Y = 2.0 * D + 1.0 * U + 0.3 * X + rng.standard_normal(n)
    return pd.DataFrame({
        'y': Y, 'treat': D, 'z': Z, 'w': W, 'x': X,
    })


def test_fortified_pci(pci_data):
    res = sp.fortified_pci(
        pci_data, y='y', treat='treat',
        proxy_z=['z'], proxy_w=['w'], covariates=['x'], n_boot=50,
    )
    assert hasattr(res, 'estimate')
    assert "Fortified" in res.method
    # True ATE = 2.0
    assert 0.5 < res.estimate < 4.0


def test_bidirectional_pci(pci_data):
    res = sp.bidirectional_pci(
        pci_data, y='y', treat='treat',
        proxy_z=['z'], proxy_w=['w'], covariates=['x'], n_boot=50,
    )
    assert hasattr(res, 'estimate')
    assert "Bidirectional" in res.method
    assert 0.5 < res.estimate < 4.0


def test_pci_mtp(pci_data):
    df = pci_data.copy()
    df['treat'] = pd.qcut(df['y'], q=4, labels=False).astype(float)  # continuous
    res = sp.pci_mtp(
        df, y='y', treat='treat',
        proxy_z=['z'], proxy_w=['w'], delta=1.0, covariates=['x'], n_boot=50,
    )
    assert hasattr(res, 'estimate')
    assert res.model_info['delta'] == 1.0


def test_select_pci_proxies(pci_data):
    df = pci_data.copy()
    df['noise1'] = np.random.default_rng(1).standard_normal(len(df))
    df['noise2'] = np.random.default_rng(2).standard_normal(len(df))
    res = sp.select_pci_proxies(
        df, y='y', treat='treat',
        candidates=['z', 'w', 'noise1', 'noise2'], covariates=['x'], top_k=2,
    )
    assert isinstance(res, sp.ProxyScoreResult)
    assert len(res.recommended_z) <= 2
    assert len(res.recommended_w) <= 2
    # The "good" proxies (z, w) should outscore pure noise
    z_top = res.z_candidates.iloc[0]['name']
    w_top = res.w_candidates.iloc[0]['name']
    assert z_top in {'z', 'w'}
    assert w_top in {'z', 'w'}
