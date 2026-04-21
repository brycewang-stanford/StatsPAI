"""Smoke tests for general bunching + unified Kink-RDD-Bunching."""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp


def test_general_bunching():
    rng = np.random.default_rng(0)
    n = 1500
    R = rng.normal(0, 0.5, size=n)
    # Add a small bunch at 0
    R = np.concatenate([R, rng.normal(0, 0.05, size=200)])
    df = pd.DataFrame({'r': R})
    res = sp.general_bunching(
        df, running='r', cutoff=0.0, bandwidth=0.5,
        polynomial_order=4, n_boot=30,
    )
    assert isinstance(res, sp.GeneralBunchingResult)
    # Bunching elasticity should be positive (mass piled at cutoff)
    assert np.isfinite(res.bias_corrected_elasticity)


def test_kink_unified():
    rng = np.random.default_rng(1)
    n = 800
    R = rng.uniform(-1, 1, size=n)
    Y = np.where(R > 0, 1.5 * R, 0.5 * R) + 0.2 * rng.standard_normal(n)
    df = pd.DataFrame({'y': Y, 'r': R})
    res = sp.kink_unified(
        df, y='y', running='r', cutoff=0.0, bandwidth=0.6,
    )
    assert isinstance(res, sp.KinkUnifiedResult)
    # RKD should pick up the slope difference (≈ 1.0)
    assert 0.3 < abs(res.rkd_effect) < 2.0
