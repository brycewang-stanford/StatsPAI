"""Tests for IIVM (Interactive IV Model) — binary D, binary Z."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


@pytest.fixture
def iivm_dgp():
    """
    Binary Z (lottery), binary D (compliance), heterogeneous LATE.

        Z ~ Bernoulli(0.5)  — random assignment
        D = Z * complier + always_taker  (one-sided noncompliance possible)
        Y = 1.5 * D + 0.5 * X + U + eps  where U is unobserved

    LATE ≈ 1.5 on compliers.
    """
    rng = np.random.default_rng(42)
    n = 3000
    X = rng.normal(0, 1, n)
    U = rng.normal(0, 0.5, n)
    # Instrument is random
    Z = rng.binomial(1, 0.5, n).astype(float)
    # Types: complier (D = Z), always-taker (D = 1), never-taker (D = 0)
    u = rng.uniform(0, 1, n)
    D = np.where(
        u < 0.6, Z,  # 60% compliers
        np.where(u < 0.8, 1.0, 0.0),  # 20% always, 20% never
    )
    Y = 1.5 * D + 0.5 * X + U + rng.normal(0, 0.3, n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': X})


def test_iivm_recovers_late(iivm_dgp):
    """IIVM should recover LATE ≈ 1.5 (tight; n=3000, 60% compliers)."""
    result = sp.dml(
        iivm_dgp, y='y', treat='d', covariates=['x'],
        model='iivm', instrument='z',
    )
    assert isinstance(result, CausalResult)
    assert result.estimand == 'LATE'
    # Tightened from 0.3 → 0.12 after post-review pass. With n=3000
    # and π_C≈0.6, the IF-based SE is on the order of 0.03, so 0.12
    # is a ~4σ envelope — catches bias but not ordinary MC noise.
    assert abs(result.estimate - 1.5) < 0.12, (
        f"IIVM estimate {result.estimate:.3f}, expected ≈ 1.5"
    )


def test_iivm_se_in_reasonable_range(iivm_dgp):
    """
    Guardrail: the reported SE should be roughly in line with what an
    IF-based calculation predicts for this DGP (σ_ψ / (√n · π_C)).
    A working-response / aggregation bug or a bad fallback on subgroup
    fits would blow this up. Keep a loose band for MC/learner noise.
    """
    result = sp.dml(
        iivm_dgp, y='y', treat='d', covariates=['x'],
        model='iivm', instrument='z',
    )
    assert 0.005 < result.se < 0.15, (
        f"IIVM SE {result.se:.4f} outside expected band; "
        f"suggests aggregation / clipping / fallback regression."
    )


def test_iivm_significance(iivm_dgp):
    result = sp.dml(
        iivm_dgp, y='y', treat='d', covariates=['x'],
        model='iivm', instrument='z',
    )
    assert result.pvalue < 0.05


def test_iivm_rejects_continuous_z():
    rng = np.random.default_rng(0)
    n = 500
    Z = rng.normal(0, 1, n)  # continuous
    D = rng.binomial(1, 1 / (1 + np.exp(-Z)), n).astype(float)
    Y = D + rng.normal(0, 0.3, n)
    X = rng.normal(0, 1, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': X})
    with pytest.raises(ValueError, match='binary'):
        sp.dml(df, y='y', treat='d', covariates=['x'],
               model='iivm', instrument='z')


def test_iivm_rejects_continuous_d():
    rng = np.random.default_rng(0)
    n = 500
    Z = rng.binomial(1, 0.5, n).astype(float)
    D = rng.normal(0, 1, n)  # continuous treatment
    Y = D + rng.normal(0, 0.3, n)
    X = rng.normal(0, 1, n)
    df = pd.DataFrame({'y': Y, 'd': D, 'z': Z, 'x': X})
    with pytest.raises(ValueError, match='binary'):
        sp.dml(df, y='y', treat='d', covariates=['x'],
               model='iivm', instrument='z')


def test_iivm_model_info(iivm_dgp):
    result = sp.dml(
        iivm_dgp, y='y', treat='d', covariates=['x'],
        model='iivm', instrument='z',
    )
    assert result.model_info['dml_model'] == 'IIVM'
    assert result.model_info['instrument'] == 'z'
    assert 'ml_r' in result.model_info
