"""Analytical parity: sp.grapple robust-adjusted profile-score MR recovery.

GRAPPLE (wang2021causal): robust profile likelihood MR with the Tukey
biweight loss. Under a clean DGP (no pleiotropic outliers) it recovers the
planted constant effect analytically. Analytical evidence tier (known-truth
recovery on a deterministic DGP; the test is loose because the biweight
loss targets heterogeneity, not bias reduction).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA = 0.4


def test_grapple_recovers_constant_effect_under_no_heterogeneity():
    rng = np.random.default_rng(0)
    K = 30
    bx = rng.uniform(0.1, 0.5, K)
    sx = np.full(K, 0.02)
    sy = np.full(K, 0.02)
    by = BETA * bx + rng.normal(0, sy)
    r = sp.grapple(bx, by, se_exposure=sx, se_outcome=sy)
    assert float(r.estimate) == pytest.approx(BETA, abs=0.15)
    assert r.converged is True


def test_grapple_recovers_across_seeds():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        K = 30
        bx = rng.uniform(0.1, 0.5, K)
        sx = np.full(K, 0.02)
        sy = np.full(K, 0.02)
        by = BETA * bx + rng.normal(0, sy)
        r = sp.grapple(bx, by, se_exposure=sx, se_outcome=sy)
        assert float(r.estimate) == pytest.approx(BETA, abs=0.15)
