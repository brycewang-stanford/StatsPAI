"""Analytical parity: sp.mr_mode (mode-based MR) known-truth recovery.

Hartwig-Davey Smith-Bowden (2017) mode-based estimator. Under a constant
treatment effect and no heterogeneity, the modal per-SNP IVW estimate
recovers the planted effect. Analytical evidence tier (known-truth
recovery on a deterministic DGP; the mode is the empirical mode of the
per-SNP IVW distribution so "exact" equality is a tolerance comparison).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA = 0.4


def test_recovers_constant_effect_under_no_heterogeneity():
    rng = np.random.default_rng(2)
    K = 50
    bx = rng.uniform(0.1, 0.5, K)
    sx = np.full(K, 0.02)
    sy = np.full(K, 0.02)
    by = BETA * bx + rng.normal(0, sy)
    r = sp.mr_mode(
        beta_exposure=bx,
        beta_outcome=by,
        se_exposure=sx,
        se_outcome=sy,
        n_boot=200,
        seed=0,
    )
    assert float(r.estimate) == pytest.approx(BETA, abs=0.1)


def test_recovers_across_seeds():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        K = 50
        bx = rng.uniform(0.1, 0.5, K)
        sx = np.full(K, 0.02)
        sy = np.full(K, 0.02)
        by = BETA * bx + rng.normal(0, sy)
        r = sp.mr_mode(
            beta_exposure=bx,
            beta_outcome=by,
            se_exposure=sx,
            se_outcome=sy,
            n_boot=200,
            seed=0,
        )
        assert float(r.estimate) == pytest.approx(BETA, abs=0.1)
