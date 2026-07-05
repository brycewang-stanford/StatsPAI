"""Analytical parity: sp.mr_raps robust-profile-score MR recovery.

Zhao et al. 2020 (RAPS): profile-likelihood MR with Tukey biweight loss to
limit influence of gross pleiotropic outliers. Under a clean constant-effect
DGP the recovered estimate is close to beta and tau^2 (the inferred
heterogeneity variance) is near zero. Analytical evidence tier (known-truth
recovery; loose tolerance because the biweight loss targets heterogeneity, not
bias reduction).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA = 0.4


def test_raps_recovers_constant_effect_and_zero_tau2():
    rng = np.random.default_rng(0)
    K = 30
    bx = rng.uniform(0.1, 0.5, K)
    sx = np.full(K, 0.02)
    sy = np.full(K, 0.02)
    by = BETA * bx + rng.normal(0, sy)
    r = sp.mr_raps(bx, by, se_exposure=sx, se_outcome=sy)
    assert r.converged is True
    assert float(r.estimate) == pytest.approx(BETA, abs=0.1)
    # Under no pleiotropic outliers the inferred tau^2 is near zero.
    assert float(r.tau2) < 0.1


def test_raps_recovers_across_seeds():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        K = 30
        bx = rng.uniform(0.1, 0.5, K)
        sx = np.full(K, 0.02)
        sy = np.full(K, 0.02)
        by = BETA * bx + rng.normal(0, sy)
        r = sp.mr_raps(bx, by, se_exposure=sx, se_outcome=sy)
        assert float(r.estimate) == pytest.approx(BETA, abs=0.15)
        assert float(r.tau2) < 0.5
