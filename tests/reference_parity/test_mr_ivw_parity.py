"""Analytical parity: sp.mr("ivw") exact inverse-variance-weighted identity.

Two-sample IVW Mendelian randomization is weighted regression of the
instrument-outcome on instrument-exposure associations through the origin with
weights 1/se_outcome^2. Exact closed forms:

    estimate == sum(w b_x b_y) / sum(w b_x^2)
    se       == sqrt(1 / sum(w b_x^2))
    Q        == sum(w (b_y - est b_x)^2),   Q_df == K - 1

All match a hand computation to machine precision (observed diff 0), and the
estimate recovers the planted causal effect beta = 0.3.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA = 0.3
K = 40


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    bx = rng.uniform(0.1, 0.5, K)
    sx = np.full(K, 0.02)
    sy = np.full(K, 0.02)
    by = BETA * bx + rng.normal(0, sy)
    r = sp.mr("ivw", beta_exposure=bx, beta_outcome=by, se_exposure=sx, se_outcome=sy)
    return bx, by, sy, r


def test_ivw_matches_weighted_origin_regression(fitted):
    """The point estimate, and both standard-error models in closed form.

    Since 1.28.0 the default follows MendelianRandomization::mr_ivw: random
    effects (fixed-effect SE times max(1, RSE)) with more than three
    variants. The fixed-effect SE is still available as model="fixed".
    """
    bx, by, sy, r = fitted
    w = 1 / sy**2
    beta = np.sum(w * bx * by) / np.sum(w * bx**2)
    se_fixed = np.sqrt(1 / np.sum(w * bx**2))
    rse = np.sqrt(np.sum(w * (by - beta * bx) ** 2) / (len(bx) - 1))
    assert r["estimate"] == pytest.approx(beta, abs=1e-12)
    assert r["model"] == "random"
    assert r["se"] == pytest.approx(se_fixed * max(1.0, rse), abs=1e-12)
    fixed = sp.mr_ivw(bx, by, np.zeros_like(bx), sy, model="fixed")
    assert fixed["se"] == pytest.approx(se_fixed, abs=1e-12)


def test_cochran_q_identity(fitted):
    bx, by, sy, r = fitted
    w = 1 / sy**2
    resid = by - r["estimate"] * bx
    assert r["Q"] == pytest.approx(float(np.sum(w * resid**2)), abs=1e-10)
    assert int(r["Q_df"]) == K - 1


def test_recovers_planted_effect(fitted):
    *_, r = fitted
    assert r["estimate"] == pytest.approx(BETA, abs=0.05)
