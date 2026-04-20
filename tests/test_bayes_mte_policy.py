"""Tests for BayesianMTEResult.policy_effect + policy-weight builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import (
    bayes_mte,
    policy_weight_ate,
    policy_weight_subsidy,
    policy_weight_prte,
    policy_weight_marginal,
)

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mte_dgp(n, mte_fn, seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U = rng.normal(size=n)
    linpred = 0.8 * Z + U
    D = (linpred > 0).astype(float)
    u_obs = 1 / (1 + np.exp(-linpred))
    tau = np.array([mte_fn(u) for u in u_obs])
    Y = 1.0 + tau * D + 0.3 * rng.normal(size=n)
    return pd.DataFrame({'y': Y, 'd': D, 'z': Z})


@pytest.fixture
def decreasing_mte_data():
    """MTE(u) = 2 - 2*u — decreasing selection on gains."""
    return _mte_dgp(500, mte_fn=lambda u: 2.0 - 2.0 * u, seed=701)


@pytest.fixture
def mte_result(decreasing_mte_data):
    return bayes_mte(
        decreasing_mte_data, y='y', treat='d', instrument='z',
        poly_u=2, draws=400, tune=400, chains=2, progressbar=False,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Weight builders: input validation
# ---------------------------------------------------------------------------


def test_policy_weight_ate_is_uniform():
    w = policy_weight_ate()(np.linspace(0, 1, 11))
    np.testing.assert_allclose(w, np.ones(11))


@pytest.mark.parametrize("u_lo, u_hi", [
    (-0.1, 0.5),
    (0.0, 1.5),
    (0.7, 0.3),
    (0.5, 0.5),
])
def test_policy_weight_subsidy_rejects_invalid_bounds(u_lo, u_hi):
    with pytest.raises(ValueError, match='u_lo|u_hi'):
        policy_weight_subsidy(u_lo, u_hi)


def test_policy_weight_subsidy_band_mask():
    # Use a grid strictly interior to the band boundaries so floating-
    # point jitter on ``linspace`` boundary values cannot flip the
    # mask in edge cases (e.g. `0.2 >= 0.2` returning False due to
    # 0.2000000000000001).
    u = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    w = policy_weight_subsidy(0.2, 0.6)(u)
    # Hits u in [0.2, 0.6]: 0.25, 0.35, 0.45, 0.55
    np.testing.assert_array_equal(
        w,
        np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float),
    )


@pytest.mark.parametrize("shift", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_policy_weight_prte_rejects_bad_shift(shift):
    with pytest.raises(ValueError, match='shift'):
        policy_weight_prte(shift)


def test_policy_weight_prte_symmetric_around_half():
    # shift=0.4 -> band [0.3, 0.7]. Use interior-of-cell grid to
    # dodge floating-point boundary sensitivity on 0.3 / 0.7.
    u = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    w = policy_weight_prte(0.4)(u)
    # Hits u in [0.3, 0.7]: 0.35, 0.45, 0.55, 0.65
    np.testing.assert_array_equal(
        w,
        np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=float),
    )


def test_policy_weight_marginal_rejects_invalid_args():
    with pytest.raises(ValueError, match='u_star'):
        policy_weight_marginal(1.5)
    with pytest.raises(ValueError, match='bandwidth'):
        policy_weight_marginal(0.5, bandwidth=0.0)


def test_policy_weight_marginal_narrow_band_mask():
    # marginal at u=0.5 with bw=0.1 → band [0.4, 0.6]. Use interior
    # grid to avoid FP-boundary ambiguity on 0.4 / 0.6.
    u = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    w = policy_weight_marginal(0.5, bandwidth=0.1)(u)
    assert w.sum() == 2.0  # 0.45 and 0.55 are in band


# ---------------------------------------------------------------------------
# policy_effect: contract + parity with `.ate`
# ---------------------------------------------------------------------------


def test_policy_effect_ate_weight_matches_native_ate(mte_result):
    """policy_effect with uniform weights should be *numerically
    identical* to the internally-computed .ate. Both use
    trapezoidal integration on the same u_grid, so they must
    agree to floating-point precision."""
    pe = mte_result.policy_effect(policy_weight_ate(), label='ate')
    assert abs(pe['estimate'] - mte_result.ate) < 1e-8


def test_policy_effect_low_band_higher_than_high_band(mte_result):
    """Decreasing MTE DGP: subsidising low-u units should produce
    a higher policy effect than subsidising high-u units."""
    pe_low = mte_result.policy_effect(
        policy_weight_subsidy(0.05, 0.3), label='low',
    )
    pe_high = mte_result.policy_effect(
        policy_weight_subsidy(0.7, 0.95), label='high',
    )
    assert pe_low['estimate'] > pe_high['estimate']


def test_policy_effect_returns_expected_keys(mte_result):
    pe = mte_result.policy_effect(policy_weight_ate(), label='test')
    for k in ('label', 'estimate', 'std_error', 'hdi_low',
              'hdi_high', 'prob_positive'):
        assert k in pe
    assert pe['label'] == 'test'
    assert pe['hdi_low'] <= pe['estimate'] <= pe['hdi_high']


def test_policy_effect_rope(mte_result):
    pe = mte_result.policy_effect(
        policy_weight_ate(), rope=(-0.1, 0.1),
    )
    assert 'prob_rope' in pe
    assert 0.0 <= pe['prob_rope'] <= 1.0


def test_policy_effect_rejects_wrong_shape_weights(mte_result):
    def _bad(u):
        return np.ones(len(u) + 1)
    with pytest.raises(ValueError, match='shape'):
        mte_result.policy_effect(_bad)


def test_policy_effect_rejects_zero_weight(mte_result):
    def _zeros(u):
        return np.zeros_like(u)
    with pytest.raises(ValueError, match='zero'):
        mte_result.policy_effect(_zeros)


# ---------------------------------------------------------------------------
# Joint first-stage mode
# ---------------------------------------------------------------------------


def test_bayes_mte_joint_runs(decreasing_mte_data):
    r = bayes_mte(
        decreasing_mte_data, y='y', treat='d', instrument='z',
        first_stage='joint', poly_u=2,
        draws=300, tune=300, chains=2, progressbar=False,
        random_state=17,
    )
    assert r.model_info['first_stage'] == 'joint'
    assert 'joint' in r.method.lower()
    assert np.isfinite(r.ate)


def test_bayes_mte_joint_vs_plugin_similar_ate(decreasing_mte_data):
    r_plugin = bayes_mte(
        decreasing_mte_data, y='y', treat='d', instrument='z',
        first_stage='plugin', poly_u=2,
        draws=300, tune=300, chains=2, progressbar=False,
        random_state=18,
    )
    r_joint = bayes_mte(
        decreasing_mte_data, y='y', treat='d', instrument='z',
        first_stage='joint', poly_u=2,
        draws=300, tune=300, chains=2, progressbar=False,
        random_state=18,
    )
    # On a well-specified DGP the two posterior means should be
    # within a wide tolerance (plug-in and joint agree asymptotically)
    assert abs(r_plugin.ate - r_joint.ate) < 1.0


def test_bayes_mte_invalid_first_stage_raises(decreasing_mte_data):
    with pytest.raises(ValueError, match='first_stage'):
        bayes_mte(
            decreasing_mte_data, y='y', treat='d', instrument='z',
            first_stage='bogus',
            draws=50, tune=50, chains=1, progressbar=False,
        )


# ---------------------------------------------------------------------------
# Top-level exports
# ---------------------------------------------------------------------------


def test_policy_weight_helpers_at_top_level():
    assert sp.policy_weight_ate is policy_weight_ate
    assert sp.policy_weight_subsidy is policy_weight_subsidy
    assert sp.policy_weight_prte is policy_weight_prte
    assert sp.policy_weight_marginal is policy_weight_marginal
