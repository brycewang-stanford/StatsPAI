"""Dependency-light policy-weight builder contracts."""

from __future__ import annotations

import numpy as np

import statspai as sp


def test_policy_weight_builders_return_expected_masks_and_shapes():
    u = np.array([0.05, 0.25, 0.35, 0.45, 0.55, 0.65, 0.85])

    subsidy = sp.policy_weight_subsidy(0.2, 0.6)(u)
    prte = sp.policy_weight_prte(0.4)(u)
    marginal = sp.policy_weight_marginal(0.5, bandwidth=0.1)(u)

    np.testing.assert_allclose(subsidy, [0, 1, 1, 1, 1, 0, 0])
    np.testing.assert_allclose(prte, [0, 0, 1, 1, 1, 1, 0])
    np.testing.assert_allclose(marginal, [0, 0, 0, 1, 1, 0, 0])

    propensity = np.linspace(0.05, 0.95, 20)
    observed = sp.policy_weight_observed_prte(propensity, shift=0.2)(u)
    np.testing.assert_allclose(len(observed), len(u))
    assert np.isfinite(observed).all()
