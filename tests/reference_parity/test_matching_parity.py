"""Matching / weighting reference parity tests.

Under selection-on-observables with the correct covariate set, matching,
IPW, CBPS, entropy balancing, and overlap weights must all recover the
ATE/ATT within sampling error.

Validates:
1. Matching (nearest-neighbor) recovers the true ATT on a CIA DGP.
2. CBPS recovers ATE.
3. Entropy balancing recovers ATT.
4. Overlap weights recover ATO (but sign/magnitude close to ATE under
   homogeneous effects).
5. Cross-method agreement within combined SE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _within_n_se(est, truth, se, n_sigma=4.0):
    return abs(est - truth) <= n_sigma * se


class TestMatchingRecovery:
    """Matching on CIA DGP must recover ATT = 2.0."""

    COVARIATES = ['X1', 'X2', 'X3']

    def test_nearest_neighbor_matching_recovers_att(self, matching_cia_data):
        truth = matching_cia_data.attrs['true_effect']
        r = sp.match(matching_cia_data, y='y', treat='d',
                     covariates=self.COVARIATES,
                     method='nearest', estimand='ATT')
        assert _within_n_se(r.estimate, truth, r.se, n_sigma=4.0), (
            f"NN matching: {r.estimate:.4f} vs truth {truth} "
            f"(SE {r.se:.4f})"
        )

    def test_ebalance_recovers_att(self, matching_cia_data):
        truth = matching_cia_data.attrs['true_effect']
        r = sp.ebalance(matching_cia_data, y='y', treat='d',
                        covariates=self.COVARIATES)
        assert _within_n_se(r.estimate, truth, r.se, n_sigma=4.0), (
            f"Ebalance: {r.estimate:.4f} vs truth {truth} "
            f"(SE {r.se:.4f})"
        )

    def test_cbps_ate_recovers(self, matching_cia_data):
        truth = matching_cia_data.attrs['true_effect']  # same for ATT/ATE under homogeneity
        r = sp.cbps(matching_cia_data, y='y', treat='d',
                    covariates=self.COVARIATES, estimand='ATE',
                    n_bootstrap=100, seed=42)
        assert _within_n_se(r.estimate, truth, r.se, n_sigma=4.0), (
            f"CBPS ATE: {r.estimate:.4f} vs truth {truth} (SE {r.se:.4f})"
        )

    def test_overlap_weights_recovers_effect(self, matching_cia_data):
        """Under homogeneous effect, ATO == ATE == ATT = 2.0."""
        truth = matching_cia_data.attrs['true_effect']
        r = sp.overlap_weights(matching_cia_data, y='y', treat='d',
                               covariates=self.COVARIATES, estimand='ATO',
                               n_bootstrap=100, seed=42)
        assert _within_n_se(r.estimate, truth, r.se, n_sigma=4.0), (
            f"Overlap weights: {r.estimate:.4f} vs truth {truth} "
            f"(SE {r.se:.4f})"
        )


class TestMatchingCrossMethodAgreement:
    """Different SOO estimators must agree within combined SE on CIA DGP."""

    COVARIATES = ['X1', 'X2', 'X3']

    def test_matching_and_ebalance_agree(self, matching_cia_data):
        r_m = sp.match(matching_cia_data, y='y', treat='d',
                       covariates=self.COVARIATES, estimand='ATT')
        r_e = sp.ebalance(matching_cia_data, y='y', treat='d',
                          covariates=self.COVARIATES)
        combined_se = np.sqrt(r_m.se**2 + r_e.se**2)
        assert abs(r_m.estimate - r_e.estimate) <= 4.0 * combined_se, (
            f"match {r_m.estimate:.4f} vs ebalance {r_e.estimate:.4f} "
            f"(combined SE {combined_se:.4f})"
        )


# ---------------------------------------------------------------------------
# Sanity: matching on unconfounded RCT-like data must recover zero bias
# ---------------------------------------------------------------------------

def test_rct_like_matching_has_small_bias():
    """With random assignment, matching bias ≈ 0."""
    rng = np.random.default_rng(10)
    n = 2000
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    d = rng.binomial(1, 0.5, n)  # random assignment
    y = 1.0 + 0.5 * X1 + 0.3 * X2 + 1.5 * d + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({'y': y, 'd': d, 'X1': X1, 'X2': X2})

    r = sp.match(df, y='y', treat='d', covariates=['X1', 'X2'],
                 method='nearest', estimand='ATT')
    assert abs(r.estimate - 1.5) < 0.3, (
        f"RCT-like matching: {r.estimate:.4f} (expected ~1.5)"
    )
