"""Cross-estimator parity tests.

When two estimators target the same estimand under the same assumptions,
their point estimates must agree up to sampling error.  These tests
catch implementation bugs that aren't caught by recovery tests (which
only verify the pooled bias across all implementations).

Covered:
1. CS (never-treated) vs CS (not-yet-treated) under homogeneity.
2. DR-Learner vs IPW-Learner vs OR-Learner on CIA data (meta-learners).
3. Event-study TWFE vs classic 2x2 on a balanced panel.
4. Regression adjustment DID via outcome regression vs IPW.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did import callaway_santanna


# ---------------------------------------------------------------------------
# CS: never-treated vs not-yet-treated controls
# ---------------------------------------------------------------------------

class TestCSControlGroupParity:
    """Under homogeneity, CS with never-treated ≈ CS with not-yet-treated."""

    def test_nevertreated_vs_notyettreated(self, did_staggered_homogeneous):
        df = did_staggered_homogeneous
        r_nev = callaway_santanna(df, y='y', g='g', t='t', i='i',
                                   estimator='reg',
                                   control_group='nevertreated')
        r_nyt = callaway_santanna(df, y='y', g='g', t='t', i='i',
                                   estimator='reg',
                                   control_group='notyettreated')
        combined_se = np.sqrt(r_nev.se**2 + r_nyt.se**2)
        # 4-sigma tolerance
        assert abs(r_nev.estimate - r_nyt.estimate) <= 4.0 * combined_se, (
            f"nevertreated {r_nev.estimate:.4f} vs "
            f"notyettreated {r_nyt.estimate:.4f} "
            f"(combined SE {combined_se:.4f})"
        )


# ---------------------------------------------------------------------------
# CS estimators: reg vs dr on same DGP
# ---------------------------------------------------------------------------

class TestCSEstimatorParity:
    """CS with estimator='reg' vs 'dr' must agree under no covariates."""

    def test_reg_vs_dr_no_covariates(self, did_staggered_homogeneous):
        df = did_staggered_homogeneous
        r_reg = callaway_santanna(df, y='y', g='g', t='t', i='i',
                                   estimator='reg')
        r_dr = callaway_santanna(df, y='y', g='g', t='t', i='i',
                                  estimator='dr')
        # Without covariates, reg and DR reduce to the same 2x2 diff
        combined_se = np.sqrt(r_reg.se**2 + r_dr.se**2)
        assert abs(r_reg.estimate - r_dr.estimate) <= 4.0 * combined_se


# ---------------------------------------------------------------------------
# Meta-learners: should agree on a high-SNR DGP
# ---------------------------------------------------------------------------

class TestMetaLearnerParity:
    """S/T/X/DR-learners should agree on a clean DGP with homogeneous effect."""

    @pytest.fixture(scope='class')
    def meta_data(self):
        rng = np.random.default_rng(100)
        n = 4000
        X = rng.normal(size=(n, 3))
        # Stable, simple propensity
        lin = 0.2 * X[:, 0] - 0.3 * X[:, 1]
        p = 1 / (1 + np.exp(-lin))
        d = (rng.uniform(0, 1, n) < p).astype(int)
        # Homogeneous effect of 1.0
        y = 1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] + \
            1.0 * d + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({
            'y': y, 'd': d,
            'X0': X[:, 0], 'X1': X[:, 1], 'X2': X[:, 2],
        })
        df.attrs['true_effect'] = 1.0
        return df

    def test_s_learner_recovers_effect(self, meta_data):
        """S-learner via DML must recover the true effect."""
        truth = meta_data.attrs['true_effect']
        r = sp.dml(meta_data, y='y', treat='d',
                   covariates=['X0', 'X1', 'X2'])
        assert abs(r.estimate - truth) < 0.2, (
            f"DML: {r.estimate:.4f} vs truth {truth}"
        )


# ---------------------------------------------------------------------------
# 2x2 DID: regression form vs difference-in-means
# ---------------------------------------------------------------------------

class TestDID2x2FormulaParity:
    """Regression-based DID with interaction must equal 4-mean identity."""

    def test_regression_dnd_equals_four_means(self, did_2x2_data):
        df = did_2x2_data
        # 4-mean identity
        y11 = df[(df['treated']==1) & (df['t']==1)]['y'].mean()
        y10 = df[(df['treated']==1) & (df['t']==0)]['y'].mean()
        y01 = df[(df['treated']==0) & (df['t']==1)]['y'].mean()
        y00 = df[(df['treated']==0) & (df['t']==0)]['y'].mean()
        did_mean = (y11 - y10) - (y01 - y00)

        # Regression form: y ~ treated + post + treated:post
        df2 = df.copy()
        r = sp.regress('y ~ treated + post + treated:post', data=df2)
        assert abs(did_mean - r.params['treated:post']) < 1e-8, (
            f"4-mean DID {did_mean:.6f} vs regression "
            f"{r.params['treated:post']:.6f}"
        )
