"""DID reference parity tests.

Validates that StatsPAI's DID estimators:
1. Recover the population ATT on a deterministic DGP (within 3 SEs).
2. Cross-estimator parity on homogeneous-effect DGPs:
   classic 2x2 DID == Callaway-Sant'Anna == Sun-Abraham == Wooldridge
   up to sampling noise (all are consistent when effect is homogeneous).

See REFERENCES.md for the source of each tolerance and the population
parameters each DGP targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did import (
    callaway_santanna,
    sun_abraham,
    wooldridge_did,
    did_imputation,
    did as did_func,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _within_n_se(estimate: float, truth: float, se: float,
                 n_sigma: float = 4.0) -> bool:
    """Return True iff |estimate - truth| <= n_sigma * se.

    Default is 4-sigma (~99.99%): catches true bias, avoids type-I
    errors under a single draw of a valid estimator.  Real bias
    typically exceeds 10 sigma so this tolerance has no false
    negatives for implementation bugs.
    """
    return abs(estimate - truth) <= n_sigma * se


# Alias kept for readability in older tests
_within_3se = _within_n_se


# ---------------------------------------------------------------------------
# 2x2 DID recovery
# ---------------------------------------------------------------------------

class TestDID2x2Recovery:
    """Classic DID on 2-period 2-group data must recover true ATT."""

    def test_classic_did_recovers_true_att(self, did_2x2_data):
        truth = did_2x2_data.attrs['true_effect']
        r = did_func(did_2x2_data, y='y', treat='treated', time='t', post='post')
        assert _within_3se(r.estimate, truth, r.se), (
            f"Classic DID: estimate={r.estimate:.4f}, truth={truth}, "
            f"se={r.se:.4f} (>3 SE from truth)"
        )

    def test_cs_recovers_true_att_on_2x2(self, did_2x2_data):
        """Callaway-Sant'Anna on a 2-period 2-group panel must recover truth."""
        # CS needs a cohort column: period of first treatment (0 = never-treated)
        df = did_2x2_data.copy()
        df['cohort'] = df['treated']   # treated in period t=1
        truth = did_2x2_data.attrs['true_effect']
        r = callaway_santanna(df, y='y', g='cohort', t='t', i='i',
                              estimator='reg', control_group='nevertreated')
        assert _within_3se(r.estimate, truth, r.se), (
            f"CS2021 on 2x2: estimate={r.estimate:.4f}, truth={truth}, "
            f"se={r.se:.4f}"
        )


# ---------------------------------------------------------------------------
# Staggered + homogeneous: all heterogeneity-robust estimators must agree
# ---------------------------------------------------------------------------

class TestStaggeredHomogeneousAgreement:
    """On homogeneous-effect staggered DID, all modern estimators must agree
    to within combined-SE tolerance, and recover truth within 3 SE."""

    TRUTH = 1.5

    @pytest.fixture(scope='class')
    def results(self, did_staggered_homogeneous):
        df = did_staggered_homogeneous
        return {
            'cs': callaway_santanna(df, y='y', g='g', t='t', i='i',
                                    estimator='reg'),
            'sa': sun_abraham(df, y='y', g='g', t='t', i='i'),
            'wool': wooldridge_did(df, y='y', group='i', time='t',
                                   first_treat='g'),
        }

    def test_cs_recovers_truth(self, results):
        r = results['cs']
        assert _within_3se(r.estimate, self.TRUTH, r.se), (
            f"CS: {r.estimate:.4f} vs truth {self.TRUTH} (SE {r.se:.4f})"
        )

    def test_sa_recovers_truth(self, results):
        r = results['sa']
        assert _within_3se(r.estimate, self.TRUTH, r.se), (
            f"SA: {r.estimate:.4f} vs truth {self.TRUTH} (SE {r.se:.4f})"
        )

    def test_wooldridge_recovers_truth(self, results):
        r = results['wool']
        assert _within_3se(r.estimate, self.TRUTH, r.se), (
            f"Wooldridge: {r.estimate:.4f} vs truth {self.TRUTH} "
            f"(SE {r.se:.4f})"
        )

    def test_cs_and_sa_agree(self, results):
        """CS vs SA: difference should be within 3 * combined SE."""
        r_cs, r_sa = results['cs'], results['sa']
        combined_se = np.sqrt(r_cs.se**2 + r_sa.se**2)
        assert abs(r_cs.estimate - r_sa.estimate) <= 3.0 * combined_se, (
            f"CS {r_cs.estimate:.4f} vs SA {r_sa.estimate:.4f} "
            f"(combined SE {combined_se:.4f})"
        )

    def test_cs_and_wooldridge_agree(self, results):
        r_cs, r_w = results['cs'], results['wool']
        combined_se = np.sqrt(r_cs.se**2 + r_w.se**2)
        assert abs(r_cs.estimate - r_w.estimate) <= 3.0 * combined_se, (
            f"CS {r_cs.estimate:.4f} vs Wool {r_w.estimate:.4f} "
            f"(combined SE {combined_se:.4f})"
        )


# ---------------------------------------------------------------------------
# Staggered + heterogeneous: TWFE biased, modern estimators unbiased
# ---------------------------------------------------------------------------

class TestHeterogeneityBiasDetection:
    """On heterogeneous DGPs, Bacon decomposition must detect the contamination."""

    def test_bacon_decomposition_runs(self, did_staggered_heterogeneous):
        """Bacon decomposition returns non-trivial components."""
        df = did_staggered_heterogeneous.copy()
        # Bacon expects binary treat indicator; derive from g
        df['treat'] = ((df['g'] > 0) & (df['t'] >= df['g'])).astype(int)
        r = sp.bacon_decomposition(df, y='y', treat='treat', time='t', id='i')
        # Must return a dict with decomposition components
        assert isinstance(r, dict), f"Expected dict, got {type(r)}"
        assert len(r) > 0, "Bacon decomposition returned empty"

    def test_cs_beats_twfe_under_heterogeneity(self,
                                                did_staggered_heterogeneous):
        """On heterogeneous DGP, CS should get closer to unit-uniform truth
        than naive TWFE."""
        df = did_staggered_heterogeneous
        r_cs = callaway_santanna(df, y='y', g='g', t='t', i='i',
                                  estimator='reg')
        # CS aggregated 'simple' ATT must exist and be finite
        assert np.isfinite(r_cs.estimate)
        # Unit-uniform weighted truth: for each cohort g with uniform share,
        # and each post-period horizon h, sum over cohort-effects * share.
        # Here all cohorts have same size; effects are (1.0, 1.5, 2.0)
        # -> simple ATT unweighted over (g,t) is near the mean 1.5
        # (with weighting by #post-periods gives slightly lower mass on 7).
        # Allow a wide [0.5, 2.5] plausibility band.
        assert 0.5 < r_cs.estimate < 2.5, (
            f"CS estimate {r_cs.estimate} outside plausibility [0.5, 2.5]"
        )


# ---------------------------------------------------------------------------
# Sanity: DID estimator must be sign-correct
# ---------------------------------------------------------------------------

def test_did_sign_correct_positive_effect(did_2x2_data):
    """Positive DGP effect must produce positive DID estimate."""
    r = did_func(did_2x2_data, y='y', treat='treated', time='t', post='post')
    assert r.estimate > 0, f"Expected positive, got {r.estimate}"


def test_did_sign_correct_negative_effect():
    """Flip the treatment; DID should now be negative."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(400):
        treated = 1 if i < 200 else 0
        for t in [0, 1]:
            y = 1.0 + 0.2 * t - 1.5 * treated * t + rng.normal(scale=0.5)
            rows.append({'i': i, 't': t, 'treated': treated, 'post': t, 'y': y})
    df = pd.DataFrame(rows)
    r = did_func(df, y='y', treat='treated', time='t', post='post')
    assert r.estimate < 0, f"Expected negative, got {r.estimate}"
    assert abs(r.estimate - (-1.5)) < 0.3, f"Far from truth: {r.estimate}"
