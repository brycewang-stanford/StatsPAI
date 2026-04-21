"""
Tests for sp.synthesise_evidence / heterogeneity_of_effect / rwd_rct_concordance.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp


def test_synthesise_evidence_pools_correctly():
    # RCT: 0.5 ± 0.1, RWD: 0.6 ± 0.1 → pooled should be between.
    res = sp.synthesise_evidence(
        rct_estimate=0.5, rct_se=0.1,
        rwd_estimate=0.6, rwd_se=0.1,
    )
    assert 0.5 < res.pooled_estimate < 0.6
    # Inverse-variance with equal SEs → weights 0.5/0.5 → pooled = 0.55
    assert abs(res.pooled_estimate - 0.55) < 1e-10
    assert res.pooled_se < 0.1  # precision gain
    lo, hi = res.pooled_ci
    assert lo < res.pooled_estimate < hi
    assert abs(res.weights["rct"] - 0.5) < 1e-10


def test_synthesise_evidence_with_transport_shift():
    res = sp.synthesise_evidence(
        rct_estimate=0.5, rct_se=0.1,
        rwd_estimate=0.8, rwd_se=0.1,
        transport_shift=0.2, transport_shift_se=0.02,
    )
    # After shift, RCT-adj = 0.7 vs RWD 0.8 → pooled ≈ 0.75 (shifted toward RWD)
    assert 0.6 < res.pooled_estimate < 0.8


def test_synthesise_evidence_rct_heavy_weighting():
    res_iv = sp.synthesise_evidence(
        rct_estimate=0.5, rct_se=0.1,
        rwd_estimate=0.9, rwd_se=0.1,
        weight_mode="inverse_variance",
    )
    res_rh = sp.synthesise_evidence(
        rct_estimate=0.5, rct_se=0.1,
        rwd_estimate=0.9, rwd_se=0.1,
        weight_mode="rct_heavy",
    )
    # rct_heavy should pull pooled toward the RCT (0.5)
    assert res_rh.pooled_estimate < res_iv.pooled_estimate


def test_synthesise_evidence_rejects_nonpositive_se():
    with pytest.raises(ValueError, match="SEs must be > 0"):
        sp.synthesise_evidence(
            rct_estimate=0.1, rct_se=0.0, rwd_estimate=0.2, rwd_se=0.1,
        )


def test_heterogeneity_i2():
    # Two studies with very different estimates → high I²
    res = sp.heterogeneity_of_effect(
        estimates=[0.1, 1.0, 0.2, 0.9],
        ses=[0.05, 0.05, 0.05, 0.05],
    )
    assert res.i2 > 0.5
    assert res.q_stat > 0
    assert "Heterogeneity" in res.summary()


def test_heterogeneity_identical_studies():
    # Identical estimates → I² = 0
    res = sp.heterogeneity_of_effect(
        estimates=[0.3, 0.3, 0.3],
        ses=[0.1, 0.1, 0.1],
    )
    assert res.i2 == 0.0


def test_rwd_rct_concordance_inside():
    res = sp.rwd_rct_concordance(
        rct_estimate=0.5, rct_se=0.1, rwd_estimate=0.55,
    )
    assert res.rwd_inside_rct_ci is True
    assert abs(res.zscore_difference) < 1.0


def test_rwd_rct_concordance_outside():
    res = sp.rwd_rct_concordance(
        rct_estimate=0.5, rct_se=0.05, rwd_estimate=0.8,
    )
    assert res.rwd_inside_rct_ci is False
    assert res.zscore_difference > 1.96


def test_evidence_synthesis_in_registry():
    fns = set(sp.list_functions())
    assert "synthesise_evidence" in fns
    assert "heterogeneity_of_effect" in fns
    assert "rwd_rct_concordance" in fns
