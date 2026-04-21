"""
Tests for the MVMR / Mediation MR / MR-BMA extensions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_mvmr_data(*, n_snps: int = 80, seed: int = 61):
    """Two-exposure MVMR DGP.

    alpha_true = [0.5, 0.2]: X1 has direct effect 0.5, X2 has 0.2.
    """
    rng = np.random.default_rng(seed)
    beta_x1 = rng.normal(0, 1, size=n_snps)
    beta_x2 = 0.3 * beta_x1 + rng.normal(0, 1, size=n_snps)  # correlated
    se_y = np.abs(rng.normal(0.1, 0.02, size=n_snps)) + 0.05
    beta_y = 0.5 * beta_x1 + 0.2 * beta_x2 + rng.normal(0, se_y)
    return pd.DataFrame({
        "beta_x1": beta_x1, "beta_x2": beta_x2,
        "beta_y": beta_y, "se_y": se_y,
    })


def test_mr_multivariable_recovers_direct_effects():
    df = _make_mvmr_data(n_snps=150, seed=61)
    res = sp.mr_multivariable(df, exposures=["beta_x1", "beta_x2"])
    de = res.direct_effect.set_index("exposure")
    assert abs(de.loc["beta_x1", "estimate"] - 0.5) < 0.1, de.loc["beta_x1"]
    assert abs(de.loc["beta_x2", "estimate"] - 0.2) < 0.1, de.loc["beta_x2"]
    # F-stat should be non-trivial (weak-instrument threshold varies by
    # correlation; we just require it to be finite and positive).
    for exp in ["beta_x1", "beta_x2"]:
        assert res.conditional_f_stats[exp] > 1.0, res.conditional_f_stats
    # Summary renders
    assert "Multivariable" in res.summary()


def test_mr_multivariable_single_exposure_errors():
    df = pd.DataFrame({"beta_x1": [1, 2, 3], "beta_y": [1, 2, 3], "se_y": [0.1, 0.1, 0.1]})
    with pytest.raises(ValueError, match=">= 2 exposures"):
        sp.mr_multivariable(df, exposures=["beta_x1"])


def test_mr_mediation_two_step():
    """DGP: X -> M -> Y + direct X -> Y.

    For two-step MR to work, the SNPs must instrument X; M comes from X.
    """
    rng = np.random.default_rng(67)
    n = 200
    beta_x = rng.normal(0, 1, size=n)
    # Mediator mediates part of the total effect of X
    beta_m = 0.4 * beta_x + rng.normal(0, 0.3, size=n)
    # Y receives a direct effect of 0.5 from X + 0.3 through M → total ≈ 0.5 + 0.4*0.3 = 0.62.
    se_y = np.abs(rng.normal(0.1, 0.02, size=n)) + 0.05
    beta_y = 0.5 * beta_x + 0.3 * beta_m + rng.normal(0, se_y)
    df = pd.DataFrame({
        "beta_x": beta_x, "se_x": np.full(n, 0.01),
        "beta_m": beta_m, "se_m": np.full(n, 0.01),
        "beta_y": beta_y, "se_y": se_y,
    })
    res = sp.mr_mediation(df)
    # total ≈ 0.62, direct ≈ 0.5, indirect ≈ 0.12
    assert abs(res.total_effect - 0.62) < 0.15, res.total_effect
    assert abs(res.direct_effect - 0.5) < 0.15, res.direct_effect
    assert abs(res.indirect_effect - 0.12) < 0.15, res.indirect_effect
    assert 0.0 < res.proportion_mediated < 0.5, res.proportion_mediated
    assert "Two-Step MR" in res.summary()


def test_mr_bma_identifies_causal_exposure():
    """MR-BMA should give high posterior mass to the model containing only x1
    when x2 is null."""
    rng = np.random.default_rng(71)
    n = 120
    beta_x1 = rng.normal(0, 1, size=n)
    beta_x2 = rng.normal(0, 1, size=n)
    beta_x3 = rng.normal(0, 1, size=n)
    se_y = np.abs(rng.normal(0.1, 0.02, size=n)) + 0.05
    # Only x1 is causal
    beta_y = 0.5 * beta_x1 + rng.normal(0, se_y)
    df = pd.DataFrame({
        "beta_x1": beta_x1, "beta_x2": beta_x2, "beta_x3": beta_x3,
        "beta_y": beta_y, "se_y": se_y,
    })
    res = sp.mr_bma(df, exposures=["beta_x1", "beta_x2", "beta_x3"])
    # x1 should have the highest marginal inclusion
    mp = res.marginal_inclusion
    assert mp["beta_x1"] > mp["beta_x2"], mp
    assert mp["beta_x1"] > mp["beta_x3"], mp
    assert mp["beta_x1"] > 0.5, mp
    # Top model should contain x1
    top = res.best_models.iloc[0]
    assert "beta_x1" in top["model"], top


def test_mr_bma_too_few_exposures_errors():
    df = pd.DataFrame({
        "beta_x1": np.random.randn(20),
        "beta_y": np.random.randn(20),
        "se_y": np.full(20, 0.1),
    })
    with pytest.raises(ValueError, match=">= 2 exposures"):
        sp.mr_bma(df, exposures=["beta_x1"])


def test_mr_extensions_in_registry():
    fns = set(sp.list_functions())
    assert "mr_multivariable" in fns
    assert "mr_mediation" in fns
    assert "mr_bma" in fns
