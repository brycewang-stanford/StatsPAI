"""Coverage campaign — weak-identification tests, weak-IV confidence sets,
JIVE variants, and nonparametric IV.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Targets ``iv/weak_identification.py``
(Kleibergen–Paap rk, Sanderson–Windmeijer, conditional-LR test),
``iv/weak_iv_ci.py`` (Anderson–Rubin / CLR / K confidence sets),
``iv/jive_variants.py`` (JIVE1/UJIVE/IJIVE/RJIVE), and ``iv/npiv.py``.

Assertions are real: with strong instruments the weak-ID F-stats are large, the
weak-robust confidence sets are non-empty and bracket the true coefficient, and
the (consistent) JIVE variants recover the DGP slope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def strong_iv():
    """Strong, over-identified IV, single endogenous regressor, true beta=2."""
    rng = np.random.default_rng(7)
    n = 600
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    z3 = rng.standard_normal(n)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = (
        0.9 * z1
        + 0.7 * z2
        + 0.5 * z3
        + 0.3 * x
        + 0.6 * u
        + 0.5 * rng.standard_normal(n)
    )
    y = 1.0 + 2.0 * d + 0.4 * x + u + rng.standard_normal(n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "z3": z3, "x": x})


# ─── weak_identification.py ──────────────────────────────────────────────


def test_kleibergen_paap_rk(strong_iv):
    df = strong_iv
    res = sp.iv.kleibergen_paap_rk(
        endog=df[["d"]], instruments=df[["z1", "z2", "z3"]], exog=df[["x"]]
    )
    assert np.isfinite(res.rk_f) and res.rk_f > 0
    assert np.isfinite(res.rk_lm)
    assert 0.0 <= res.rk_lm_pvalue <= 1.0
    # strong instruments → comfortably above the rule-of-thumb 10
    assert res.rk_f > 10.0


def test_sanderson_windmeijer(strong_iv):
    df = strong_iv
    res = sp.iv.sanderson_windmeijer(
        endog=df[["d"]], instruments=df[["z1", "z2", "z3"]], exog=df[["x"]]
    )
    assert res is not None
    # SW conditional F is a per-endogenous-regressor mapping, e.g. {'d': 491.0}.
    sw_f = res.sw_f
    assert isinstance(sw_f, dict) and "d" in sw_f
    f_d = float(sw_f["d"])
    # strong instruments → SW conditional F well above the rule-of-thumb 10
    assert np.isfinite(f_d) and f_d > 10.0
    assert 0.0 <= float(res.sw_pvalue["d"]) <= 1.0


def test_conditional_lr_test(strong_iv):
    df = strong_iv
    res = sp.iv.conditional_lr_test(
        y="y", endog="d", instruments=["z1", "z2", "z3"], exog=["x"], data=df
    )
    assert res is not None
    pval = getattr(res, "pvalue", getattr(res, "p_value", None))
    if pval is not None:
        assert 0.0 <= float(pval) <= 1.0


# ─── weak_iv_ci.py — weak-robust confidence sets ─────────────────────────


@pytest.mark.parametrize(
    "fn_name", ["anderson_rubin_ci", "conditional_lr_ci", "k_test_ci"]
)
def test_weak_iv_confidence_sets_bracket_truth(strong_iv, fn_name):
    df = strong_iv
    fn = getattr(sp.iv, fn_name)
    cs = fn(y="y", endog="d", instruments=["z1", "z2", "z3"], exog=["x"], data=df)
    assert not cs.is_empty
    intervals = cs.as_intervals()
    assert len(intervals) >= 1
    # with strong instruments the set is a connected interval containing beta=2
    assert any(lo - 1e-6 <= 2.0 <= hi + 1e-6 for lo, hi in intervals)


def test_anderson_rubin_ci_is_connected_when_strong(strong_iv):
    df = strong_iv
    cs = sp.iv.anderson_rubin_ci(
        y="y", endog="d", instruments=["z1", "z2", "z3"], exog=["x"], data=df
    )
    assert cs.is_connected
    assert not cs.is_unbounded


# ─── jive_variants.py ────────────────────────────────────────────────────


@pytest.mark.parametrize("variant", ["jive1", "ujive", "ijive", "rjive"])
def test_jive_variants_recover_slope(strong_iv, variant):
    df = strong_iv
    fn = getattr(sp.iv, variant)
    res = fn(y="y", endog="d", instruments=["z1", "z2", "z3"], exog=["x"], data=df)
    beta = float(res.params["endog0"])
    assert np.isfinite(beta)
    # JIVE family is consistent; with strong instruments it lands near 2.
    assert abs(beta - 2.0) < 0.7
    assert res.first_stage_f > 0


def test_rjive_ridge_parameter(strong_iv):
    df = strong_iv
    res = sp.iv.rjive(
        y="y", endog="d", instruments=["z1", "z2", "z3"], exog=["x"], data=df, ridge=5.0
    )
    assert np.isfinite(float(res.params["endog0"]))


# ─── npiv.py ─────────────────────────────────────────────────────────────


def test_npiv_structural_function(strong_iv):
    df = strong_iv
    res = sp.iv.npiv(y="y", endog="d", instruments=df[["z1", "z2", "z3"]], data=df)
    assert res is not None
    h = np.asarray(res.h_values, dtype=float)
    assert h.size > 0 and np.all(np.isfinite(h))
    assert np.asarray(res.d_grid).size == h.size
    # tidy frame export path
    frame = res.to_frame()
    assert isinstance(frame, pd.DataFrame) and len(frame) > 0
