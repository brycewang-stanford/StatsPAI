"""Tests for the design-stage power calculators (previously untested):
``sp.power_did``, ``sp.power_iv``, ``sp.power_rd``, ``sp.power_cluster_rct``.

These are closed-form (deterministic), so the assertions are the qualitative
laws every power function must obey: power rises with sample size and effect
size, is bounded in [0, 1], collapses to the nominal level under a null
effect, and — for clustered designs — falls as the intra-cluster correlation
grows.
"""

import pytest
from scipy.stats import norm

import statspai as sp


def _did(n, effect=0.3):
    return sp.power_did(n=n, effect_size=effect, n_periods=4, n_treated_periods=2).power


def _rd(n, effect=0.3):
    return sp.power_rd(n=n, effect_size=effect, bandwidth=0.5).power


def _iv(n, effect=0.3, f=20):
    return sp.power_iv(n=n, effect_size=effect, first_stage_f=f).power


def _cluster(g, effect=0.3, icc=0.05):
    return sp.power_cluster_rct(
        n_clusters=g, cluster_size=20, effect_size=effect, icc=icc
    ).power


# --------------------------------------------------------------------------
# Bounds + contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("power", [_did(200), _rd(1000), _iv(500), _cluster(40)])
def test_power_is_a_probability(power):
    assert 0.0 <= power <= 1.0


def test_result_objects_carry_design_tag():
    assert (
        sp.power_did(n=200, effect_size=0.3, n_periods=4, n_treated_periods=2).design
        == "did"
    )
    assert sp.power_rd(n=1000, effect_size=0.3, bandwidth=0.5).design == "rd"


def test_power_did_matches_closed_form_formula():
    res = sp.power_did(
        n=400,
        effect_size=0.3,
        n_periods=4,
        n_treated_periods=2,
        rho=0.5,
    )
    z_alpha = norm.ppf(0.975)
    se = (1.0 / 400**0.5) * ((1.0 + 3.0 * 0.5) / (2.0 * (1.0 - 2.0 / 4.0))) ** 0.5
    expected = norm.cdf(0.3 / se - z_alpha)
    assert res.power == pytest.approx(expected)


def test_power_rd_matches_closed_form_formula():
    res = sp.power_rd(n=1000, effect_size=0.3, bandwidth=0.5)
    z_alpha = norm.ppf(0.975)
    n_eff = 1000.0 * 0.5 * 1.0 * 0.75
    n_side = n_eff / 2.0
    se = (2.0 / n_side) ** 0.5
    expected = norm.cdf(0.3 / se - z_alpha)
    assert res.power == pytest.approx(expected)


def test_power_ols_matches_closed_form_formula():
    res = sp.power_ols(
        n=200,
        effect_size=0.2,
        n_covariates=3,
        r2_other=0.5,
    )
    z_alpha = norm.ppf(0.975)
    se = (1.0 - 0.5) ** 0.5 / (200.0 - 3.0 - 1.0) ** 0.5
    expected = norm.cdf(0.2 / se - z_alpha)
    assert res.power == pytest.approx(expected)


# --------------------------------------------------------------------------
# Monotonicity laws
# --------------------------------------------------------------------------
def test_power_increases_with_sample_size():
    assert _did(100) < _did(1000)
    assert _rd(200) < _rd(2000)


def test_power_increases_with_effect_size():
    assert _rd(1000, effect=0.1) < _rd(1000, effect=0.5)
    assert _did(300, effect=0.1) < _did(300, effect=0.6)


def test_iv_power_increases_with_first_stage_strength():
    # A stronger instrument (higher first-stage F) yields more power.
    assert _iv(500, f=5) < _iv(500, f=50)


def test_cluster_power_decreases_with_icc():
    # Higher intra-cluster correlation inflates the design effect -> less power.
    assert _cluster(40, icc=0.01) > _cluster(40, icc=0.30)


def test_cluster_power_increases_with_more_clusters():
    assert _cluster(10) < _cluster(80)


# --------------------------------------------------------------------------
# Null-effect boundary: power collapses to (around) the nominal level
# --------------------------------------------------------------------------
def test_zero_effect_gives_near_nominal_power():
    # Under H0 (effect = 0) the rejection rate equals the test's nominal size,
    # i.e. alpha/2 for the two-sided default — never appreciably above alpha.
    assert _did(500, effect=0.0) <= 0.06
    assert _rd(2000, effect=0.0) <= 0.06
