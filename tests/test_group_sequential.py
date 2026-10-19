"""Known-truth and identity tests for ``sp.group_sequential``.

Stata 18 ``gsbounds`` parity is in
``tests/reference_parity/test_group_sequential_stata_parity.py``; here the
boundaries are checked against facts that do not depend on any reference:
a single look is the fixed design, the crossing probabilities of a two-look
design equal bivariate normal probabilities computed independently, the
spending functions are spent exactly, lower designs mirror upper ones, and a
Monte Carlo of the Brownian-motion model reproduces alpha and power.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def test_single_look_is_the_fixed_design():
    d = sp.group_sequential(n_looks=1, alpha=0.05, power=0.8)
    assert d.boundaries["efficacy_upper"].iloc[0] == pytest.approx(
        stats.norm.isf(0.025), rel=1e-12
    )
    assert d.info_ratio == pytest.approx(1.0, abs=1e-9)
    one = sp.group_sequential(n_looks=1, side="upper", alpha=0.025)
    assert one.boundaries["efficacy_upper"].iloc[0] == pytest.approx(
        stats.norm.isf(0.025), rel=1e-12
    )


@pytest.mark.parametrize("efficacy", ["obfleming", "pocock", "errobfleming"])
def test_two_look_alpha_equals_bivariate_normal_probability(efficacy):
    # Two-sided: P(|Z1| >= c1) + P(|Z1| < c1, |Z2| >= c2) = alpha, with
    # corr(Z1, Z2) = sqrt(t1). The bivariate normal CDF (Genz) is an
    # independent route to the same number.
    t1 = 0.4
    d = sp.group_sequential(information=[t1, 1.0], efficacy=efficacy)
    c1, c2 = d.boundaries["efficacy_upper"]
    rho = np.sqrt(t1)
    bvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    inside = (
        bvn.cdf([c1, c2])
        - bvn.cdf([-c1, c2])
        - bvn.cdf([c1, -c2])
        + bvn.cdf([-c1, -c2])
    )
    alpha = 1.0 - inside
    assert alpha == pytest.approx(0.05, abs=2e-7)


def test_spending_function_is_spent_exactly():
    d = sp.group_sequential(
        information=[0.2, 0.45, 0.7, 1.0], efficacy="hsdecani", efficacy_param=-4
    )
    t = np.array([0.2, 0.45, 0.7, 1.0])
    target = 0.05 * (1 - np.exp(4 * t)) / (1 - np.exp(4))
    np.testing.assert_allclose(d.boundaries["alpha_spent"], target, atol=1e-12)


def test_lan_demets_obf_two_sided_splits_alpha_by_tail():
    # alpha*(t) = 4 - 4 Phi(z_{1 - alpha/4} / sqrt(t)) for two-sided designs.
    d = sp.group_sequential(n_looks=4, efficacy="errobfleming")
    t = np.arange(1, 5) / 4
    target = 4 * stats.norm.sf(stats.norm.isf(0.05 / 4) / np.sqrt(t))
    np.testing.assert_allclose(d.boundaries["alpha_spent"], target, atol=1e-12)


def test_lower_design_mirrors_upper():
    kw = dict(n_looks=3, efficacy="errobfleming", futility="errpocock")
    up = sp.group_sequential(side="upper", **kw)
    lo = sp.group_sequential(side="lower", **kw)
    np.testing.assert_allclose(
        lo.boundaries["efficacy_lower"], -up.boundaries["efficacy_upper"], rtol=1e-12
    )
    np.testing.assert_allclose(
        lo.boundaries["futility_upper"], -up.boundaries["futility_lower"], rtol=1e-12
    )
    assert lo.info_ratio == pytest.approx(up.info_ratio, rel=1e-12)


def test_futility_designs_close_at_the_final_look():
    d = sp.group_sequential(
        n_looks=3, efficacy="errobfleming", futility="errpocock", side="upper"
    )
    b = d.boundaries
    assert b["futility_lower"].iloc[-1] == pytest.approx(
        b["efficacy_upper"].iloc[-1], rel=1e-14
    )
    t = np.arange(1, 4) / 3
    beta_target = 0.2 * np.log1p((np.e - 1) * t)
    np.testing.assert_allclose(b["beta_spent"], beta_target, atol=1e-9)
    assert d.power_attained == pytest.approx(0.8, abs=1e-9)
    # Nonbinding: efficacy bounds equal the efficacy-only design's.
    e = sp.group_sequential(n_looks=3, efficacy="errobfleming", side="upper")
    np.testing.assert_allclose(b["efficacy_upper"], e.boundaries["efficacy_upper"])


def _simulate(d, theta_drift, n_rep, seed):
    """Monte Carlo of the canonical joint distribution of (Z_1, ..., Z_K)."""
    rng = np.random.default_rng(seed)
    t = np.asarray(d.boundaries["info_frac"])
    info = t * d.info_ratio * d.detail["info_fixed_per_unit_drift"]
    inc = np.diff(np.r_[0.0, info])
    s = np.cumsum(
        theta_drift * inc + np.sqrt(inc) * rng.standard_normal((n_rep, len(t))), axis=1
    )
    z = s / np.sqrt(info)
    b = d.boundaries
    up = np.nan_to_num(b["efficacy_upper"].to_numpy(), nan=np.inf)
    fl = np.nan_to_num(b["futility_lower"].to_numpy(), nan=-np.inf)
    alive = np.ones(n_rep, dtype=bool)
    reject = np.zeros(n_rep, dtype=bool)
    for k in range(len(t)):
        hit = alive & (z[:, k] >= up[k])
        reject |= hit
        alive &= ~hit & (z[:, k] > fl[k])
    return reject.mean()


@pytest.mark.parametrize(
    "kw",
    [
        dict(n_looks=4, efficacy="pocock", side="upper"),
        dict(
            n_looks=3,
            efficacy="errobfleming",
            futility="errpocock",
            side="upper",
            binding=True,
        ),
    ],
)
def test_monte_carlo_reproduces_alpha_and_power(kw):
    # S-level check: 200,000 draws, MC SE of alpha ~5e-4, of power ~9e-4;
    # tolerance 4 MC SE.
    d = sp.group_sequential(**kw)
    n_rep = 200_000
    a = _simulate(d, 0.0, n_rep, seed=11)
    p = _simulate(d, 1.0, n_rep, seed=12)
    assert abs(a - 0.05) < 4 * np.sqrt(0.05 * 0.95 / n_rep)
    assert abs(p - 0.8) < 4 * np.sqrt(0.8 * 0.2 / n_rep)


def test_binding_futility_lowers_efficacy_bounds_and_keeps_alpha():
    nb = sp.group_sequential(
        n_looks=3, efficacy="errobfleming", futility="errpocock", side="upper"
    )
    bd = sp.group_sequential(
        n_looks=3,
        efficacy="errobfleming",
        futility="errpocock",
        side="upper",
        binding=True,
    )
    # With futility stops honoured, the binding design spends exactly alpha.
    assert bd.boundaries["alpha_spent_futility"].iloc[-1] == pytest.approx(
        0.05, abs=1e-10
    )
    # The nonbinding design spends alpha only if futility is ignored.
    assert nb.boundaries["alpha_spent"].iloc[-1] == pytest.approx(0.05, abs=1e-10)
    assert bd.info_ratio < nb.info_ratio


def test_sample_sizes_scale_with_n_fixed():
    d = sp.group_sequential(n_looks=3, n_fixed=300.0)
    assert d.n_max == pytest.approx(300.0 * d.info_ratio, rel=1e-14)
    np.testing.assert_allclose(d.boundaries["n"], d.n_max * np.arange(1, 4) / 3)


def test_decision_applies_the_bounds():
    d = sp.group_sequential(
        n_looks=3, efficacy="errobfleming", futility="errpocock", side="upper"
    )
    assert d.decision([1.0, 2.5]).action.tolist() == ["continue", "stop: efficacy"]
    assert d.decision([0.1]).action.tolist() == ["stop: futility"]
    assert d.decision([1.0, 1.5, 1.6]).action.tolist()[-1] == "stop: futility"
    two = sp.group_sequential(n_looks=2)
    assert two.decision([-3.0]).action.tolist() == ["stop: efficacy"]
    with pytest.raises(MethodIncompatibility):
        d.decision([1, 1, 1, 1])


def test_invalid_specifications_raise():
    with pytest.raises(MethodIncompatibility):
        sp.group_sequential(efficacy="obfleming", futility="errpocock")
    with pytest.raises(MethodIncompatibility):
        sp.group_sequential(binding=True)
    with pytest.raises(MethodIncompatibility):
        sp.group_sequential(efficacy="kdemets")  # needs rho
    with pytest.raises(MethodIncompatibility):
        sp.group_sequential(efficacy="nonsense")
    with pytest.raises(MethodIncompatibility):
        sp.group_sequential(information=[1, 0.5])
    with pytest.raises(MethodIncompatibility):
        sp.group_sequential(alpha=0.7)
