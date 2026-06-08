"""Tier D analytic special-case tests — power & minimum detectable effect.

Part of the P1 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). These four entry points previously had no
numerical-assertion test (``scripts/tierd_classify.py`` graded them
``untested`` / ``smoke``). Power calculations are pure closed forms in the
normal-approximation, so each test re-derives the exact value from
``scipy.stats.norm`` or exercises an exact limiting identity.

Entry points covered:
    sp.power('rct'|...)     dispatcher: power, or solve-for-n
    sp.mde                  minimum detectable effect (inverts power)
    sp.power_cluster_rct    cluster-RCT power with a design effect
    sp.power_iv             2SLS power with a weak-first-stage penalty

Underlying model (power.py): power = Phi(|es| * sqrt(N_eff_factor) - z_{alpha/2}).
Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pytest
from scipy.stats import norm

import statspai as sp

Z_ALPHA = norm.ppf(1 - 0.05 / 2)  # 1.959964...


# ---------------------------------------------------------------------------
# sp.power — two-arm RCT closed form and solve-for-n
# ---------------------------------------------------------------------------
class TestPowerRCTAnalytic:

    @pytest.mark.parametrize("n,es", [(400, 0.3), (1000, 0.2), (250, 0.5)])
    def test_rct_power_matches_normal_closed_form(self, n, es):
        # Equal allocation (ratio=1) -> p(1-p) = 0.25; power = Phi(es*sqrt(0.25 n) - z).
        expected = norm.cdf(es * np.sqrt(0.25 * n) - Z_ALPHA)
        res = sp.power("rct", n=n, effect_size=es, alpha=0.05)
        assert res.power == pytest.approx(expected, abs=1e-9)

    def test_rct_power_monotone_in_n(self):
        p_small = sp.power("rct", n=200, effect_size=0.3).power
        p_large = sp.power("rct", n=800, effect_size=0.3).power
        assert p_small < p_large

    def test_solve_for_n_is_minimal_to_hit_target(self):
        # Closed form: n* = (z_a + z_b)^2 / (es^2 * 0.25); the returned integer
        # n must achieve >= target while n-1 falls short (minimal sample size).
        es, target = 0.3, 0.8
        z_b = norm.ppf(target)
        n_closed = (Z_ALPHA + z_b) ** 2 / (es**2 * 0.25)
        res = sp.power("rct", n=None, effect_size=es, power_target=target)
        assert res.n == int(np.ceil(n_closed))
        assert sp.power("rct", n=res.n, effect_size=es).power >= target
        assert sp.power("rct", n=res.n - 1, effect_size=es).power < target


# ---------------------------------------------------------------------------
# sp.mde — minimum detectable effect inverts power exactly
# ---------------------------------------------------------------------------
class TestMDEAnalytic:

    @pytest.mark.parametrize("n,target", [(400, 0.8), (900, 0.9)])
    def test_mde_matches_closed_form(self, n, target):
        # MDE = (z_{alpha/2} + z_{power}) / sqrt(0.25 n) for the two-arm RCT.
        z_b = norm.ppf(target)
        expected = (Z_ALPHA + z_b) / np.sqrt(0.25 * n)
        res = sp.mde("rct", n=n, power_target=target)
        assert res.effect_size == pytest.approx(expected, abs=1e-5)

    def test_mde_round_trips_through_power(self):
        # Feeding the MDE back into power() must recover the target power.
        n, target = 400, 0.8
        mde_es = sp.mde("rct", n=n, power_target=target).effect_size
        assert sp.power("rct", n=n, effect_size=mde_es).power == pytest.approx(
            target, abs=1e-4
        )


# ---------------------------------------------------------------------------
# sp.power_cluster_rct — design-effect limiting identities
# ---------------------------------------------------------------------------
class TestPowerClusterRCTAnalytic:

    def test_icc_zero_equals_individual_rct(self):
        # ICC=0 -> design effect 1 -> n_eff = n_clusters * cluster_size, so a
        # cluster RCT collapses to an individual RCT on the full sample.
        nc, m, es = 40, 10, 0.2
        cluster = sp.power_cluster_rct(nc, m, es, icc=0.0).power
        indiv = sp.power("rct", n=nc * m, effect_size=es).power
        assert cluster == pytest.approx(indiv, abs=1e-9)

    def test_icc_one_equals_cluster_level_rct(self):
        # ICC=1 -> design effect = m -> n_eff = n_clusters: the cluster is the
        # effective unit, so power matches an RCT with N = n_clusters.
        nc, m, es = 40, 10, 0.5
        cluster = sp.power_cluster_rct(nc, m, es, icc=1.0).power
        indiv = sp.power("rct", n=nc, effect_size=es).power
        assert cluster == pytest.approx(indiv, abs=1e-9)

    def test_design_effect_formula(self):
        # Re-derive against power = Phi(es*sqrt(0.25 * n_eff) - z).
        nc, m, es, icc = 50, 20, 0.15, 0.05
        design_effect = 1 + (m - 1) * icc
        n_eff = nc * m / design_effect
        expected = norm.cdf(es * np.sqrt(0.25 * n_eff) - Z_ALPHA)
        assert sp.power_cluster_rct(nc, m, es, icc=icc).power == pytest.approx(
            expected, abs=1e-9
        )

    def test_power_decreases_with_icc(self):
        base = dict(n_clusters=40, cluster_size=15, effect_size=0.3)
        lo = sp.power_cluster_rct(**base, icc=0.01).power
        hi = sp.power_cluster_rct(**base, icc=0.20).power
        assert hi < lo  # more within-cluster correlation -> less effective info


# ---------------------------------------------------------------------------
# sp.power_iv — weak-first-stage penalty
# ---------------------------------------------------------------------------
class TestPowerIVAnalytic:

    def test_no_penalty_equals_ols_power(self):
        # With no first-stage information the penalty is 1: IV power equals the
        # OLS benchmark Phi(es*sqrt(n) - z) (note: no p(1-p) factor here).
        n, es = 500, 0.3
        expected = norm.cdf(es * np.sqrt(n) - Z_ALPHA)
        assert sp.power_iv(n, es).power == pytest.approx(expected, abs=1e-9)

    def test_first_stage_f_one_halves_power(self):
        # adjustment = F / (F + 1); F=1 -> exactly half the OLS-benchmark power.
        n, es = 500, 0.3
        ols = norm.cdf(es * np.sqrt(n) - Z_ALPHA)
        assert sp.power_iv(n, es, first_stage_f=1.0).power == pytest.approx(
            0.5 * ols, abs=1e-9
        )

    def test_strong_first_stage_recovers_ols(self):
        n, es = 500, 0.3
        ols = norm.cdf(es * np.sqrt(n) - Z_ALPHA)
        strong = sp.power_iv(n, es, first_stage_f=1e6).power
        assert strong == pytest.approx(ols, rel=1e-5)

    def test_r2z_path_matches_f_approximation(self):
        # r2_z route: F ~ n * r2_z / (1 - r2_z); adjustment = F / (F + 1).
        n, es, r2 = 500, 0.3, 0.1
        ols = norm.cdf(es * np.sqrt(n) - Z_ALPHA)
        f_approx = n * r2 / (1 - r2)
        expected = ols * f_approx / (f_approx + 1)
        assert sp.power_iv(n, es, r2_z=r2).power == pytest.approx(expected, abs=1e-7)

    def test_first_stage_f_takes_precedence_over_r2z(self):
        n, es = 500, 0.3
        with_both = sp.power_iv(n, es, first_stage_f=4.0, r2_z=0.9).power
        f_only = sp.power_iv(n, es, first_stage_f=4.0).power
        assert with_both == pytest.approx(f_only, abs=1e-12)
