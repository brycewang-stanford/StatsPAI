"""``sp.optimal_design`` against the two-sample power formula.

Until 1.29.0 the individual and stratified branches reported the TOTAL sample
``z^2 sigma^2 / (mde^2 p (1 - p))`` as the per-arm size (doubling both
``n_per_arm`` and ``n_total``), the cluster branch doubled the number of
clusters the same way, the MDE branch ignored the sample size altogether, and
the cost-optimal cluster size was computed after -- and never used in -- the
sample-size calculation. These tests pin the formula, its inverse, and
agreement with ``sp.power`` / ``sp.power_cluster_rct``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import statspai as sp

Z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)


def test_individual_matches_power_formula_and_sp_power():
    res = sp.optimal_design(mde=0.2, sigma=1.0)
    per_arm = int(np.ceil(Z**2 / (0.2**2 * 0.5)))
    assert (res.n_per_arm, res.n_total) == (per_arm, 2 * per_arm)
    total_exact = Z**2 / (0.2**2 * 0.25)
    assert sp.power("rct", power=0.8, effect_size=0.2).n == int(np.ceil(total_exact))
    assert res.n_total - total_exact < 2  # per-arm rounding only


def test_unequal_allocation_sizes_each_arm_by_its_share():
    p = 0.3
    res = sp.optimal_design(mde=0.25, sigma=2.0, prop_treat=p)
    k = Z**2 * 4.0 / 0.25**2
    arms = [int(np.ceil(k / (1 - p))), int(np.ceil(k / p))]
    assert res.n_total == sum(arms)
    assert res.n_per_arm == max(arms)
    # The treated arm is p of the total, up to rounding.
    assert abs(arms[0] / res.n_total - p) < 1e-3


def test_mde_is_the_inverse_of_the_sample_size():
    res = sp.optimal_design(mde=0.2, sigma=1.0, r2=0.3)
    back = sp.optimal_design(n=res.n_total, sigma=1.0, r2=0.3)
    assert back.mde == pytest.approx(0.2, rel=5e-3)
    assert back.mde <= 0.2


def test_cluster_design_is_the_smallest_even_design_with_target_power():
    res = sp.optimal_design(
        design="cluster", mde=0.2, sigma=1.0, icc=0.05, cluster_size=20
    )
    assert res.n_clusters % 2 == 0
    assert res.n_total == res.n_clusters * 20
    ok = sp.power_cluster_rct(
        n_clusters=res.n_clusters, cluster_size=20, effect_size=0.2, icc=0.05
    )
    short = sp.power_cluster_rct(
        n_clusters=res.n_clusters - 2, cluster_size=20, effect_size=0.2, icc=0.05
    )
    assert ok.power >= 0.8 > short.power
    back = sp.optimal_design(
        design="cluster", n_clusters=res.n_clusters, icc=0.05, cluster_size=20
    )
    assert back.mde <= 0.2


def test_cost_optimal_cluster_size_drives_the_calculation():
    res = sp.optimal_design(
        design="cluster",
        mde=0.2,
        icc=0.05,
        cost_per_cluster=400.0,
        cost_per_unit=10.0,
    )
    m = int(np.round(np.sqrt(40.0 * 0.95 / 0.05)))
    assert res.cluster_size == m
    assert res.n_total == res.n_clusters * m
    fixed = sp.optimal_design(design="cluster", mde=0.2, icc=0.05, cluster_size=m)
    assert (fixed.n_clusters, fixed.n_total) == (res.n_clusters, res.n_total)


def test_multi_arm_sizes_every_arm_for_its_comparison_with_control():
    res = sp.optimal_design(mde=0.2, n_arms=3)
    per_arm = int(np.ceil(2 * Z**2 / 0.2**2))
    assert (res.n_per_arm, res.n_total) == (per_arm, 3 * per_arm)


class TestRefusals:
    def test_neither_mde_nor_sample_size(self):
        with pytest.raises(ValueError, match="mde= or n="):
            sp.optimal_design()
        with pytest.raises(ValueError, match="mde= or n_clusters="):
            sp.optimal_design(design="cluster", icc=0.05, cluster_size=20)

    def test_invalid_design_inputs(self):
        with pytest.raises(ValueError, match="Unknown design"):
            sp.optimal_design(design="factorial", mde=0.2)
        with pytest.raises(ValueError, match="prop_treat"):
            sp.optimal_design(mde=0.2, prop_treat=1.0)
        with pytest.raises(ValueError, match="icc"):
            sp.optimal_design(
                design="cluster", mde=0.2, cost_per_cluster=1.0, cost_per_unit=1.0
            )

    def test_cluster_options_on_individual_design_warn(self):
        with pytest.warns(UserWarning, match="ignores icc"):
            sp.optimal_design(mde=0.2, icc=0.05, cluster_size=20)

    def test_default_cluster_size_warns(self):
        with pytest.warns(UserWarning, match="assumes 20"):
            sp.optimal_design(design="cluster", mde=0.2, icc=0.05)
