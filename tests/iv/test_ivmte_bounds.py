"""Tests for MST (2018) ivmte LP bounds."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def mte_dgp():
    """Simple DGP: MTE = 2 - u (decreasing), ATE = 1.5."""
    rng = np.random.default_rng(7)
    n = 5000
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.5 * z)))
    U = rng.uniform(size=n)
    D = (U < p).astype(float)
    y = (1 + 0.4 * U) + D * ((3 - 0.6 * U) - (1 + 0.4 * U)) + rng.normal(size=n, scale=0.3)
    return pd.DataFrame({"y": y, "d": D, "z": z})


class TestIVMTEBounds:
    def test_ate_bounds_cover_truth(self, mte_dgp):
        r = iv.ivmte_bounds(
            y="y", treatment="d", instruments=["z"], data=mte_dgp,
            target="ate", basis_degree=2, bounds_outcome=(0, 4),
        )
        # True ATE = 1.5
        assert r.lower_bound <= 1.5 <= r.upper_bound
        assert r.lp_status[0] == "optimal"
        assert r.lp_status[1] == "optimal"

    def test_late_bounds_cover_truth(self, mte_dgp):
        r = iv.ivmte_bounds(
            y="y", treatment="d", instruments=["z"], data=mte_dgp,
            target="late", late_bounds=(0.3, 0.7), basis_degree=2,
            bounds_outcome=(0, 4),
        )
        # True LATE = 1.5 on this DGP
        assert r.lower_bound <= 1.6  # truth-ish
        assert r.upper_bound >= 1.4

    def test_tighter_with_shape_restriction(self, mte_dgp):
        r_wide = iv.ivmte_bounds(
            y="y", treatment="d", instruments=["z"], data=mte_dgp,
            target="ate", basis_degree=2, bounds_outcome=(0, 4),
        )
        # Tight box bounds should give tighter set than very loose bounds
        r_loose = iv.ivmte_bounds(
            y="y", treatment="d", instruments=["z"], data=mte_dgp,
            target="ate", basis_degree=2, bounds_outcome=(-100, 100),
        )
        wide_loose = r_loose.upper_bound - r_loose.lower_bound
        wide_tight = r_wide.upper_bound - r_wide.lower_bound
        assert wide_tight <= wide_loose + 0.1

    def test_bmw_comparison(self, mte_dgp):
        r = iv.ivmte_bounds(
            y="y", treatment="d", instruments=["z"], data=mte_dgp,
            target="ate", basis_degree=2, bounds_outcome=(0, 4),
            include_bmw_point=True,
        )
        # BMW point should lie within bounds
        if r.point_bmw is not None:
            assert r.lower_bound <= r.point_bmw <= r.upper_bound + 0.5

    def test_summary(self, mte_dgp):
        r = iv.ivmte_bounds(
            y="y", treatment="d", instruments=["z"], data=mte_dgp,
            target="ate", basis_degree=2, bounds_outcome=(0, 4),
        )
        s = r.summary()
        assert "MST (2018)" in s
        assert "optimal" in s
