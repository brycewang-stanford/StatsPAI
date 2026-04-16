"""Tests for sp.iv.{anderson_rubin_ci, conditional_lr_ci, k_test_ci}."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def strong_iv_dgp():
    rng = np.random.default_rng(0)
    n = 1000
    z = rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.8 * z + 0.5 * eps + rng.normal(size=n, scale=0.3)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z": z})


@pytest.fixture
def weak_iv_dgp():
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.03 * z + 0.8 * eps + rng.normal(size=n, scale=0.3)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z": z})


class TestAndersonRubinCI:
    def test_strong_id_covers_truth(self, strong_iv_dgp):
        ci = iv.anderson_rubin_ci(y="y", endog="d", instruments=["z"], data=strong_iv_dgp)
        assert not ci.is_empty
        assert ci.is_connected
        assert ci.lower <= 2.0 <= ci.upper
        # Narrow under strong ID
        assert ci.upper - ci.lower < 0.3

    def test_weak_id_can_be_disconnected_or_unbounded(self, weak_iv_dgp):
        ci = iv.anderson_rubin_ci(y="y", endog="d", instruments=["z"], data=weak_iv_dgp, n_grid=801)
        # At least SOMETHING funny should happen: wide, disconnected, or unbounded
        wide = (ci.upper - ci.lower) > 5
        assert wide or (not ci.is_connected) or ci.is_unbounded

    def test_empty_set_flagged(self, strong_iv_dgp):
        # Force an out-of-support grid → empty set
        grid = np.linspace(50, 60, 51)
        ci = iv.anderson_rubin_ci(
            y="y", endog="d", instruments=["z"], data=strong_iv_dgp,
            beta_grid=grid,
        )
        assert ci.is_empty

    def test_as_intervals(self, weak_iv_dgp):
        ci = iv.anderson_rubin_ci(
            y="y", endog="d", instruments=["z"], data=weak_iv_dgp, n_grid=801,
        )
        intervals = ci.as_intervals()
        if not ci.is_connected:
            assert len(intervals) >= 2


class TestConditionalLRCI:
    def test_strong_id_covers_truth(self, strong_iv_dgp):
        ci = iv.conditional_lr_ci(
            y="y", endog="d", instruments=["z"], data=strong_iv_dgp, random_state=0,
        )
        assert not ci.is_empty
        assert ci.lower <= 2.0 <= ci.upper

    def test_clr_tighter_than_ar_under_strong_id(self, strong_iv_dgp):
        # CLR is UMPI; should be at least as tight as AR when identification is strong
        ar = iv.anderson_rubin_ci(y="y", endog="d", instruments=["z"], data=strong_iv_dgp)
        clr = iv.conditional_lr_ci(
            y="y", endog="d", instruments=["z"], data=strong_iv_dgp, random_state=0,
            n_grid=401,  # match AR resolution
        )
        ar_width = ar.upper - ar.lower
        clr_width = clr.upper - clr.lower
        # CLR no wider than AR + small grid tolerance
        assert clr_width <= ar_width * 1.1


class TestKTestCI:
    def test_strong_id_covers_truth(self, strong_iv_dgp):
        ci = iv.k_test_ci(y="y", endog="d", instruments=["z"], data=strong_iv_dgp)
        assert not ci.is_empty
        assert ci.lower <= 2.0 <= ci.upper

    def test_overid(self):
        # Build a properly exogenous over-id DGP inside the test
        rng = np.random.default_rng(0)
        n = 1000
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        eps = rng.normal(size=n)
        d = 0.6 * z1 + 0.5 * z2 + 0.5 * eps + rng.normal(size=n, scale=0.3)
        y = 1 + 2.0 * d + eps
        df = pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2})
        ci = iv.k_test_ci(y="y", endog="d", instruments=["z1", "z2"], data=df)
        assert not ci.is_empty
        assert ci.lower <= 2.0 <= ci.upper
