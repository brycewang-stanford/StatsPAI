"""Tests for sp.iv.plausibly_exogenous_uci / plausibly_exogenous_ltz."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def dgp():
    rng = np.random.default_rng(123)
    n = 3000
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.6 * z1 + 0.4 * z2 + 0.5 * eps + rng.normal(size=n, scale=0.5)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2})


class TestLTZ:
    def test_zero_prior_variance_reproduces_2sls(self, dgp):
        res = iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_mean=0.0, gamma_var=0.0, data=dgp,
        )
        # γ=0, Ω=0 ⇒ exactly 2SLS
        assert abs(res.beta_hat - res.extra["beta_ltz"]) < 1e-8
        assert abs(res.se_hat - res.extra["se_ltz"]) < 1e-8

    def test_nonzero_variance_widens_ci(self, dgp):
        res0 = iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_mean=0.0, gamma_var=0.0, data=dgp,
        )
        res1 = iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_mean=0.0, gamma_var=0.05, data=dgp,
        )
        width0 = res0.ci_upper - res0.ci_lower
        width1 = res1.ci_upper - res1.ci_lower
        assert width1 > width0

    def test_nonzero_mean_shifts_estimate(self, dgp):
        res_pos = iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_mean=[0.1, 0.1], gamma_var=0.0, data=dgp,
        )
        res_zero = iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_mean=0.0, gamma_var=0.0, data=dgp,
        )
        assert res_pos.extra["beta_ltz"] != res_zero.beta_hat


class TestUCI:
    def test_zero_grid_equals_2sls_ci(self, dgp):
        res = iv.plausibly_exogenous_uci(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_grid=[[0.0, 0.0]], data=dgp,
        )
        ci_width_tsls = 2 * 1.96 * res.se_hat
        ci_width_uci = res.ci_upper - res.ci_lower
        # With γ-grid = {0} the union is just the 2SLS CI
        assert abs(ci_width_uci - ci_width_tsls) < 1e-4

    def test_wider_grid_widens_ci(self, dgp):
        narrow = iv.plausibly_exogenous_uci(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_grid=[[0.0, 0.0]], data=dgp,
        )
        grid = np.array([[g1, g2] for g1 in np.linspace(-0.1, 0.1, 5)
                         for g2 in np.linspace(-0.1, 0.1, 5)])
        wide = iv.plausibly_exogenous_uci(
            y="y", endog="d", instruments=["z1", "z2"],
            gamma_grid=grid, data=dgp,
        )
        assert (wide.ci_upper - wide.ci_lower) > (narrow.ci_upper - narrow.ci_lower)
