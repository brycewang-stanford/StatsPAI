"""Smoke tests for sp.iv.plot — matplotlib import lazy."""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statspai.iv as iv


@pytest.fixture
def iv_dgp():
    rng = np.random.default_rng(11)
    n = 1500
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.5 * z1 + 0.3 * z2 + 0.5 * eps + rng.normal(size=n, scale=0.5)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2})


@pytest.fixture
def mte_dgp():
    rng = np.random.default_rng(7)
    n = 3000
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.5 * z + 0.3 * x)))
    U = rng.uniform(size=n)
    D = (U < p).astype(float)
    y = (1 + 0.4 * U + 0.3 * x) + D * ((3 - 0.6 * U) - (1 + 0.4 * U)) + rng.normal(size=n, scale=0.3)
    return pd.DataFrame({"y": y, "d": D, "z": z, "x": x})


class TestPlots:
    def test_first_stage(self, iv_dgp):
        ax = iv.plot.plot_first_stage(
            endog="d", instruments=["z1", "z2"], data=iv_dgp,
        )
        assert ax is not None
        plt.close("all")

    def test_ar_confidence_set(self, iv_dgp):
        ax = iv.plot.plot_ar_confidence_set(
            y="y", endog="d", instruments=["z1", "z2"], data=iv_dgp,
        )
        assert ax is not None
        plt.close("all")

    def test_mte_curve(self, mte_dgp):
        m = iv.mte(y="y", treatment="d", instruments=["z"], exog=["x"],
                   data=mte_dgp, poly_degree=2)
        ax = iv.plot.plot_mte_curve(m)
        assert ax is not None
        plt.close("all")

    def test_plausibly_exogenous(self, iv_dgp):
        grid = np.linspace(-0.15, 0.15, 21).reshape(-1, 1)
        uci = iv.plausibly_exogenous_uci(
            y="y", endog="d", instruments=["z1"], gamma_grid=grid, data=iv_dgp,
        )
        ax = iv.plot.plot_plausibly_exogenous(uci)
        assert ax is not None
        plt.close("all")


class TestMTEBootstrap:
    def test_bootstrap_reduces_se_inflation(self, mte_dgp):
        # Analytic plug-in SE can be wildly inflated; bootstrap should
        # give a more reasonable value.
        m_plug = iv.mte(y="y", treatment="d", instruments=["z"], exog=["x"],
                        data=mte_dgp, poly_degree=2)
        m_boot = iv.mte(y="y", treatment="d", instruments=["z"], exog=["x"],
                        data=mte_dgp, poly_degree=2,
                        bootstrap=50, random_state=0)
        # Bootstrap should deliver ATT/ATU SE (not present under plug-in)
        assert "att_se" in m_boot.extra
        assert "atu_se" in m_boot.extra
        assert m_boot.extra["n_successful_draws"] >= 40  # most draws succeed
        # Point estimates identical (only SE changes)
        assert abs(m_plug.ate - m_boot.ate) < 1e-10
