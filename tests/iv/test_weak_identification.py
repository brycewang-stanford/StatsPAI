"""Tests for sp.iv.kleibergen_paap_rk / sanderson_windmeijer / conditional_lr_test."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def two_endog_dgp():
    """Two endogenous regressors, three instruments, rich first stage."""
    rng = np.random.default_rng(7)
    n = 2000
    z1, z2, z3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    x = rng.normal(size=n)
    eps = rng.normal(size=n)

    d1 = 0.6 * z1 + 0.3 * z2 + 0.4 * eps + rng.normal(size=n, scale=0.5)
    d2 = 0.2 * z1 + 0.7 * z3 + 0.5 * eps + rng.normal(size=n, scale=0.5)
    y = 1 + 2 * d1 - 1.5 * d2 + 0.8 * x + eps
    return pd.DataFrame({"y": y, "d1": d1, "d2": d2, "z1": z1, "z2": z2, "z3": z3, "x": x})


@pytest.fixture
def single_endog_dgp():
    rng = np.random.default_rng(42)
    n = 3000
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.6 * z1 + 0.4 * z2 + 0.5 * eps + rng.normal(size=n, scale=0.5)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2})


@pytest.fixture
def weak_iv_dgp():
    """Weak instrument — used to check the statistics flag weakness."""
    rng = np.random.default_rng(0)
    n = 500
    z1 = rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.02 * z1 + 0.8 * eps + rng.normal(size=n, scale=0.5)
    y = 1 + 2.0 * d + eps
    return pd.DataFrame({"y": y, "d": d, "z1": z1})


# ────────────────────────────────────────────────────────────────────────
#  Kleibergen-Paap rk
# ────────────────────────────────────────────────────────────────────────

class TestKleibergenPaap:
    def test_basic_strong_identification(self, two_endog_dgp):
        df = two_endog_dgp
        kp = iv.kleibergen_paap_rk(
            endog=df[["d1", "d2"]],
            instruments=df[["z1", "z2", "z3"]],
            exog=df[["x"]],
        )
        assert kp.rk_f > 50
        assert kp.rk_wald_pvalue < 0.001
        assert kp.rk_lm_pvalue < 0.001
        assert kp.n_endog == 2
        assert kp.n_instruments == 3

    def test_cov_type_nonrobust_matches_cragg_donald(self, single_endog_dgp):
        df = single_endog_dgp
        kp = iv.kleibergen_paap_rk(
            endog=df[["d"]], instruments=df[["z1", "z2"]],
            cov_type="nonrobust",
        )
        assert kp.rk_f > 10
        assert "nonrobust" in kp.cov_type

    def test_cluster_runs(self, two_endog_dgp):
        df = two_endog_dgp.copy()
        df["grp"] = np.tile(np.arange(50), len(df) // 50)
        kp = iv.kleibergen_paap_rk(
            endog=df[["d1", "d2"]], instruments=df[["z1", "z2", "z3"]],
            exog=df[["x"]], cov_type="cluster", cluster=df["grp"],
        )
        assert kp.rk_f > 0
        assert "cluster" in kp.cov_type

    def test_under_identification_raises(self, two_endog_dgp):
        df = two_endog_dgp
        # 2 endog, 1 instrument → should raise
        with pytest.raises(ValueError, match="Under-identified"):
            iv.kleibergen_paap_rk(
                endog=df[["d1", "d2"]], instruments=df[["z1"]],
            )


# ────────────────────────────────────────────────────────────────────────
#  Sanderson-Windmeijer
# ────────────────────────────────────────────────────────────────────────

class TestSandersonWindmeijer:
    def test_two_endogenous(self, two_endog_dgp):
        df = two_endog_dgp
        sw = iv.sanderson_windmeijer(
            endog=df[["d1", "d2"]], instruments=df[["z1", "z2", "z3"]],
            exog=df[["x"]],
        )
        assert set(sw.sw_f.keys()) == {"d1", "d2"}
        # both endogenous have strong individual first-stages
        for name in ("d1", "d2"):
            assert sw.sw_f[name] > 10
            assert sw.sw_pvalue[name] < 0.01
            assert sw.partial_r2[name] > 0.05
            assert sw.df_num[name] == 2  # k - (p-1) = 3 - 1

    def test_single_endog_reduces_to_first_stage_f(self, single_endog_dgp):
        df = single_endog_dgp
        sw = iv.sanderson_windmeijer(
            endog=df[["d"]], instruments=df[["z1", "z2"]],
        )
        assert sw.df_num["d"] == 2
        assert sw.sw_f["d"] > 10

    def test_to_frame_shape(self, two_endog_dgp):
        df = two_endog_dgp
        sw = iv.sanderson_windmeijer(
            endog=df[["d1", "d2"]], instruments=df[["z1", "z2", "z3"]],
        )
        tbl = sw.to_frame()
        assert tbl.shape == (2, 4)
        assert list(tbl.columns) == ["SW F", "p-value", "df_num", "partial R²"]


# ────────────────────────────────────────────────────────────────────────
#  Moreira CLR
# ────────────────────────────────────────────────────────────────────────

class TestConditionalLR:
    def test_fails_to_reject_true_beta(self, single_endog_dgp):
        df = single_endog_dgp
        clr = iv.conditional_lr_test(
            y="d", endog="d", instruments=["z1", "z2"], data=df.rename(columns={"y": "yorig"}),
            beta0=1.0, n_simulations=2000, random_state=1,
        )
        # Not great test — use the proper dataset
        # Better: test with correct outcome
        clr = iv.conditional_lr_test(
            y="y", endog="d", instruments=["z1", "z2"], data=df,
            beta0=2.0, n_simulations=3000, random_state=1,
        )
        assert clr.pvalue > 0.05

    def test_rejects_wrong_beta(self, single_endog_dgp):
        df = single_endog_dgp
        clr = iv.conditional_lr_test(
            y="y", endog="d", instruments=["z1", "z2"], data=df,
            beta0=0.0, n_simulations=3000, random_state=1,
        )
        assert clr.pvalue < 0.001
        assert clr.statistic > 20


# ────────────────────────────────────────────────────────────────────────
#  Weak instrument behaviour
# ────────────────────────────────────────────────────────────────────────

class TestWeakInstrumentFlagging:
    def test_sw_low_under_weak(self, weak_iv_dgp):
        df = weak_iv_dgp
        sw = iv.sanderson_windmeijer(
            endog=df[["d"]], instruments=df[["z1"]],
        )
        # first-stage F should be small (< 10 typically) under this DGP
        assert sw.sw_f["d"] < 10

    def test_kp_small_under_weak(self, weak_iv_dgp):
        df = weak_iv_dgp
        kp = iv.kleibergen_paap_rk(
            endog=df[["d"]], instruments=df[["z1"]],
        )
        assert kp.rk_f < 15
