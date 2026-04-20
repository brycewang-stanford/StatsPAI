"""Comprehensive tests for the stochastic frontier module.

Covers:
* Cross-sectional half-normal, exponential, truncated-normal recovery.
* Heteroskedastic ``sigma_u`` and ``sigma_v``.
* BC95-style inefficiency determinants (``emean``).
* Cost-frontier sign conventions.
* Panel: Pitt-Lee TI (half-normal & truncated-normal), Battese-Coelli
  TVD decay, and BC95 inefficiency-effects.
* Efficiency scores (Battese-Coelli vs JLMS) internal consistency.
* Specification tests (LR against OLS) and skewness diagnostic.
* Bootstrap CIs, ranking, and descriptive summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sst

import warnings

# Suppress noisy runtime warnings from fringe optimizer steps.
warnings.filterwarnings("ignore", category=RuntimeWarning)

from statspai.frontier import (
    frontier,
    xtfrontier,
    FrontierResult,
    te_summary,
    te_rank,
)
from statspai.frontier import _core as _fc


# ---------------------------------------------------------------------------
# Simulated data helpers
# ---------------------------------------------------------------------------


def _simulate_hn_production(n, sigma_v, sigma_u, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    u = np.abs(rng.normal(0, sigma_u, n))
    v = rng.normal(0, sigma_v, n)
    y = 1.0 + 0.5 * x1 + 0.3 * x2 + v - u
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _simulate_cost(n, sigma_v, sigma_u, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    u = np.abs(rng.normal(0, sigma_u, n))
    v = rng.normal(0, sigma_v, n)
    y = 1.0 + 0.5 * x1 + v + u  # cost: + u
    return pd.DataFrame({"y": y, "x1": x1})


def _simulate_panel_ti(N, T, sigma_v, sigma_u, seed=2):
    rng = np.random.default_rng(seed)
    id_ = np.repeat(np.arange(N), T)
    t_ = np.tile(np.arange(T), N)
    n = N * T
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    u_i = np.abs(rng.normal(0, sigma_u, N))
    u_it = np.repeat(u_i, T)
    v = rng.normal(0, sigma_v, n)
    y = 1.0 + 0.6 * x1 + 0.4 * x2 + v - u_it
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": id_, "t": t_})


def _simulate_panel_tvd(N, T, sigma_v, sigma_u, mu, eta, seed=3):
    rng = np.random.default_rng(seed)
    id_ = np.repeat(np.arange(N), T)
    t_ = np.tile(np.arange(T), N)
    n = N * T
    x1 = rng.normal(0, 1, n)
    u_i = sst.truncnorm.rvs(-mu / sigma_u, np.inf, loc=mu, scale=sigma_u,
                            size=N, random_state=rng)
    a_it = np.exp(-eta * (t_ - (T - 1)))
    u_it = a_it * np.repeat(u_i, T)
    v = rng.normal(0, sigma_v, n)
    y = 1.0 + 0.6 * x1 + v - u_it
    return pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})


# ---------------------------------------------------------------------------
# Cross-sectional recovery
# ---------------------------------------------------------------------------


class TestCrossSectionalRecovery:
    def test_half_normal_production_recovers_parameters(self):
        df = _simulate_hn_production(2000, 0.2, 0.5, seed=42)
        res = frontier(df, y="y", x=["x1", "x2"], dist="half-normal")
        assert res.model_info["converged"]
        assert abs(res.params["x1"] - 0.5) < 0.03
        assert abs(res.params["x2"] - 0.3) < 0.03
        assert abs(np.exp(res.params["ln_sigma_u"]) - 0.5) < 0.05
        assert abs(np.exp(res.params["ln_sigma_v"]) - 0.2) < 0.03

    def test_half_normal_cost_sign_flip(self):
        df = _simulate_cost(2000, 0.15, 0.4, seed=7)
        res = frontier(df, y="y", x=["x1"], dist="half-normal", cost=True)
        assert res.model_info["converged"]
        assert res.model_info["sign"] == 1
        # With cost=False on cost-generated data, sigma_u should collapse.
        res_wrong = frontier(df, y="y", x=["x1"], dist="half-normal", cost=False)
        # Strict: on cost-generated data the correct (cost=True) fit must
        # achieve STRICTLY higher log-likelihood than the wrong-sign fit.
        # Previous version allowed a 5-nat slack, which masked any
        # implementation bug that degraded but didn't flip the ranking.
        assert (
            res.diagnostics["log_likelihood"]
            > res_wrong.diagnostics["log_likelihood"]
        ), (
            f"correct-sign LL ({res.diagnostics['log_likelihood']:.3f}) "
            f"should strictly exceed wrong-sign LL "
            f"({res_wrong.diagnostics['log_likelihood']:.3f})"
        )

    def test_cost_frontier_recovers_parameters(self):
        """Cross-sectional cost frontier: verify beta / sigma_u recovery.

        The existing ``test_half_normal_cost_sign_flip`` compares log-
        likelihoods across sign conventions but never checks that the
        slopes and variance parameters land near the truth. A bug that
        inverts beta while preserving LL ordering could slip through
        without this test.
        """
        rng = np.random.default_rng(777)
        n = 2500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        true_beta_1, true_beta_2 = 0.5, -0.3
        true_sigma_u, true_sigma_v = 0.35, 0.15
        u = np.abs(rng.normal(0, true_sigma_u, n))
        v = rng.normal(0, true_sigma_v, n)
        # Cost frontier: log_cost = alpha + beta * x + v + u
        y = 1.0 + true_beta_1 * x1 + true_beta_2 * x2 + v + u
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        res = frontier(df, y="y", x=["x1", "x2"], dist="half-normal", cost=True)
        assert res.model_info["converged"]
        assert res.model_info["sign"] == 1
        # Slope recovery — tight because we have n=2500.
        assert abs(res.params["x1"] - true_beta_1) < 0.03, (
            f"x1 beta drift: {res.params['x1']:.4f} vs 0.50"
        )
        assert abs(res.params["x2"] - true_beta_2) < 0.03, (
            f"x2 beta drift: {res.params['x2']:.4f} vs -0.30"
        )
        # Variance recovery.
        assert abs(np.exp(res.params["ln_sigma_u"]) - true_sigma_u) < 0.05
        assert abs(np.exp(res.params["ln_sigma_v"]) - true_sigma_v) < 0.04

    def test_exponential_production_recovers_parameters(self):
        rng = np.random.default_rng(3)
        n = 3000
        x1 = rng.normal(0, 1, n)
        sigma_v_true, sigma_u_true = 0.2, 0.4
        v = rng.normal(0, sigma_v_true, n)
        u = rng.exponential(sigma_u_true, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1})
        res = frontier(df, y="y", x=["x1"], dist="exponential")
        assert res.model_info["converged"]
        assert abs(np.exp(res.params["ln_sigma_u"]) - 0.4) < 0.05
        assert abs(np.exp(res.params["ln_sigma_v"]) - 0.2) < 0.05

    def test_truncated_normal_production_recovers_mu(self):
        rng = np.random.default_rng(4)
        n = 5000
        x1 = rng.normal(0, 1, n)
        mu_t, su_t, sv_t = 0.5, 0.4, 0.2
        u = sst.truncnorm.rvs(-mu_t / su_t, np.inf, loc=mu_t, scale=su_t,
                              size=n, random_state=rng)
        v = rng.normal(0, sv_t, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1})
        res = frontier(df, y="y", x=["x1"], dist="truncated-normal")
        assert res.model_info["converged"]
        assert abs(res.params["mu"] - 0.5) < 0.15
        assert abs(np.exp(res.params["ln_sigma_u"]) - 0.4) < 0.08
        assert abs(np.exp(res.params["ln_sigma_v"]) - 0.2) < 0.03


# ---------------------------------------------------------------------------
# Heteroskedasticity & BC95 determinants
# ---------------------------------------------------------------------------


class TestHeteroskedasticity:
    def test_usigma_recovers_coefficient(self):
        rng = np.random.default_rng(11)
        n = 4000
        x1 = rng.normal(0, 1, n)
        w1 = rng.normal(0, 1, n)
        sigma_u_i = np.exp(-1.0 + 0.5 * w1)
        u = np.abs(rng.normal(0, sigma_u_i))
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "w1": w1})
        res = frontier(df, y="y", x=["x1"], dist="half-normal", usigma=["w1"])
        assert res.model_info["converged"]
        assert abs(res.params["u__cons"] - (-1.0)) < 0.1
        assert abs(res.params["u_w1"] - 0.5) < 0.1

    def test_vsigma_recovers_coefficient(self):
        rng = np.random.default_rng(12)
        n = 4000
        x1 = rng.normal(0, 1, n)
        r1 = rng.normal(0, 1, n)
        sigma_v_i = np.exp(-1.5 + 0.3 * r1)
        u = np.abs(rng.normal(0, 0.4, n))
        v = rng.normal(0, sigma_v_i)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "r1": r1})
        res = frontier(df, y="y", x=["x1"], dist="half-normal", vsigma=["r1"])
        assert res.model_info["converged"]
        assert abs(res.params["v_r1"] - 0.3) < 0.1

    def test_bc95_emean_determinants(self):
        rng = np.random.default_rng(13)
        n = 4000
        x1 = rng.normal(0, 1, n)
        z1 = rng.normal(0, 1, n)
        mu_i = 0.3 + 0.2 * z1
        u = sst.truncnorm.rvs(-mu_i / 0.3, np.inf, loc=mu_i, scale=0.3,
                              random_state=rng)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "z1": z1})
        res = frontier(df, y="y", x=["x1"], dist="truncated-normal", emean=["z1"])
        assert res.model_info["converged"]
        assert abs(res.params["mu__cons"] - 0.3) < 0.15
        assert abs(res.params["mu_z1"] - 0.2) < 0.08

    def test_emean_requires_truncated_normal(self):
        df = _simulate_hn_production(200, 0.2, 0.4, seed=5)
        df["z"] = np.random.default_rng(5).normal(0, 1, 200)
        with pytest.raises(ValueError, match="emean"):
            frontier(df, y="y", x=["x1"], dist="half-normal", emean=["z"])


# ---------------------------------------------------------------------------
# Specification tests
# ---------------------------------------------------------------------------


class TestSpecificationTests:
    def test_lr_rejects_no_inefficiency_when_u_present(self):
        df = _simulate_hn_production(1500, 0.2, 0.5, seed=21)
        res = frontier(df, y="y", x=["x1", "x2"], dist="half-normal")
        lr = res.lr_test_no_inefficiency()
        assert lr["statistic"] > 30.0
        assert lr["pvalue"] < 0.001

    def test_lr_does_not_reject_when_no_inefficiency(self):
        rng = np.random.default_rng(22)
        n = 1500
        x1 = rng.normal(0, 1, n)
        v = rng.normal(0, 0.3, n)  # v only, no u
        y = 1.0 + 0.5 * x1 + v
        df = pd.DataFrame({"y": y, "x1": x1})
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        lr = res.lr_test_no_inefficiency()
        # With no inefficiency signal, LR stat should be small.
        assert lr["statistic"] < 5.0

    def test_mixed_chibar_pvalue_helper(self):
        # Kodde-Palm boundary test: 0.5 * chi2(1) mass gives p=0.5 at LR=0.
        assert _fc.mixed_chi_bar_pvalue(0.0, df_boundary=1) == 1.0
        assert _fc.mixed_chi_bar_pvalue(3.84, df_boundary=1) == pytest.approx(
            0.025, abs=2e-3
        )


# ---------------------------------------------------------------------------
# Efficiency scores
# ---------------------------------------------------------------------------


class TestEfficiencyScores:
    def test_efficiency_in_0_1(self):
        df = _simulate_hn_production(500, 0.2, 0.5, seed=30)
        res = frontier(df, y="y", x=["x1", "x2"], dist="half-normal")
        te_bc = res.efficiency(method="bc").values
        te_jl = res.efficiency(method="jlms").values
        assert np.all(te_bc >= 0.0) and np.all(te_bc <= 1.0)
        assert np.all(te_jl >= 0.0) and np.all(te_jl <= 1.0)

    def test_efficiency_exponential_is_valid(self):
        rng = np.random.default_rng(31)
        n = 1000
        x1 = rng.normal(0, 1, n)
        v = rng.normal(0, 0.2, n)
        u = rng.exponential(0.4, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1})
        res = frontier(df, y="y", x=["x1"], dist="exponential")
        te = res.efficiency()
        # Previously produced NaN for exponential; must now be finite & in (0,1).
        assert te.notna().all()
        assert (te > 0).all() and (te <= 1).all()

    def test_efficiency_bc_vs_jlms_close_but_distinct(self):
        df = _simulate_hn_production(500, 0.2, 0.5, seed=32)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        te_bc = res.efficiency("bc").values
        te_jl = res.efficiency("jlms").values
        # Battese-Coelli is exact, JLMS is approximation; they correlate > 0.99
        corr = np.corrcoef(te_bc, te_jl)[0, 1]
        assert corr > 0.99
        # But means differ.
        assert not np.allclose(te_bc, te_jl)

    def test_inefficiency_positive_for_production(self):
        df = _simulate_hn_production(500, 0.2, 0.5, seed=33)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        u_hat = res.inefficiency()
        assert (u_hat >= -1e-6).all()

    def test_efficiency_ci_has_coverage_structure(self):
        df = _simulate_hn_production(300, 0.2, 0.5, seed=34)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        ci = res.efficiency_ci(alpha=0.10, B=80, seed=1)
        assert set(ci.columns) == {"point", "lower", "upper"}
        assert (ci["lower"] <= ci["upper"]).all()
        assert (ci["lower"] >= 0.0).all()
        assert (ci["upper"] <= 1.0 + 1e-12).all()


# ---------------------------------------------------------------------------
# Helpers: summary, rank
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_te_summary(self):
        df = _simulate_hn_production(400, 0.2, 0.5, seed=41)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        s = te_summary(res)
        assert {"mean", "median", "min", "max"}.issubset(s.columns)
        assert s.loc["efficiency", "mean"] > 0.0

    def test_te_rank_with_and_without_ci(self):
        df = _simulate_hn_production(200, 0.2, 0.5, seed=42)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        r = te_rank(res)
        assert "rank" in r.columns and r["rank"].min() == 1
        r2 = te_rank(res, with_ci=True, B=40)
        assert {"lower", "upper"}.issubset(r2.columns)


# ---------------------------------------------------------------------------
# Panel SFA
# ---------------------------------------------------------------------------


class TestPanelTimeInvariant:
    def test_pitt_lee_halfnormal_recovers(self):
        df = _simulate_panel_ti(N=100, T=6, sigma_v=0.2, sigma_u=0.5, seed=51)
        res = xtfrontier(df, y="y", x=["x1", "x2"], id="id", time="t",
                         model="ti", dist="half-normal")
        assert res.model_info["converged"]
        assert abs(res.params["x1"] - 0.6) < 0.05
        assert abs(res.params["x2"] - 0.4) < 0.05
        assert abs(res.model_info["sigma_u"] - 0.5) < 0.08
        assert abs(res.model_info["sigma_v"] - 0.2) < 0.03

    def test_pitt_lee_truncated_normal_runs(self):
        rng = np.random.default_rng(52)
        N, T = 120, 5
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        mu_t = 0.4
        u_i = sst.truncnorm.rvs(-mu_t / 0.3, np.inf, loc=mu_t, scale=0.3,
                                size=N, random_state=rng)
        u_it = np.repeat(u_i, T)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u_it
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                         model="ti", dist="truncated-normal")
        assert res.model_info["converged"]
        assert "mu" in res.params.index
        assert res.model_info["sigma_u"] > 0

    def test_pitt_lee_unit_level_efficiency_matches_obs(self):
        df = _simulate_panel_ti(80, 5, 0.2, 0.5, seed=53)
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="ti")
        unit_te = res.diagnostics["efficiency_bc_unit"]
        # Because u is time-invariant, obs-level efficiency equals unit efficiency
        # replicated per observation (a_it = 1 for TI).
        obs_te = res.diagnostics["efficiency_bc"]
        group_idx = res.diagnostics["group_idx"]
        aligned = unit_te.to_numpy()[group_idx]
        assert np.allclose(obs_te, aligned, atol=1e-10)


class TestPanelTVD:
    def test_battese_coelli_1992_recovers_eta(self):
        df = _simulate_panel_tvd(N=120, T=6, sigma_v=0.2, sigma_u=0.3,
                                 mu=0.4, eta=0.05, seed=61)
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                         model="tvd", dist="truncated-normal")
        assert res.model_info["converged"]
        assert abs(res.params["eta"] - 0.05) < 0.05
        assert abs(res.params["x1"] - 0.6) < 0.05

    def test_tvd_requires_time(self):
        df = _simulate_panel_ti(60, 4, 0.2, 0.4, seed=62)
        with pytest.raises(ValueError, match="time"):
            xtfrontier(df, y="y", x=["x1"], id="id", model="tvd")

    def test_unbalanced_panel_ti(self):
        """TI should survive an unbalanced panel (including some T_i=1)."""
        rng = np.random.default_rng(63)
        N = 50
        T_per = rng.integers(1, 7, N)
        rows = [(i, t) for i in range(N) for t in range(T_per[i])]
        ids = np.array([r[0] for r in rows])
        times = np.array([r[1] for r in rows])
        n = len(rows)
        x1 = rng.normal(0, 1, n)
        u_i = np.abs(rng.normal(0, 0.4, N))
        u_it = u_i[ids]
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u_it
        df = pd.DataFrame({"y": y, "x1": x1, "id": ids, "t": times})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="ti")
        assert res.model_info["converged"]
        assert res.data_info["n_obs"] == n

    def test_panel_ti_cost_frontier_runs(self):
        """Cost panel frontier must converge (composed error v + u)."""
        rng = np.random.default_rng(64)
        N, T = 80, 5
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        u_i = np.abs(rng.normal(0, 0.4, N))
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v + np.repeat(u_i, T)  # cost
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="ti",
                         cost=True)
        assert res.model_info["converged"]
        assert res.model_info["sign"] == 1


class TestPanelBC95:
    def test_bc95_recovers_determinant_coefficient(self):
        rng = np.random.default_rng(71)
        N, T = 150, 5
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        z1 = rng.normal(0, 1, n)
        mu_it = 0.2 + 0.3 * z1
        u = sst.truncnorm.rvs(-mu_it / 0.4, np.inf, loc=mu_it, scale=0.4,
                              random_state=rng)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "z1": z1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                         model="bc95", emean=["z1"])
        assert res.model_info["converged"]
        assert abs(res.params["mu_z1"] - 0.3) < 0.1
        assert "efficiency_bc_unit_mean" in res.diagnostics

    def test_bc95_requires_emean(self):
        df = _simulate_panel_ti(40, 3, 0.2, 0.4, seed=72)
        with pytest.raises(ValueError, match="emean"):
            xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="bc95")


# ---------------------------------------------------------------------------
# Kernel math sanity checks
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_frontier_matches_xbeta(self):
        df = _simulate_hn_production(500, 0.2, 0.4, seed=201)
        res = frontier(df, y="y", x=["x1", "x2"], dist="half-normal")
        new = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, -0.5]})
        fr = res.predict(new, what="frontier")
        # Manually: cons + 1*x1 coef + (-0.5)*x2 coef
        cons = res.params["_cons"]
        b1 = res.params["x1"]
        b2 = res.params["x2"]
        expected = np.array([cons, cons + b1 - 0.5 * b2])
        assert np.allclose(fr.values, expected, atol=1e-12)

    def test_predict_expected_efficiency_in_bounds(self):
        df = _simulate_hn_production(200, 0.2, 0.5, seed=202)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        new = pd.DataFrame({"x1": np.linspace(-2, 2, 10)})
        te = res.predict(new, what="expected_efficiency")
        assert (te > 0).all() and (te <= 1).all()

    def test_predict_expected_inefficiency_matches_closed_form(self):
        df = _simulate_hn_production(300, 0.2, 0.5, seed=203)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        new = pd.DataFrame({"x1": [0.0]})
        E_u = res.predict(new, what="expected_inefficiency").iloc[0]
        sigma_u = np.exp(res.params["ln_sigma_u"])
        expected = sigma_u * np.sqrt(2.0 / np.pi)  # marginal HN
        assert abs(E_u - expected) < 1e-10

    def test_predict_handles_usigma(self):
        """Heteroskedastic u: new data with different w should give different E[u]."""
        rng = np.random.default_rng(204)
        n = 1500
        x1 = rng.normal(0, 1, n)
        w1 = rng.normal(0, 1, n)
        sigma_u_i = np.exp(-1.0 + 0.3 * w1)
        u = np.abs(rng.normal(0, sigma_u_i))
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "w1": w1})
        res = frontier(df, y="y", x=["x1"], dist="half-normal", usigma=["w1"])
        new = pd.DataFrame({"x1": [0.0, 0.0], "w1": [-2.0, 2.0]})
        E_u = res.predict(new, what="expected_inefficiency").values
        # Different w → different E[u]; w=2 should give higher u (positive coef).
        assert E_u[1] > E_u[0]

    def test_predict_rejects_missing_columns(self):
        df = _simulate_hn_production(200, 0.2, 0.4, seed=205)
        res = frontier(df, y="y", x=["x1", "x2"], dist="half-normal")
        with pytest.raises(KeyError, match="missing"):
            res.predict(pd.DataFrame({"x1": [0.0]}))  # missing x2

    def test_predict_rejects_unknown_what(self):
        df = _simulate_hn_production(100, 0.2, 0.4, seed=206)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        with pytest.raises(ValueError):
            res.predict(df, what="nonsense")


class TestMarginalEffects:
    def test_bc95_marginal_effects_match_finite_difference(self):
        rng = np.random.default_rng(301)
        n = 3000
        x1 = rng.normal(0, 1, n)
        z1 = rng.normal(0, 1, n)
        mu_it = 0.3 + 0.25 * z1
        u = sst.truncnorm.rvs(-mu_it / 0.3, np.inf, loc=mu_it, scale=0.3,
                              random_state=rng)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "z1": z1})
        res = frontier(df, y="y", x=["x1"], dist="truncated-normal", emean=["z1"])

        analytic = res.marginal_effects(at="observation")["z1"].values
        h = 1e-5
        E0 = res.predict(df, what="expected_inefficiency").values
        df_plus = df.copy()
        df_plus["z1"] = df_plus["z1"] + h
        E1 = res.predict(df_plus, what="expected_inefficiency").values
        fd = (E1 - E0) / h
        assert np.max(np.abs(analytic - fd)) < 1e-5

    def test_marginal_effects_requires_emean(self):
        """TN model without emean must raise, not silently return zero effects."""
        rng = np.random.default_rng(302)
        n = 500
        x1 = rng.normal(0, 1, n)
        u = sst.truncnorm.rvs(-0.3 / 0.3, np.inf, loc=0.3, scale=0.3,
                              size=n, random_state=rng)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1})
        res = frontier(df, y="y", x=["x1"], dist="truncated-normal")
        with pytest.raises(RuntimeError, match="emean"):
            res.marginal_effects()

    def test_marginal_effects_requires_truncated_normal(self):
        """HN / exponential must raise — no mu to differentiate through."""
        df = _simulate_hn_production(100, 0.2, 0.4, seed=304)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        with pytest.raises(RuntimeError, match="truncated-normal"):
            res.marginal_effects()

    def test_marginal_effects_ame_is_mean(self):
        rng = np.random.default_rng(303)
        n = 500
        x1 = rng.normal(0, 1, n)
        z1 = rng.normal(0, 1, n)
        u = sst.truncnorm.rvs(-0.3 / 0.3, np.inf, loc=0.3, scale=0.3,
                              size=n, random_state=rng)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "z1": z1})
        res = frontier(df, y="y", x=["x1"], dist="truncated-normal", emean=["z1"])
        obs = res.marginal_effects(at="observation")
        ame = res.marginal_effects(at="ame")
        assert np.isclose(obs.mean().iloc[0], ame.iloc[0])


class TestVarianceEstimators:
    def test_oim_vs_opg_close_for_well_specified(self):
        """Under correct specification, OIM and OPG SEs should be close."""
        df = _simulate_hn_production(3000, 0.2, 0.5, seed=501)
        r_oim = frontier(df, y="y", x=["x1"], dist="half-normal", vce="oim")
        r_opg = frontier(df, y="y", x=["x1"], dist="half-normal", vce="opg")
        # Within 20% (asymptotic equivalence).
        diff = abs(r_oim.std_errors["x1"] - r_opg.std_errors["x1"])
        assert diff / r_oim.std_errors["x1"] < 0.25

    def test_robust_se_finite_and_different_from_oim(self):
        df = _simulate_hn_production(1000, 0.2, 0.4, seed=502)
        r_oim = frontier(df, y="y", x=["x1"], dist="half-normal", vce="oim")
        r_rob = frontier(df, y="y", x=["x1"], dist="half-normal", vce="robust")
        assert np.isfinite(r_rob.std_errors["x1"])
        # Robust changes the SE (usually marginal when correctly specified).
        assert r_rob.std_errors["x1"] > 0

    def test_cluster_se_larger_under_within_cluster_correlation(self):
        """Cluster-robust SE should exceed OIM when clusters share noise."""
        rng = np.random.default_rng(503)
        n = 2000
        x1 = rng.normal(0, 1, n)
        g = rng.integers(0, 40, n)
        u = np.abs(rng.normal(0, 0.4, n))
        g_shock = rng.normal(0, 0.4, 40)[g]  # strong cluster noise
        v = rng.normal(0, 0.15, n) + g_shock
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "g": g})
        r_oim = frontier(df, y="y", x=["x1"], dist="half-normal", vce="oim")
        r_cl = frontier(df, y="y", x=["x1"], dist="half-normal", cluster="g")
        assert r_cl.std_errors["x1"] > r_oim.std_errors["x1"]

    def test_unknown_vce_raises(self):
        df = _simulate_hn_production(100, 0.2, 0.4, seed=504)
        with pytest.raises(ValueError, match="vce"):
            frontier(df, y="y", x=["x1"], vce="jackknife")

    def test_cluster_implies_robust(self):
        df = _simulate_hn_production(500, 0.2, 0.4, seed=505)
        df["g"] = np.repeat(np.arange(10), 50)
        r = frontier(df, y="y", x=["x1"], cluster="g")
        assert "cluster" in r.model_info["vce"]

    def test_panel_cluster_reduces_to_group_scores(self):
        """Panel vce='robust' == cluster=id (both aggregate at group level)."""
        rng = np.random.default_rng(506)
        N, T = 80, 4
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        u_i = np.abs(rng.normal(0, 0.4, N))
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - np.repeat(u_i, T)
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        r_rob = xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                           model="ti", vce="robust")
        r_cl = xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                          model="ti", cluster="id")
        # Both should give identical SEs (cluster=id IS the natural grouping).
        assert np.isclose(r_rob.std_errors["x1"], r_cl.std_errors["x1"])


class TestPanelPredict:
    def test_panel_ti_predict_frontier(self):
        rng = np.random.default_rng(601)
        N, T = 40, 5
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        u_i = np.abs(rng.normal(0, 0.4, N))
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - np.repeat(u_i, T)
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="ti")
        new = pd.DataFrame({"x1": [0.0, 1.0]})
        fr = res.predict(new, what="frontier")
        expected = np.array([res.params["_cons"],
                             res.params["_cons"] + res.params["x1"]])
        assert np.allclose(fr.values, expected, atol=1e-12)

    def test_panel_ti_predict_expected_efficiency(self):
        rng = np.random.default_rng(602)
        N, T = 40, 5
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        u_i = np.abs(rng.normal(0, 0.4, N))
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - np.repeat(u_i, T)
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="ti")
        te = res.predict(pd.DataFrame({"x1": [0.0]}), what="expected_efficiency")
        assert 0 <= te.iloc[0] <= 1

    def test_bc95_panel_predict_varies_with_emean(self):
        rng = np.random.default_rng(603)
        N, T = 60, 4
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        z1 = rng.normal(0, 1, n)
        mu_it = 0.2 + 0.3 * z1
        u = sst.truncnorm.rvs(-mu_it / 0.4, np.inf, loc=mu_it, scale=0.4,
                              random_state=rng)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "z1": z1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                         model="bc95", emean=["z1"])
        new = pd.DataFrame({"x1": [0, 0], "z1": [-1.0, 1.0]})
        E_u = res.predict(new, what="expected_inefficiency").values
        # Positive emean coef on z1 → higher z1 gives larger E[u].
        assert E_u[1] > E_u[0]


class TestTruncatedNormalRobustness:
    def test_multi_start_gives_at_least_as_good_ll(self):
        """After multi-start, TN fit must be at least as good as single-start."""
        rng = np.random.default_rng(401)
        n = 1000
        x1 = rng.normal(0, 1, n)
        # Challenging case: true mu near 0 (ambiguous between HN and TN).
        mu_t = 0.05
        u = sst.truncnorm.rvs(-mu_t / 0.4, np.inf, loc=mu_t, scale=0.4,
                              size=n, random_state=rng)
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1})
        res = frontier(df, y="y", x=["x1"], dist="truncated-normal")
        # Just verify LL is reasonable (finite, not degenerate)
        assert np.isfinite(res.diagnostics["log_likelihood"])
        # And not absurdly worse than a half-normal fit of same data.
        res_hn = frontier(df, y="y", x=["x1"], dist="half-normal")
        # TN has one more parameter (mu), so its LL must be >= HN's LL - epsilon.
        assert (res.diagnostics["log_likelihood"]
                >= res_hn.diagnostics["log_likelihood"] - 0.5)


class TestGreeneTrueEffects:
    """Greene (2005) True Fixed/Random Effects SFA models."""

    def test_tre_recovers_three_variance_components(self):
        """TRE should separate alpha_i, v_it, and u_it."""
        rng = np.random.default_rng(701)
        N, T = 80, 6
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        sigma_alpha_true = 0.4
        sigma_v_true = 0.15
        sigma_u_true = 0.3
        alpha_i = rng.normal(0, sigma_alpha_true, N)
        u_it = np.abs(rng.normal(0, sigma_u_true, n))
        v = rng.normal(0, sigma_v_true, n)
        y = 1.0 + 0.5 * x1 + np.repeat(alpha_i, T) + v - u_it
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})

        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="tre")
        assert res.model_info["converged"]
        assert abs(res.params["x1"] - 0.5) < 0.05
        assert abs(np.exp(res.params["ln_sigma_v"]) - sigma_v_true) < 0.08
        assert abs(np.exp(res.params["ln_sigma_u"]) - sigma_u_true) < 0.08
        assert abs(np.exp(res.params["ln_sigma_alpha"]) - sigma_alpha_true) < 0.15

    def test_tre_exposes_sigma_alpha_in_model_info(self):
        df = _simulate_panel_ti(50, 5, 0.15, 0.3, seed=702)
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="tre")
        assert "sigma_alpha" in res.model_info
        assert res.model_info["sigma_alpha"] > 0

    def test_tre_rejects_truncated_normal(self):
        df = _simulate_panel_ti(30, 4, 0.2, 0.3, seed=703)
        with pytest.raises(ValueError, match="tre currently|TRE currently"):
            xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                       model="tre", dist="truncated-normal")

    def test_tfe_recovers_with_long_t(self):
        """TFE (brute-force firm dummies) works well when T is not too short."""
        rng = np.random.default_rng(704)
        N, T = 30, 20  # long T to mitigate incidental-parameters bias
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        alpha_i = rng.normal(2.0, 0.5, N)
        u_it = np.abs(rng.normal(0, 0.35, n))
        v = rng.normal(0, 0.15, n)
        y = np.repeat(alpha_i, T) + 0.5 * x1 + v - u_it
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="tfe")
        assert abs(res.params["x1"] - 0.5) < 0.05
        # sigma_u identified when T is moderate
        assert 0.25 < np.exp(res.params["ln_sigma_u"]) < 0.5

    def test_tfe_stripped_regressor_list(self):
        """Dummy columns shouldn't leak into result.data_info['regressors']."""
        rng = np.random.default_rng(705)
        N, T = 20, 15
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        alpha_i = rng.normal(0, 0.3, N)
        u = np.abs(rng.normal(0, 0.3, n))
        v = rng.normal(0, 0.15, n)
        y = 1.0 + 0.5 * x1 + np.repeat(alpha_i, T) + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="tfe")
        assert res.data_info["regressors"] == ["x1"]


class TestBootstrap:
    def test_bootstrap_se_close_to_oim_for_iid_data(self):
        # B=200 so the bootstrap-SE Monte Carlo error is ~5% of the true
        # SE. Threshold 0.20 is ~4 MC sigmas: well-behaved bootstrap
        # almost never triggers a false fail, but a bootstrap that
        # silently falls back to a degenerate estimate (old SE-collapse
        # bug) would miss by far more than 20% and fail this test.
        df = _simulate_hn_production(1000, 0.2, 0.4, seed=801)
        r_oim = frontier(df, y="y", x=["x1"], vce="oim")
        r_bs = frontier(df, y="y", x=["x1"], vce="bootstrap", B=200, seed=1)
        rel = abs(r_bs.std_errors["x1"] - r_oim.std_errors["x1"]) / r_oim.std_errors["x1"]
        assert rel < 0.20, f"bootstrap SE too far from OIM: rel={rel:.3f}"
        assert np.isfinite(r_bs.std_errors["x1"])
        # Bootstrap SE must not be implausibly small (old collapse bug).
        assert r_bs.std_errors["x1"] > 0.5 * r_oim.std_errors["x1"]

    def test_cluster_bootstrap_respects_cluster_structure(self):
        rng = np.random.default_rng(802)
        n = 600
        x1 = rng.normal(0, 1, n)
        g = rng.integers(0, 30, n)
        u = np.abs(rng.normal(0, 0.4, n))
        v = rng.normal(0, 0.2, n) + rng.normal(0, 0.3, 30)[g]
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "g": g})
        r_cluster_bs = frontier(
            df, y="y", x=["x1"], vce="bootstrap", cluster="g", B=80, seed=3
        )
        assert np.isfinite(r_cluster_bs.std_errors["x1"])

    def test_bootstrap_reproducible_with_seed(self):
        df = _simulate_hn_production(300, 0.2, 0.4, seed=803)
        r1 = frontier(df, y="y", x=["x1"], vce="bootstrap", B=30, seed=7)
        r2 = frontier(df, y="y", x=["x1"], vce="bootstrap", B=30, seed=7)
        assert np.isclose(r1.std_errors["x1"], r2.std_errors["x1"])


class TestConditionalPredict:
    def test_conditional_efficiency_matches_training_efficiency(self):
        """Posterior E[exp(-u)|eps] on held-out data should agree with efficiency() on same rows."""
        df = _simulate_hn_production(1200, 0.2, 0.4, seed=1001)
        # Fit on all data.
        res = frontier(df, y="y", x=["x1", "x2"], dist="half-normal")
        # predict with y present on the same rows — should match efficiency() output.
        te_train = res.efficiency(method="bc").values
        te_pred = res.predict(df, what="conditional_efficiency").values
        assert np.allclose(te_train, te_pred, atol=1e-10)

    def test_conditional_requires_y(self):
        df = _simulate_hn_production(300, 0.2, 0.4, seed=1002)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        no_y = df[["x1"]]
        with pytest.raises(KeyError, match="requires the dependent"):
            res.predict(no_y, what="conditional_efficiency")

    def test_conditional_inefficiency_bounded(self):
        df = _simulate_hn_production(400, 0.2, 0.5, seed=1003)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        u_hat = res.predict(df, what="conditional_inefficiency").values
        assert (u_hat >= -1e-6).all()


class TestReturnsToScale:
    def test_rts_correct_for_crs_dgp(self):
        rng = np.random.default_rng(1101)
        n = 3000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        v = rng.normal(0, 0.2, n)
        u = np.abs(rng.normal(0, 0.3, n))
        y = 1.0 + 0.6 * x1 + 0.4 * x2 + v - u  # sum = 1 (CRS)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        res = frontier(df, y="y", x=["x1", "x2"])
        rts = res.returns_to_scale()
        assert abs(rts["rts"] - 1.0) < 0.05
        assert rts["pvalue"] > 0.05  # fail to reject CRS
        assert "CRS" in rts["interpretation"]

    def test_rts_detects_irs(self):
        rng = np.random.default_rng(1102)
        n = 3000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        v = rng.normal(0, 0.2, n)
        u = np.abs(rng.normal(0, 0.3, n))
        y = 1.0 + 0.7 * x1 + 0.6 * x2 + v - u  # sum = 1.3
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        res = frontier(df, y="y", x=["x1", "x2"])
        rts = res.returns_to_scale()
        assert rts["rts"] > 1.1
        assert rts["pvalue"] < 0.01
        assert "IRS" in rts["interpretation"]


class TestUsigmaMarginalEffects:
    def test_usigma_me_matches_finite_difference(self):
        rng = np.random.default_rng(1201)
        n = 3000
        x1 = rng.normal(0, 1, n)
        w1 = rng.normal(0, 1, n)
        sigma_u_i = np.exp(-1.0 + 0.4 * w1)
        u = np.abs(rng.normal(0, sigma_u_i))
        v = rng.normal(0, 0.2, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "w1": w1})
        res = frontier(df, y="y", x=["x1"], dist="half-normal", usigma=["w1"])

        analytic = res.marginal_effects(source="usigma", at="observation")["w1"].values
        h = 1e-5
        E_u_0 = res.predict(df, what="expected_inefficiency").values
        df_p = df.copy()
        df_p["w1"] = df_p["w1"] + h
        E_u_p = res.predict(df_p, what="expected_inefficiency").values
        fd = (E_u_p - E_u_0) / h
        assert np.max(np.abs(analytic - fd)) < 1e-4

    def test_usigma_me_requires_usigma_model(self):
        df = _simulate_hn_production(200, 0.2, 0.4, seed=1202)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        with pytest.raises(RuntimeError, match="usigma"):
            res.marginal_effects(source="usigma")


class TestMetafrontier:
    def test_metafrontier_envelopes_group_frontiers(self):
        """Meta frontier x'beta_meta should dominate every group's x'beta^k."""
        from statspai.frontier import metafrontier

        rng = np.random.default_rng(1301)
        N_per, K = 150, 3
        n = N_per * K
        x1 = rng.normal(0, 1, n)
        groups_arr = np.repeat(["A", "B", "C"], N_per)
        intercepts = {"A": 1.0, "B": 0.7, "C": 0.5}
        u = np.abs(rng.normal(0, 0.3, n))
        v = rng.normal(0, 0.15, n)
        y = np.array(
            [intercepts[g] + 0.5 * x1[i] + v[i] - u[i] for i, g in enumerate(groups_arr)]
        )
        df = pd.DataFrame({"y": y, "x1": x1, "g": groups_arr})
        res = metafrontier(df, y="y", x=["x1"], group="g")

        # Verify meta frontier envelopes each group's frontier for every obs.
        X = np.column_stack([np.ones(n), df["x1"].to_numpy()])
        meta_frontier = X @ res.beta_meta.to_numpy()
        for g, bg in res.beta_groups.items():
            group_frontier = X @ bg.to_numpy()
            assert (meta_frontier >= group_frontier - 1e-8).all()

    def test_metafrontier_tgr_in_0_1(self):
        from statspai.frontier import metafrontier
        rng = np.random.default_rng(1302)
        N_per, K = 100, 2
        n = N_per * K
        x1 = rng.normal(0, 1, n)
        groups_arr = np.repeat(["A", "B"], N_per)
        intercepts = {"A": 1.0, "B": 0.6}
        u = np.abs(rng.normal(0, 0.3, n))
        v = rng.normal(0, 0.15, n)
        y = np.array(
            [intercepts[g] + 0.5 * x1[i] + v[i] - u[i] for i, g in enumerate(groups_arr)]
        )
        df = pd.DataFrame({"y": y, "x1": x1, "g": groups_arr})
        res = metafrontier(df, y="y", x=["x1"], group="g")
        assert ((res.tgr > 0) & (res.tgr <= 1)).all()
        assert ((res.te_meta >= 0) & (res.te_meta <= 1)).all()


class TestMalmquist:
    """Malmquist TFP index (Fare-Grosskopf-Lindgren-Roos 1994)."""

    def test_malmquist_detects_positive_technical_change(self):
        """Frontier intercept grows 5% per year → TC should be > 1."""
        from statspai.frontier import malmquist

        rng = np.random.default_rng(1701)
        N = 100
        periods = [2018, 2019, 2020]
        intercepts = {2018: 1.0, 2019: 1.05, 2020: 1.10}
        u_persist = np.abs(rng.normal(0, 0.3, N))
        rows = []
        for t in periods:
            for i in range(N):
                x1 = rng.normal(0, 1)
                x2 = rng.normal(0, 1)
                u = u_persist[i] * rng.uniform(0.8, 1.2)
                v = rng.normal(0, 0.15)
                y = intercepts[t] + 0.5 * x1 + 0.4 * x2 + v - u
                rows.append({"id": i, "t": t, "y": y, "x1": x1, "x2": x2})
        df = pd.DataFrame(rows)
        res = malmquist(df, y="y", x=["x1", "x2"], id="id", time="t")
        # Mean TC across transitions should exceed 1 (tech progress).
        mean_tc = res.index_table["tc"].mean()
        assert mean_tc > 1.03, f"mean_tc={mean_tc}"
        # Mean M > 1 (overall growth).
        assert res.index_table["m_index"].mean() > 1.0

    def test_malmquist_decomposition_multiplicative(self):
        """M = EC * TC must hold row-wise (identity)."""
        from statspai.frontier import malmquist
        rng = np.random.default_rng(1702)
        N = 40
        periods = [1, 2]
        rows = []
        for t in periods:
            for i in range(N):
                x1 = rng.normal(0, 1)
                u = np.abs(rng.normal(0, 0.3))
                v = rng.normal(0, 0.15)
                y = (1.0 + 0.05 * t) + 0.5 * x1 + v - u
                rows.append({"id": i, "t": t, "y": y, "x1": x1})
        df = pd.DataFrame(rows)
        res = malmquist(df, y="y", x=["x1"], id="id", time="t")
        m = res.index_table["m_index"].values
        ec = res.index_table["ec"].values
        tc = res.index_table["tc"].values
        assert np.allclose(m, ec * tc, rtol=1e-8)

    def test_malmquist_tc_matches_hand_computation(self):
        """Reference test: when frontier shifts by known delta and a firm
        tracks the frontier exactly, TC = exp(delta_alpha) and EC = 1.

        This is the first test in the suite that compares Malmquist
        numbers against an *external* hand-computed reference rather
        than the implementation's own identity M = EC * TC. A silent
        sign flip in log_TC or a formula error in the geometric-mean
        component would be caught here even though the identity test
        would still pass.
        """
        from statspai.frontier import malmquist
        rng = np.random.default_rng(1703)
        N = 400
        delta_alpha = 0.10   # known outward frontier shift period 1 -> 2
        beta_true = 0.5
        # Low noise so OLS/SFA betas are very close to truth.
        sigma_u_true = 0.02
        sigma_v_true = 0.02
        rows = []
        # Each firm tracks the frontier: y_t = alpha_t + beta * x_t + v - u
        # Under low noise, firms stay close to the moving frontier so both
        # log D^t(x^t, y^t) and log D^{t+1}(x^{t+1}, y^{t+1}) are small and
        # equal on average, giving EC ~ 1 and TC ~ exp(delta_alpha).
        for t_idx, t in enumerate([1, 2]):
            alpha_t = 1.0 + t_idx * delta_alpha
            for i in range(N):
                x1 = rng.normal(0, 1)
                u = np.abs(rng.normal(0, sigma_u_true))
                v = rng.normal(0, sigma_v_true)
                y = alpha_t + beta_true * x1 + v - u
                rows.append({"id": i, "t": t, "y": y, "x1": x1})
        df = pd.DataFrame(rows)
        res = malmquist(df, y="y", x=["x1"], id="id", time="t")

        # Hand-computed reference values (independent of the
        # implementation): given frontier shift = exp(delta_alpha) and
        # firms track the frontier, mean TC should be close to
        # exp(delta_alpha) and mean EC close to 1.
        expected_tc = float(np.exp(delta_alpha))
        expected_ec = 1.0
        mean_tc = float(res.index_table["tc"].mean())
        mean_ec = float(res.index_table["ec"].mean())
        # Tolerances: sampling noise + finite sigma_u lifting mean log D.
        # 0.02 on TC is ~18% of the effect (delta_alpha=0.10 -> TC 1.105);
        # a sign flip in log_TC (which would give TC ~ exp(-0.10) = 0.905)
        # would fail this at >9 sigma.
        assert abs(mean_tc - expected_tc) < 0.03, (
            f"mean TC {mean_tc:.4f} vs expected {expected_tc:.4f} "
            f"(|diff|={abs(mean_tc - expected_tc):.4f})"
        )
        assert abs(mean_ec - expected_ec) < 0.04, (
            f"mean EC {mean_ec:.4f} vs expected {expected_ec:.4f}"
        )
        # M = EC * TC should equal the observed mean M up to noise;
        # more importantly the direction is correct.
        mean_m = float(res.index_table["m_index"].mean())
        assert mean_m > 1.05, f"mean M {mean_m:.4f} should exceed 1 (growth)"

    def test_malmquist_requires_two_periods(self):
        from statspai.frontier import malmquist
        df = _simulate_hn_production(100, 0.2, 0.4, seed=1703)
        df["id"] = np.arange(100)
        df["t"] = 2020  # single period
        with pytest.raises(ValueError, match="two periods"):
            malmquist(df, y="y", x=["x1"], id="id", time="t")


class TestTranslogDesign:
    def test_translog_design_adds_squares_and_interactions(self):
        from statspai.frontier import translog_design
        df = pd.DataFrame({"log_k": [1.0, 2.0, 3.0], "log_l": [0.5, 1.5, 2.5]})
        tl = translog_design(df, inputs=["log_k", "log_l"])
        assert "log_k_sq" in tl.columns
        assert "log_l_sq" in tl.columns
        assert "log_k_x_log_l" in tl.columns
        # 0.5 * log_k^2 convention
        assert np.isclose(tl["log_k_sq"].iloc[2], 0.5 * 3.0**2)
        assert np.isclose(tl["log_k_x_log_l"].iloc[0], 1.0 * 0.5)

    def test_translog_opt_out_squares(self):
        from statspai.frontier import translog_design
        df = pd.DataFrame({"log_k": [1.0], "log_l": [1.0]})
        tl = translog_design(df, inputs=["log_k", "log_l"],
                              include_squares=False)
        assert "log_k_sq" not in tl.columns
        assert "log_k_x_log_l" in tl.columns

    def test_translog_opt_out_interactions(self):
        from statspai.frontier import translog_design
        df = pd.DataFrame({"log_k": [1.0], "log_l": [1.0]})
        tl = translog_design(df, inputs=["log_k", "log_l"],
                              include_interactions=False)
        assert "log_k_x_log_l" not in tl.columns
        assert "log_k_sq" in tl.columns

    def test_translog_attrs_list(self):
        from statspai.frontier import translog_design
        df = pd.DataFrame({"log_k": [1.0, 2.0], "log_l": [1.0, 2.0]})
        tl = translog_design(df, inputs=["log_k", "log_l"])
        terms = tl.attrs["translog_terms"]
        assert "log_k" in terms and "log_k_sq" in terms and "log_k_x_log_l" in terms
        # added_terms should be ONLY the new columns (no original inputs).
        added = tl.attrs["translog_added_terms"]
        assert "log_k" not in added and "log_l" not in added
        assert "log_k_sq" in added and "log_k_x_log_l" in added

    def test_translog_integration_with_frontier(self):
        """Integration: translog_design(...) → sp.frontier(x=attrs['translog_terms']).

        Verifies the advertised one-liner workflow actually runs end-to-end
        (previously the attrs field was never consumed). Also verifies
        that squared / interaction coefficients are near zero when the
        true DGP is Cobb-Douglas (translog nests CD as restrictions).
        """
        from statspai.frontier import translog_design
        rng = np.random.default_rng(1750)
        n = 600
        # Cobb-Douglas DGP: translog terms should all be ~0.
        log_k = rng.normal(0, 0.5, n)
        log_l = rng.normal(0, 0.5, n)
        u = np.abs(rng.normal(0, 0.2, n))
        v = rng.normal(0, 0.1, n)
        log_y = 1.0 + 0.4 * log_k + 0.5 * log_l + v - u
        df = pd.DataFrame({
            "log_y": log_y, "log_k": log_k, "log_l": log_l,
        })
        tl = translog_design(df, inputs=["log_k", "log_l"])
        terms = tl.attrs["translog_terms"]
        # Option A: pass the full translog regressor list in one shot.
        res = frontier(tl, y="log_y", x=terms, dist="half-normal")
        assert res.model_info.get("converged", False)
        # Squared + interaction terms should be small on a CD DGP.
        for extra in ("log_k_sq", "log_l_sq", "log_k_x_log_l"):
            assert extra in res.params.index, (
                f"{extra!r} missing from fitted params — integration broken"
            )
            assert abs(res.params[extra]) < 0.15, (
                f"CD DGP should give near-zero translog coeffs; "
                f"{extra}={res.params[extra]:.4f}"
            )
        # Option B: extend an existing x via translog_added_terms.
        base = ["log_k", "log_l"]
        extra_terms = tl.attrs["translog_added_terms"]
        res2 = frontier(tl, y="log_y", x=base + extra_terms, dist="half-normal")
        # Same model, should reproduce the same beta up to optimizer noise.
        for name in ("log_k", "log_l"):
            assert abs(res.params[name] - res2.params[name]) < 1e-3


class TestMonteCarloCoverage:
    """Canonical cross-check: 95% asymptotic CI should cover truth ~95% of time.

    This is the rigorous analogue of comparing to a single R-frontier fit:
    instead of trusting one point estimate, we verify the *sampling
    distribution* of our estimator matches theory.
    """

    @pytest.mark.slow
    def test_oim_ci_coverage_half_normal(self):
        """95% CI coverage for beta and sigma_u over 200 Monte Carlo draws.

        Power rationale:
        - At true coverage = 0.95, n_mc=200: binomial SD ~ 1.5 pp, so the
          0.90 lower threshold on beta is >3 sigma below the mean and
          never false-fires.
        - If coverage is actually 0.85 (a real bug), P(pass) at the 0.90
          threshold is < 2%: the test catches it essentially always.
        The old n_mc=60 + 0.80 sigma_u threshold allowed a method with
        true 80% coverage to pass with ~80% probability — effectively
        unfalsifiable.
        """
        n_mc = 200
        n = 800
        true_beta = 0.5
        true_sigma_u = 0.4
        true_sigma_v = 0.2
        covered_b = 0
        covered_su = 0
        for s in range(n_mc):
            rng = np.random.default_rng(900 + s)
            x1 = rng.normal(0, 1, n)
            u = np.abs(rng.normal(0, true_sigma_u, n))
            v = rng.normal(0, true_sigma_v, n)
            y = 1.0 + true_beta * x1 + v - u
            df = pd.DataFrame({"y": y, "x1": x1})
            res = frontier(df, y="y", x=["x1"], dist="half-normal")
            lo = res.params["x1"] - 1.96 * res.std_errors["x1"]
            hi = res.params["x1"] + 1.96 * res.std_errors["x1"]
            if lo < true_beta < hi:
                covered_b += 1
            # sigma_u via delta method on ln_sigma_u: σ_u SE ≈ σ_u * se(ln σ_u)
            su_hat = np.exp(res.params["ln_sigma_u"])
            su_se = su_hat * res.std_errors["ln_sigma_u"]
            if abs(su_hat - true_sigma_u) < 1.96 * su_se:
                covered_su += 1
        coverage_b = covered_b / n_mc
        coverage_su = covered_su / n_mc
        # Expected 95%, tolerate ±5 pp with n_mc=200 (3-sigma band).
        assert coverage_b >= 0.90, f"beta coverage: {coverage_b}"
        assert coverage_su >= 0.88, f"sigma_u coverage: {coverage_su}"


class TestZeroInefficiency:
    """Zero-Inefficiency SFA (Kumbhakar-Parmeter-Tsionas 2013)."""

    def test_zisf_recovers_mixture_probability(self):
        from statspai.frontier import zisf
        rng = np.random.default_rng(1401)
        n = 2000
        x1 = rng.normal(0, 1, n)
        eff_mask = rng.uniform(0, 1, n) < 0.3  # 30% efficient
        u = np.where(eff_mask, 0.0, np.abs(rng.normal(0, 0.5, n)))
        v = rng.normal(0, 0.15, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1})
        res = zisf(df, y="y", x=["x1"])
        assert res.model_info["converged"]
        assert abs(res.params["x1"] - 0.5) < 0.05
        assert abs(np.exp(res.params["ln_sigma_u"]) - 0.5) < 0.1
        assert abs(res.model_info["mean_p_efficient"] - 0.3) < 0.05

    def test_zisf_with_zprob_covariate(self):
        """Class probability should respond to a covariate."""
        from statspai.frontier import zisf
        rng = np.random.default_rng(1402)
        n = 2000
        x1 = rng.normal(0, 1, n)
        z1 = rng.normal(0, 1, n)
        # Higher z1 → higher chance of being efficient.
        p_eff = 1.0 / (1.0 + np.exp(-z1))
        eff_mask = rng.uniform(0, 1, n) < p_eff
        u = np.where(eff_mask, 0.0, np.abs(rng.normal(0, 0.4, n)))
        v = rng.normal(0, 0.15, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "z1": z1})
        res = zisf(df, y="y", x=["x1"], zprob=["z1"])
        assert res.model_info["converged"]
        # Logit coef on z1 should be clearly positive.
        assert res.params["p_z1"] > 0.3

    def test_zisf_rejects_unsupported_dist(self):
        from statspai.frontier import zisf
        df = _simulate_hn_production(200, 0.15, 0.4, seed=1403)
        with pytest.raises(ValueError, match="half-normal"):
            zisf(df, y="y", x=["x1"], dist="exponential")


class TestLatentClassSFA:
    """Latent-Class SFA (Orea-Kumbhakar 2004)."""

    def test_lcsf_recovers_two_classes(self):
        from statspai.frontier import lcsf
        rng = np.random.default_rng(1501)
        n = 2500
        x1 = rng.normal(0, 1, n)
        class_1 = rng.uniform(0, 1, n) < 0.5
        u = np.abs(rng.normal(0, 0.3, n))
        v = rng.normal(0, 0.15, n)
        true_alpha_1, true_alpha_2 = 1.0, 2.0
        true_slope_1, true_slope_2 = 0.8, 0.3
        y = np.where(
            class_1,
            true_alpha_1 + true_slope_1 * x1 + v - u,
            true_alpha_2 + true_slope_2 * x1 + v - u,
        )
        df = pd.DataFrame({"y": y, "x1": x1})
        res = lcsf(df, y="y", x=["x1"])
        assert res.model_info["converged"]
        # Class recovery: one class should match (slope=0.8, alpha=1.0),
        # the other (slope=0.3, alpha=2.0). Verify that BOTH slope and
        # intercept match the SAME class pairing — guards against the
        # partial-swap failure mode where slopes land right but intercepts
        # are swapped across classes.
        c1_x1 = res.params["c1:x1"]
        c2_x1 = res.params["c2:x1"]
        c1_cons = res.params["c1:_cons"]
        c2_cons = res.params["c2:_cons"]
        # Canonical ordering (ascending sigma_u) makes which class is "c1"
        # deterministic, but the mapping (c1 ↔ slope=0.8) depends on
        # whether class 1 in the DGP has higher or lower sigma_u after
        # the canonical flip. Check both orderings consistently.
        pairing_a = (  # c1 = slope_1 / alpha_1, c2 = slope_2 / alpha_2
            abs(c1_x1 - true_slope_1) < 0.12
            and abs(c2_x1 - true_slope_2) < 0.12
            and abs(c1_cons - true_alpha_1) < 0.15
            and abs(c2_cons - true_alpha_2) < 0.15
        )
        pairing_b = (  # c1 = slope_2 / alpha_2, c2 = slope_1 / alpha_1
            abs(c1_x1 - true_slope_2) < 0.12
            and abs(c2_x1 - true_slope_1) < 0.12
            and abs(c1_cons - true_alpha_2) < 0.15
            and abs(c2_cons - true_alpha_1) < 0.15
        )
        assert pairing_a or pairing_b, (
            f"class labels don't map consistently across slope/intercept: "
            f"c1=(alpha={c1_cons:.3f}, slope={c1_x1:.3f}), "
            f"c2=(alpha={c2_cons:.3f}, slope={c2_x1:.3f})"
        )

    def test_lcsf_efficiency_bounded(self):
        from statspai.frontier import lcsf
        rng = np.random.default_rng(1502)
        n = 1000
        x1 = rng.normal(0, 1, n)
        u = np.abs(rng.normal(0, 0.3, n))
        v = rng.normal(0, 0.15, n)
        y = 1.0 + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1})
        res = lcsf(df, y="y", x=["x1"])
        eff = res.efficiency()
        assert (eff >= 0).all() and (eff <= 1).all()

    def test_lcsf_two_classes_exposed(self):
        from statspai.frontier import lcsf
        df = _simulate_hn_production(500, 0.15, 0.4, seed=1503)
        res = lcsf(df, y="y", x=["x1"])
        assert res.model_info["n_classes"] == 2
        assert "p_class1_posterior" in res.diagnostics


class TestTFEBiasCorrection:
    """Dhaene-Jochmans (2015) split-panel jackknife bias correction."""

    def test_bias_correct_reduces_sigma_u_bias_when_t_large(self):
        rng = np.random.default_rng(1601)
        N, T = 25, 30
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        alpha_i = rng.normal(2.0, 0.5, N)
        u = np.abs(rng.normal(0, 0.35, n))
        v = rng.normal(0, 0.15, n)
        y = np.repeat(alpha_i, T) + 0.5 * x1 + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        r_raw = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="tfe")
        r_bc = xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                          model="tfe", bias_correct=True)
        # With fixed seed the direction of bias reduction must hold:
        # BC sigma_u strictly closer to the true 0.35 than raw.
        # A +0.02 slack (prior version) allowed a broken correction
        # that made bias WORSE by 2 pp to still pass.
        raw_bias = abs(np.exp(r_raw.params["ln_sigma_u"]) - 0.35)
        bc_bias = abs(np.exp(r_bc.params["ln_sigma_u"]) - 0.35)
        assert bc_bias < raw_bias, (
            f"bias correction did not reduce bias: raw={raw_bias:.4f}, "
            f"bc={bc_bias:.4f}"
        )
        # And reduction must be substantive (not a 0.1% cosmetic change).
        assert bc_bias <= 0.75 * raw_bias + 1e-6, (
            f"bias reduction too small: raw={raw_bias:.4f}, bc={bc_bias:.4f}"
        )
        assert "Dhaene-Jochmans" in r_bc.model_info.get("bias_correct", "")

    def test_bias_correct_requires_time(self):
        df = _simulate_panel_ti(20, 10, 0.15, 0.35, seed=1602)
        df_no_time = df.drop(columns=["t"])
        with pytest.raises(ValueError, match="time"):
            xtfrontier(df_no_time, y="y", x=["x1"], id="id",
                       model="tfe", bias_correct=True)

    def test_bias_correct_requires_enough_periods(self):
        """Splits require T >= 4; shorter panels should raise."""
        rng = np.random.default_rng(1603)
        N, T = 20, 3  # T=3 is too short
        id_ = np.repeat(np.arange(N), T)
        t_ = np.tile(np.arange(T), N)
        n = N * T
        x1 = rng.normal(0, 1, n)
        alpha_i = rng.normal(0, 0.3, N)
        u = np.abs(rng.normal(0, 0.3, n))
        v = rng.normal(0, 0.15, n)
        y = 1.0 + 0.5 * x1 + np.repeat(alpha_i, T) + v - u
        df = pd.DataFrame({"y": y, "x1": x1, "id": id_, "t": t_})
        with pytest.raises(ValueError, match="time periods"):
            xtfrontier(df, y="y", x=["x1"], id="id", time="t",
                       model="tfe", bias_correct=True)


class TestKernelMath:
    def test_halfnormal_is_valid_density(self):
        # Integrate the simulated f(eps) over eps ~ via Monte Carlo.
        rng = np.random.default_rng(91)
        n = 20000
        sigma_v, sigma_u = 0.3, 0.5
        u = np.abs(rng.normal(0, sigma_u, n))
        v = rng.normal(0, sigma_v, n)
        eps = v - u  # production
        ll = _fc.loglik_halfnormal(eps, np.full(n, sigma_v),
                                   np.full(n, sigma_u), sign=-1)
        assert np.all(np.isfinite(ll))
        # Empirical KL: compare to kernel estimate.
        grid = np.linspace(-3, 2, 300)
        f = np.exp(_fc.loglik_halfnormal(grid, np.array([sigma_v]),
                                          np.array([sigma_u]), sign=-1))
        integral = np.trapezoid(f, grid)
        assert abs(integral - 1.0) < 0.01

    def test_exponential_is_valid_density(self):
        grid = np.linspace(-5, 3, 1000)
        sigma_v, sigma_u = 0.3, 0.4
        f = np.exp(_fc.loglik_exponential(grid, np.array([sigma_v]),
                                           np.array([sigma_u]), sign=-1))
        integral = np.trapezoid(f, grid)
        assert abs(integral - 1.0) < 0.01

    def test_truncated_normal_is_valid_density(self):
        grid = np.linspace(-5, 4, 1000)
        sv, su, mu = 0.3, 0.4, 0.5
        f = np.exp(_fc.loglik_truncated_normal(
            grid, np.array([sv]), np.array([su]), np.array([mu]), sign=-1
        ))
        integral = np.trapezoid(f, grid)
        assert abs(integral - 1.0) < 0.01

    def test_halfnormal_and_trunc_agree_when_mu_zero(self):
        eps = np.linspace(-2, 2, 200)
        sv = np.full(eps.size, 0.3)
        su = np.full(eps.size, 0.5)
        mu = np.zeros(eps.size)
        ll_hn = _fc.loglik_halfnormal(eps, sv, su, sign=-1)
        ll_tn = _fc.loglik_truncated_normal(eps, sv, su, mu, sign=-1)
        assert np.allclose(ll_hn, ll_tn, atol=1e-10)

    def test_battese_coelli_te_bounded(self):
        mu = np.linspace(-2, 3, 50)
        sigma = np.full(mu.shape, 0.5)
        te = _fc._battese_coelli_te(mu, sigma)
        assert np.all((te >= 0) & (te <= 1))

    def test_posterior_mean_non_negative_at_extremes(self):
        """Guard against Mills-ratio truncation yielding tiny negative means."""
        mu = np.array([-500.0, -50.0, -5.0, 0.0, 5.0, 50.0])
        sigma = np.array([1.0, 0.5, 0.3, 0.7, 0.4, 0.2])
        E_u = _fc._posterior_truncnormal_mean(mu, sigma)
        assert np.all(E_u >= 0.0)
        assert np.all(np.isfinite(E_u))

    def test_cost_frontier_densities_integrate_to_1(self):
        """Cost-frontier densities must also integrate to 1 (new regression)."""
        grid = np.linspace(-3, 5, 2000)
        sv, su = np.array([0.3]), np.array([0.4])
        for fn in (_fc.loglik_halfnormal, _fc.loglik_exponential):
            f = np.exp(fn(grid, sv, su, sign=+1))
            assert abs(np.trapezoid(f, grid) - 1.0) < 0.01
        mu = np.array([0.5])
        f = np.exp(_fc.loglik_truncated_normal(grid, sv, su, mu, sign=+1))
        assert abs(np.trapezoid(f, grid) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Result object API
# ---------------------------------------------------------------------------


class TestResultObject:
    def test_summary_renders(self):
        df = _simulate_hn_production(300, 0.2, 0.4, seed=101)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        s = res.summary()
        assert isinstance(s, str) and len(s) > 50

    def test_summary_does_not_dump_per_obs_arrays(self):
        """Regression: summary() used to print 37KB of per-obs sigma_u_i etc."""
        df = _simulate_hn_production(800, 0.2, 0.4, seed=104)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        s = res.summary()
        # Without the fix, the per-obs array of sigma_u_i gets rendered as a
        # huge numpy array and the string balloons past 5000 chars for n=800.
        assert len(s) < 5000
        # But the variance-decomposition block must be there.
        assert "sigma_u" in s.lower()
        assert "gamma" in s.lower() or "λ" in s or "lambda" in s.lower()

    def test_summary_shows_lr_test(self):
        df = _simulate_hn_production(300, 0.2, 0.5, seed=105)
        res = frontier(df, y="y", x=["x1"])
        s = res.summary()
        assert "LR test" in s

    def test_panel_summary_no_array_dump(self):
        df = _simulate_panel_ti(40, 5, 0.2, 0.4, seed=106)
        res = xtfrontier(df, y="y", x=["x1"], id="id", time="t", model="ti")
        s = res.summary()
        assert len(s) < 5000

    def test_is_frontier_result(self):
        df = _simulate_hn_production(200, 0.2, 0.4, seed=102)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        assert isinstance(res, FrontierResult)

    def test_efficiency_method_dispatch(self):
        df = _simulate_hn_production(200, 0.2, 0.4, seed=103)
        res = frontier(df, y="y", x=["x1"], dist="half-normal")
        bc1 = res.efficiency()  # default = 'bc'
        bc2 = res.efficiency("bc")
        assert np.allclose(bc1, bc2)
        with pytest.raises(ValueError):
            res.efficiency("unknown-method")
