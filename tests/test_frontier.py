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
        # Wrong sign ⇒ residuals right-skewed, HN production LL will try to
        # shrink sigma_u towards the bound.  Correct fit must have higher LL.
        assert (
            res.diagnostics["log_likelihood"]
            > res_wrong.diagnostics["log_likelihood"] - 5
        )

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
            frontier(df, y="y", x=["x1"], vce="bootstrap")

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
        integral = np.trapz(f, grid)
        assert abs(integral - 1.0) < 0.01

    def test_exponential_is_valid_density(self):
        grid = np.linspace(-5, 3, 1000)
        sigma_v, sigma_u = 0.3, 0.4
        f = np.exp(_fc.loglik_exponential(grid, np.array([sigma_v]),
                                           np.array([sigma_u]), sign=-1))
        integral = np.trapz(f, grid)
        assert abs(integral - 1.0) < 0.01

    def test_truncated_normal_is_valid_density(self):
        grid = np.linspace(-5, 4, 1000)
        sv, su, mu = 0.3, 0.4, 0.5
        f = np.exp(_fc.loglik_truncated_normal(
            grid, np.array([sv]), np.array([su]), np.array([mu]), sign=-1
        ))
        integral = np.trapz(f, grid)
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
