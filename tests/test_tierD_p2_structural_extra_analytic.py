"""Tier D P2 known-truth anchors — structural / production-function estimators.

Part of the P2 "Tier D analytic special-cases" campaign. These entry points
were graded ``weak`` (boolean / shape / not-None asserts only). Each test below
bakes a *known truth* into the data-generating process and checks recovery:

    sp.prod_fn          Cobb-Douglas elasticities (op/lp/acf/wrdg dispatch)
    sp.olley_pakes      Cobb-Douglas elasticities via investment proxy
    sp.wooldridge_prod  Cobb-Douglas elasticities via intermediate-input proxy
    sp.markup           De Loecker-Warzynski mu = theta_v / cost_share exactly
    sp.metafrontier     nested group frontiers: TGR=1 on the meta-frontier
    sp.te_rank          dominated firm ranks last, near-frontier firm ranks top

DGP design (production functions). The proxy-variable timing assumptions are
satisfied by drawing labour ``l`` *exogenously* (independent of the
productivity innovation) so it is identified separately from omega — this
avoids the Ackerberg-Caves-Frazer collinearity that biases OP/LP when labour
responds to contemporaneous productivity. The investment proxy ``i`` and the
materials proxy ``m`` are both strictly monotone in (omega, k) as the
inversions require. Production-function estimators are noisy in finite
samples, so the labour tolerance is abs=0.05 and capital abs=0.10 (capital is
the harder, smaller-variance state input); a recovered elasticity off by 0.5
on this N would be a bug, not noise.

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

BETA_L, BETA_K = 0.60, 0.35
RHO = 0.7


def _identified_panel(seed=0, n_firms=300, n_periods=15):
    """Cobb-Douglas panel satisfying the OP/LP/Wooldridge assumptions."""
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(n_firms):
        omega = rng.normal(0.0, 0.2 / np.sqrt(1 - RHO**2))
        k = rng.normal(0.0, 0.5)
        for t in range(n_periods):
            omega = RHO * omega + rng.normal(0.0, 0.2)
            ell = rng.normal(0.5, 0.4)  # exogenous labour
            m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)  # proxy (LP/wrdg)
            i = np.exp(0.5 + 0.6 * omega + 0.3 * k + rng.normal(0.0, 0.05))
            y = BETA_L * ell + BETA_K * k + omega + rng.normal(0.0, 0.10)
            rows.append(
                {"id": fid, "year": t, "y": y, "l": ell, "k": k, "m": m, "i": i}
            )
            k = 0.9 * k + 0.1 * np.log(i + 1e-6)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sp.prod_fn — unified dispatcher (op / lp / acf / wrdg)
# ---------------------------------------------------------------------------
class TestProdFnAnalytic:
    @pytest.mark.parametrize(
        "method,proxy",
        [("op", "i"), ("lp", "m"), ("acf", "m"), ("wrdg", "m")],
    )
    def test_recovers_cobb_douglas_elasticities(self, method, proxy):
        df = _identified_panel()
        res = sp.prod_fn(
            df,
            output="y",
            free="l",
            state="k",
            proxy=proxy,
            panel_id="id",
            time="year",
            method=method,
        )
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.05)
        assert res.coef["k"] == pytest.approx(BETA_K, abs=0.10)

    def test_dispatch_matches_direct_estimator(self):
        # prod_fn(method="op") must dispatch identically to sp.olley_pakes.
        df = _identified_panel(seed=3, n_firms=150, n_periods=12)
        a = sp.prod_fn(
            df,
            output="y",
            free="l",
            state="k",
            proxy="i",
            panel_id="id",
            time="year",
            method="op",
        )
        b = sp.olley_pakes(
            df,
            output="y",
            free="l",
            state="k",
            proxy="i",
            panel_id="id",
            time="year",
        )
        assert a.coef["l"] == pytest.approx(b.coef["l"], abs=1e-10)
        assert a.coef["k"] == pytest.approx(b.coef["k"], abs=1e-10)


# ---------------------------------------------------------------------------
# sp.olley_pakes — investment proxy
# ---------------------------------------------------------------------------
class TestOlleyPakesAnalytic:
    def test_recovers_cobb_douglas_elasticities(self):
        df = _identified_panel()
        res = sp.olley_pakes(
            df,
            output="y",
            free="l",
            state="k",
            proxy="i",
            panel_id="id",
            time="year",
        )
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.05)
        assert res.coef["k"] == pytest.approx(BETA_K, abs=0.10)

    def test_first_stage_explains_output(self):
        # Stage-1 polynomial in (l, k, i) should fit a high-signal DGP well.
        df = _identified_panel()
        res = sp.olley_pakes(
            df,
            output="y",
            free="l",
            state="k",
            proxy="i",
            panel_id="id",
            time="year",
        )
        assert res.diagnostics["stage1_r2"] > 0.5


# ---------------------------------------------------------------------------
# sp.wooldridge_prod — intermediate-input proxy
# ---------------------------------------------------------------------------
class TestWooldridgeProdAnalytic:
    def test_recovers_cobb_douglas_elasticities(self):
        df = _identified_panel()
        res = sp.wooldridge_prod(
            df,
            output="y",
            free="l",
            state="k",
            proxy="m",
            panel_id="id",
            time="year",
        )
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.05)
        assert res.coef["k"] == pytest.approx(BETA_K, abs=0.10)


# ---------------------------------------------------------------------------
# sp.markup — De Loecker-Warzynski mu = theta_v / cost_share
# ---------------------------------------------------------------------------
class TestMarkupAnalytic:
    """The markup is ``mu_it = theta_v * (PQ) / (P_v V)``. With labour as the
    flexible input, theta_v is the recovered elasticity beta_l, and the cost
    share ``(P_v V)/(PQ)`` is the inverse second factor. We construct log
    revenue and log input cost so that the eta-corrected cost share equals a
    KNOWN constant; the returned markup must then equal beta_l / cost_share
    for every observation, exactly.
    """

    def _fitted(self, seed=0):
        df = _identified_panel(seed=seed)
        return sp.levinsohn_petrin(
            df,
            output="y",
            free="l",
            state="k",
            proxy="m",
            panel_id="id",
            time="year",
        )

    def test_markup_equals_elasticity_over_known_cost_share(self):
        res = self._fitted()
        n = len(res.sample)
        eta = res.sample["eta"].to_numpy()
        # eta-corrected cost share = exp(log_cost - (log_rev - eta)).
        # Pin it to a known constant by solving for log_cost.
        target_share = 0.30
        log_rev = np.full(n, 5.0)
        log_cost = np.log(target_share) + (log_rev - eta)
        samp = res.sample.copy()
        samp["log_rev"] = log_rev
        samp["log_cost"] = log_cost
        res.sample = samp

        mu = sp.markup(
            res, revenue="log_rev", input_cost="log_cost", flexible_input="l"
        )
        expected = res.coef["l"] / target_share
        assert np.allclose(mu.to_numpy(), expected, atol=1e-9)
        # And the recovered theta_v (= beta_l) is itself the known truth.
        assert res.coef["l"] == pytest.approx(BETA_L, abs=0.05)

    def test_markup_scales_inversely_with_cost_share(self):
        # Halving the cost share doubles the markup (mu = theta_v / share).
        res = self._fitted(seed=1)
        n = len(res.sample)
        eta = res.sample["eta"].to_numpy()
        log_rev = np.full(n, 5.0)
        samp = res.sample.copy()
        samp["log_rev"] = log_rev
        samp["lc_a"] = np.log(0.40) + (log_rev - eta)
        samp["lc_b"] = np.log(0.20) + (log_rev - eta)
        res.sample = samp
        mu_a = sp.markup(res, revenue="log_rev", input_cost="lc_a", flexible_input="l")
        mu_b = sp.markup(res, revenue="log_rev", input_cost="lc_b", flexible_input="l")
        assert np.allclose(mu_b.to_numpy(), 2.0 * mu_a.to_numpy(), atol=1e-9)


# ---------------------------------------------------------------------------
# sp.metafrontier / sp.te_rank — nested-frontier known truth
# ---------------------------------------------------------------------------
def _nested_frontier_panel(seed=0, n_per_group=150):
    """Two groups sharing one slope; group A's frontier strictly dominates
    group B's (higher intercept). Both groups draw the same inefficiency
    distribution. Known truths:

      * group A sits ON the meta-frontier -> mean TGR_A approx 1.
      * group B is technologically dominated -> TGR_B < 1, so te_meta_B is
        well below te_group_B even though their te_group means coincide.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for grp, intercept in [("A", 2.0), ("B", 1.0)]:
        for _ in range(n_per_group):
            x = rng.uniform(1.0, 3.0)
            u = abs(rng.normal(0.0, 0.15))  # inefficiency
            v = rng.normal(0.0, 0.05)  # noise
            y = intercept + 0.5 * x - u + v
            rows.append({"y": y, "x": x, "grp": grp})
    return pd.DataFrame(rows)


class TestMetafrontierAnalytic:
    def test_dominant_group_on_metafrontier(self):
        df = _nested_frontier_panel()
        mf = sp.metafrontier(df, y="y", x=["x"], group="grp")
        gv = mf.data_info["group_vec"]
        mean_tgr = mf.tgr.groupby(gv).mean()
        # Group A defines the meta-frontier: its technology-gap ratio is ~1.
        assert mean_tgr["A"] == pytest.approx(1.0, abs=0.02)
        # Group B is dominated: a strictly smaller technology-gap ratio.
        assert mean_tgr["B"] < 0.9
        assert mean_tgr["A"] > mean_tgr["B"]

    def test_tgr_bounded_and_identity_holds(self):
        df = _nested_frontier_panel(seed=2)
        mf = sp.metafrontier(df, y="y", x=["x"], group="grp")
        # TGR in (0, 1] by construction of the enveloping LP.
        assert mf.tgr.min() > 0.0
        assert mf.tgr.max() == pytest.approx(1.0, abs=1e-6)
        # Decomposition identity TE_meta = TE_group * TGR holds exactly.
        lhs = mf.te_meta.to_numpy()
        rhs = mf.te_group.to_numpy() * mf.tgr.to_numpy()
        assert np.allclose(lhs, rhs, atol=1e-10)

    def test_metafrontier_te_below_group_te_for_dominated(self):
        df = _nested_frontier_panel(seed=5)
        mf = sp.metafrontier(df, y="y", x=["x"], group="grp")
        gv = mf.data_info["group_vec"]
        te_meta = mf.te_meta.groupby(gv).mean()
        te_group = mf.te_group.groupby(gv).mean()
        # Dominated group: meta-efficiency is strictly below group-efficiency.
        assert te_meta["B"] < te_group["B"] - 0.2
        # The two groups face the same inefficiency draws -> similar te_group.
        assert te_group["A"] == pytest.approx(te_group["B"], abs=0.05)


class TestTERankAnalytic:
    """Efficiency ranking known truth: a deterministically dominated firm must
    rank dead last; a near-frontier firm must rank near the top and score
    strictly higher than the dominated one.
    """

    def test_dominated_firm_ranks_last(self):
        rng = np.random.default_rng(7)
        n = 200
        x = rng.uniform(1.0, 3.0, n)
        u = abs(rng.normal(0.0, 0.2, n))
        v = rng.normal(0.0, 0.03, n)
        y = 1.0 + 0.5 * x - u + v
        df = pd.DataFrame({"y": y, "x": x})
        # Firm 0: essentially on the frontier (tiny inefficiency).
        df.loc[0, ["y", "x"]] = [1.0 + 0.5 * 2.0 - 0.001, 2.0]
        # Firm 1: heavily inefficient (large shortfall from the frontier).
        df.loc[1, ["y", "x"]] = [1.0 + 0.5 * 2.0 - 1.2, 2.0]

        fr = sp.frontier(df, y="y", x=["x"])
        tab = sp.te_rank(fr)

        # rank column is 1..n, sorted descending by efficiency.
        assert int(tab.iloc[0]["rank"]) == 1
        effs = tab["efficiency"].to_numpy()
        assert np.all(effs[:-1] >= effs[1:])  # monotone non-increasing
        # The dominated firm is last; the near-frontier firm beats it.
        assert int(tab.loc[1, "rank"]) == n
        assert tab.loc[0, "efficiency"] > tab.loc[1, "efficiency"]
        assert int(tab.loc[0, "rank"]) < n // 4  # firm 0 ranks in the top quartile
