"""Fifth batch of anchors from the 2026-10 pass: the long tail.

Functions that compose other estimators (workflows, comparison tables,
dispatchers) are checked for the one thing that can go wrong in them:
that the number they report is the number the underlying estimator
returns on the same data. Tests and simulators are checked against a
planted answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def staggered():
    df = sp.dgp_did(n_units=200, n_periods=8, effect=0.5, staggered=True, seed=0)
    df["g"] = df["first_treat"].fillna(0)
    return df


@pytest.fixture(scope="module")
def sharp_rd():
    df = sp.dgp_rd(n=2000, effect=0.5, seed=0)
    df["z"] = np.random.default_rng(0).normal(size=len(df))
    return df


@pytest.fixture(scope="module")
def synth_panel():
    return sp.dgp_synth(n_units=15, n_periods=24, treatment_time=16, effect=1.0, seed=0)


SYNTH = dict(outcome="y", unit="unit", time="time", treated_unit=0, treatment_time=16)


# --------------------------------------------------------------------- #
#  cate_pretrend_test
# --------------------------------------------------------------------- #


def _forest_panel(seed, pre, n=150, periods=10):
    """Effect 1 + x from adoption; treated units with high x drift by
    `pre * x` a period before adoption when `pre` is not zero."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        x = rng.normal()
        g = rng.choice([5, 7, 0], p=[0.35, 0.35, 0.3])
        a = rng.normal()
        for t in range(periods):
            d = int(g > 0 and t >= g)
            drift = pre * x * (t - g) if (g > 0 and t < g) else 0.0
            y = a + 0.2 * t + 0.5 * x + d * (1.0 + x) + drift + rng.normal(0, 0.5)
            rows.append(dict(id=i, t=t, d=d, x=x, y=y))
    return pd.DataFrame(rows)


def _pretrend_pvalues(pre, seeds):
    joint, equal = [], []
    for seed in seeds:
        forest = sp.causal_forest(
            data=_forest_panel(seed, pre),
            y="y",
            d="d",
            x=["x"],
            id="id",
            time="t",
            fe="twoway",
            n_estimators=150,
            random_state=seed,
        )
        res = sp.cate_pretrend_test(forest, n_groups=2, leads=3)
        joint.append(res["joint_zero"]["p"])
        equal.append(res["equal_across_groups"]["p"])
    return np.array(joint), np.array(equal)


def test_cate_pretrend_test_size_under_parallel_trends():
    # Real effect heterogeneity, no pre-trend. 100 replications: 5% and 2%.
    joint, equal = _pretrend_pvalues(0.0, range(40))
    assert np.mean(joint < 0.05) <= 0.15
    assert np.mean(equal < 0.05) <= 0.15


def test_cate_pretrend_test_detects_a_group_specific_pretrend():
    # 100 replications: both tests reject every time.
    joint, equal = _pretrend_pvalues(0.1, range(15))
    assert np.mean(joint < 0.05) >= 0.9
    assert np.mean(equal < 0.05) >= 0.9


# --------------------------------------------------------------------- #
#  DiD workflows report the estimator's own numbers
# --------------------------------------------------------------------- #


def test_cs_report_overall_is_the_aggregated_callaway_santanna(staggered):
    report = sp.cs_report(staggered, y="y", g="g", t="time", i="unit")
    fit = sp.callaway_santanna(staggered, y="y", g="g", t="time", i="unit")
    assert report.overall["estimate"] == pytest.approx(fit.estimate, rel=1e-10)
    simple = sp.aggte(fit, type="simple")
    assert report.overall["estimate"] == pytest.approx(simple.estimate, rel=1e-10)
    assert abs(report.overall["estimate"] - 0.5) <= 4.0 * report.overall["se"]


def test_did_analysis_main_result_is_the_named_method(staggered):
    out = sp.did_analysis(staggered, y="y", treat="g", time="time", id="unit")
    assert out.design == "staggered"
    fit = sp.callaway_santanna(staggered, y="y", g="g", t="time", i="unit")
    assert out.main_result.estimate == pytest.approx(fit.estimate, rel=1e-10)
    # The Goodman-Bacon pieces add up to the two-way fixed effects slope.
    assert out.bacon["weighted_sum"] == pytest.approx(out.bacon["beta_twfe"], rel=1e-8)


def test_did_summary_is_near_the_planted_effect(staggered):
    out = sp.did_summary(
        staggered,
        y="y",
        time="time",
        first_treat="g",
        group="unit",
        methods=["cs", "sa", "bjs"],
    )
    assert abs(out.estimate - 0.5) <= 4.0 * out.se
    assert 0.01 < out.se < 0.15


def test_did_cluster_diagnostics_counts(staggered):
    out = sp.did_cluster_diagnostics(staggered, unit="unit", first_treat="g")
    per_unit = staggered.groupby("unit")["g"].first()
    assert out.n_clusters == per_unit.size
    assert out.n_treated_clusters == int((per_unit > 0).sum())
    assert out.n_control_clusters == int((per_unit == 0).sum())
    want = per_unit[per_unit > 0].value_counts().to_dict()
    assert {int(k): int(v) for k, v in out.clusters_per_cohort.items()} == {
        int(k): int(v) for k, v in want.items()
    }


def test_compare_event_study_conventions_benchmark_row():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        tr = i < 100
        a = rng.normal()
        for t in range(8):
            y = a + 0.2 * t + (1.0 if (tr and t >= 4) else 0.0) + rng.normal()
            rows.append(dict(id=i, t=t, g=(4 if tr else 0), y=y))
    df = pd.DataFrame(rows)
    out = sp.compare_event_study_conventions(
        df, y="y", unit="id", time="t", first_treat="g"
    )
    table = out.table.set_index("key")
    # The dynamic TWFE path is the benchmark: zero gap against itself.
    assert table.loc["event_study", "shape_gap"] == 0.0
    assert bool(table.loc["event_study", "matches_twfe"])
    # In a single-cohort design, Callaway-Sant'Anna with a universal base
    # period is the same set of long differences.
    cs = table.loc["callaway_santanna[base_period=universal]"]
    assert abs(cs["shape_gap"]) < 1e-8
    es = out.paths[out.paths["key"] == "event_study"].set_index("relative_time")
    assert es.loc[-1, "att"] == 0.0
    assert es.loc[es.index >= 0, "att"].mean() == pytest.approx(1.0, abs=0.3)


def test_compare_event_study_conventions_refuses_staggered(staggered):
    with pytest.raises(ValueError, match="non-staggered"):
        sp.compare_event_study_conventions(
            staggered, y="y", unit="unit", time="time", first_treat="g"
        )


# --------------------------------------------------------------------- #
#  RD tables report rdrobust's own numbers
# --------------------------------------------------------------------- #


def test_rdsummary_headline_is_rdrobust(sharp_rd):
    out = sp.rdsummary(sharp_rd, y="y", x="x", verbose=False)
    direct = sp.rdrobust(sharp_rd, y="y", x="x")
    assert out["estimate"].estimate == direct.estimate
    assert out["estimate"].se == direct.se
    # One row of the bandwidth sweep is the MSE-optimal bandwidth itself.
    sweep = out["bw_sensitivity"]
    h = direct.model_info.get("h", direct.model_info.get("bandwidth_h"))
    if h is not None:
        assert np.min(np.abs(sweep["bandwidth"] - float(np.ravel(h)[0]))) < 1e-8


def test_rd_compare_rows_are_the_direct_fits(sharp_rd):
    table = sp.rd_compare(sharp_rd, y="y", x="x").set_index("method")
    # Every default row is a result: the local-randomisation row, which
    # needs a window and failed on every call without one, is no longer
    # in the default set.
    assert list(table.index) == ["rdrobust", "honest"]
    assert (table["status"] == "ok").all()
    direct = sp.rdrobust(sharp_rd, y="y", x="x")
    assert table.loc["rdrobust", "estimate"] == direct.estimate
    assert table.loc["rdrobust", "se"] == direct.se
    with_window = sp.rd_compare(
        sharp_rd,
        y="y",
        x="x",
        methods=["randinf"],
        method_kwargs={"randinf": {"wl": -0.1, "wr": 0.1}},
    )
    assert with_window.loc[0, "status"] == "ok"
    assert np.isfinite(with_window.loc[0, "estimate"])


def test_rd_robustness_table_cells_are_rdrobust_calls(sharp_rd):
    table = sp.rd_robustness_table(sharp_rd, y="y", x="x")
    assert (table["status"] == "ok").all()
    assert len(table) == 3 * 3 * 2
    for _, row in table.iloc[[0, 5, 11]].iterrows():
        direct = sp.rdrobust(
            sharp_rd,
            y="y",
            x="x",
            kernel=row["kernel"],
            bwselect=row["bwselect"],
            p=int(row["p"]),
        )
        assert row["estimate_rbc"] == pytest.approx(direct.estimate, rel=1e-10)
        assert row["se_rbc"] == pytest.approx(direct.se, rel=1e-10)


def test_rd_external_validity_passes_when_the_score_adds_nothing():
    # y depends on the score only through z, so conditional independence
    # of the outcome and the score given z holds on both sides.
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(size=n)
    x = 0.6 * z + rng.normal(0, 0.8, n)
    y = 0.8 * z + 1.0 * (x >= 0) + rng.normal(0, 0.5, n)
    out = sp.rd_external_validity(
        pd.DataFrame(dict(y=y, x=x, z=z)), y="y", x="x", c=0, covs=["z"]
    )
    assert out["ci_test"]["ci_holds"] is True
    # And fails when the score has its own effect.
    y2 = y + 0.8 * x
    out2 = sp.rd_external_validity(
        pd.DataFrame(dict(y=y2, x=x, z=z)), y="y", x="x", c=0, covs=["z"]
    )
    assert out2["ci_test"]["ci_holds"] is False


# --------------------------------------------------------------------- #
#  Synthetic-control helpers
# --------------------------------------------------------------------- #


def test_synth_compare_rows_are_the_direct_fits(synth_panel):
    out = sp.synth_compare(
        synth_panel, methods=["classic", "sdid"], placebo=False, **SYNTH
    )
    table = out.comparison_table.set_index("method")
    direct = sp.synth(synth_panel, method="classic", placebo=False, **SYNTH)
    assert table.loc["classic", "att"] == pytest.approx(direct.estimate, rel=1e-10)
    assert out.results["classic"].estimate == pytest.approx(direct.estimate, rel=1e-10)
    assert abs(table.loc["classic", "att"] - 1.0) < 0.3


def test_synth_sensitivity_leave_one_out_rows(synth_panel):
    out = sp.synth_sensitivity(synth_panel, n_donor_samples=5, seed=0, **SYNTH)
    loo = out["loo"]
    donors = sorted(set(synth_panel["unit"]) - {0})
    assert sorted(loo["dropped_unit"]) == donors
    dropped = int(loo["dropped_unit"].iloc[0])
    direct = sp.synth(
        synth_panel[synth_panel["unit"] != dropped],
        method="classic",
        placebo=False,
        **SYNTH,
    )
    assert loo["att"].iloc[0] == pytest.approx(direct.estimate, rel=1e-6)
    # In-time placebos before the true date are near zero; the effect is 1.
    assert out["time_placebo"]["att"].abs().median() < 0.4


def test_qqsynth_is_discos_with_the_quantile_method(synth_panel):
    a = sp.qqsynth(synth_panel, placebo=False, seed=0, **SYNTH)
    b = sp.discos(synth_panel, method="quantile", placebo=False, seed=0, **SYNTH)
    assert a.estimate == b.estimate


def test_synth_mde_is_the_first_grid_effect_reaching_the_target():
    df = sp.dgp_synth(n_units=30, n_periods=24, treatment_time=16, effect=0.0, seed=0)
    kw = dict(SYNTH, n_simulations=60, seed=0)
    mde = sp.synth_mde(df, **kw)
    curve = sp.synth_power(df, **kw)
    reached = curve[curve["power"] >= 0.8]
    assert np.isfinite(mde)
    assert mde == float(reached["effect_size"].iloc[0])


def test_synth_mde_says_why_it_is_infinite(synth_panel):
    # 14 placebo units: the smallest attainable p-value is 1/15 > 0.05.
    # This used to come back as a bare inf.
    with pytest.warns(RuntimeWarning, match="smallest attainable"):
        mde = sp.synth_mde(synth_panel, n_simulations=30, seed=0, **SYNTH)
    assert mde == np.inf


# --------------------------------------------------------------------- #
#  Regression diagnostics dispatchers
# --------------------------------------------------------------------- #


def test_diagnose_runs_the_named_tests(sharp_rd):
    out = sp.diagnose(sharp_rd, "y", ["x", "z"], print_results=False)
    het = sp.het_test(sharp_rd, "y", ["x", "z"])
    reset = sp.reset_test(sharp_rd, "y", ["x", "z"])
    assert out["het_test"]["statistic"] == pytest.approx(het["statistic"], rel=1e-10)
    assert out["reset_test"]["statistic"] == pytest.approx(
        reset["statistic"], rel=1e-10
    )
    np.testing.assert_allclose(
        out["vif"]["VIF"], sp.vif(sharp_rd, ["x", "z"])["VIF"], rtol=1e-10
    )
    # Two unrelated regressors: variance inflation of 1.
    np.testing.assert_allclose(out["vif"]["VIF"], 1.0, atol=0.01)


def test_estat_and_diagnose_result_agree_with_the_direct_tests(sharp_rd):
    fit = sp.regress("y ~ x", data=sharp_rd)
    het = sp.het_test(sharp_rd, "y", ["x"])
    rows = sp.estat(fit, print_results=False)
    bp = next(r for r in rows if r["test"].startswith("Breusch-Pagan"))
    assert bp["statistic"] == pytest.approx(het["statistic"], rel=1e-8)
    assert bp["pvalue"] == pytest.approx(het["pvalue"], rel=1e-8)
    dw = next(r for r in rows if r["test"].startswith("Durbin-Watson"))
    resid = fit.residuals() if callable(fit.residuals) else fit.residuals
    resid = np.asarray(resid, dtype=float)
    assert dw["statistic"] == pytest.approx(
        float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2)), rel=1e-8
    )
    battery = sp.diagnose_result(fit, print_results=False)
    assert battery["method_type"] == "ols"
    assert len(battery["checks"]) >= 1


def test_robustness_report_baseline_is_the_regression(sharp_rd):
    out = sp.robustness_report(sharp_rd, "y ~ x + z", x="x")
    fit = sp.regress("y ~ x + z", data=sharp_rd)
    assert float(out.baseline_estimate) == pytest.approx(
        float(fit.params["x"]), rel=1e-10
    )
    table = out.results_df.set_index("check")
    dropped = sp.regress("y ~ x", data=sharp_rd)
    assert table.loc["- z", "estimate"] == pytest.approx(
        float(dropped.params["x"]), rel=1e-10
    )


# --------------------------------------------------------------------- #
#  Dispatchers
# --------------------------------------------------------------------- #


def test_conformal_dispatcher_reaches_the_named_estimator():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(60):
        shock = rng.normal(0, 0.5)
        for _ in range(8):
            tr = rng.integers(0, 2)
            x = rng.normal()
            rows.append(
                dict(
                    cl=c,
                    d=tr,
                    x=x,
                    y=1 + 0.6 * tr + 0.4 * x + shock + rng.normal(0, 0.5),
                )
            )
    df = pd.DataFrame(rows)
    kw = dict(
        y="y",
        treatment="d",
        cluster="cl",
        covariates=["x"],
        test_clusters=list(range(40, 60)),
        alpha=0.1,
        random_state=0,
    )
    via = sp.conformal("interference", data=df, **kw)
    direct = sp.conformal_interference(df, **kw)
    pd.testing.assert_frame_equal(via.predictions, direct.predictions)
    with pytest.raises(Exception):
        sp.conformal("no-such-kind", data=df)


def test_bridge_did_sc_two_paths_recover_the_effect():
    rng = np.random.default_rng(0)
    rows = []
    for u in [f"u{i}" for i in range(12)] + ["CA"]:
        base = rng.normal(10, 1)
        for yr in range(1985, 1995):
            eff = 2.0 if (u == "CA" and yr >= 1990) else 0.0
            y = base + 0.1 * (yr - 1985) + eff + rng.normal(0, 0.2)
            rows.append({"state": u, "year": yr, "gdp": y})
    out = sp.bridge(
        kind="did_sc",
        data=pd.DataFrame(rows),
        y="gdp",
        unit="state",
        time="year",
        treated_unit="CA",
        treatment_time=1990,
    )
    assert out.kind == "did_sc"
    estimates = [
        float(v)
        for k, v in vars(out).items()
        if k.startswith("estimate") and np.isscalar(v)
    ]
    assert estimates, "bridge result exposes no path estimates"
    for value in estimates:
        assert value == pytest.approx(2.0, abs=0.5)


# --------------------------------------------------------------------- #
#  DAG helpers and benchmark generators
# --------------------------------------------------------------------- #


def _chain(seed, n):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(size=n)
    c = 0.5 * b + rng.normal(size=n)
    return pd.DataFrame(dict(a=a, b=b, c=c))


class TestDagValidate:
    def test_true_edges_are_supported_and_a_spurious_one_is_not(self):
        df = _chain(0, 3000)
        out = sp.llm_dag_validate(sp.dag("a -> b; b -> c; a -> c"), df)
        ev = out.edge_evidence.set_index("edge")
        assert bool(ev.loc[[("a", "b")], "supported"].iloc[0])
        assert bool(ev.loc[[("b", "c")], "supported"].iloc[0])
        assert not bool(ev.loc[[("a", "c")], "supported"].iloc[0])
        assert (out.n_supported, out.n_unsupported) == (2, 1)

    def test_spurious_edge_is_kept_at_the_nominal_rate(self):
        # a and c are independent given b, so "supported" for a -> c is a
        # false positive: 5.5% over 200 samples at alpha = 0.05.
        kept = []
        for seed in range(80):
            ev = sp.llm_dag_validate(
                [("a", "b"), ("b", "c"), ("a", "c")], _chain(seed, 500)
            ).edge_evidence.set_index("edge")
            kept.append(bool(ev.loc[[("a", "c")], "supported"].iloc[0]))
        assert np.mean(kept) <= 0.15

    def test_edge_list_and_dict_give_the_same_answer_as_a_dag(self):
        # A plain edge list used to be read as "no edges" and returned
        # 0 supported, 0 unsupported, which looks like a clean validation.
        df = _chain(1, 2000)
        ref = sp.llm_dag_validate(sp.dag("a -> b; b -> c"), df)
        as_list = sp.llm_dag_validate([("a", "b"), ("b", "c")], df)
        as_dict = sp.llm_dag_validate({"a": ["b"], "b": ["c"]}, df)
        assert (ref.n_supported, ref.n_unsupported) == (2, 0)
        for other in (as_list, as_dict):
            assert (other.n_supported, other.n_unsupported) == (2, 0)

    def test_unreadable_input_raises(self):
        with pytest.raises(ValueError, match="cannot read edges"):
            sp.llm_dag_validate(42, _chain(0, 100))


def test_dag_simulate_reproduces_the_two_collider_stories():
    # Discrimination: the true effect on wages is -1; adding occupation
    # (a collider with ability) flips the sign; adding ability restores it.
    d = sp.dag_simulate("discrimination", n=20000, seed=0)

    def coef(cols):
        X = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
        return np.linalg.lstsq(X, d["wage"], rcond=None)[0][1]

    assert coef(["female", "occupation"]) > 0
    assert coef(["female", "occupation", "ability"]) == pytest.approx(-1.0, abs=0.1)
    # Movie stars: beauty and talent are independent in the population and
    # negatively correlated among stars.
    m = sp.dag_simulate("movie_star", n=20000, seed=0)
    assert abs(np.corrcoef(m["beauty"], m["talent"])[0, 1]) < 0.03
    stars = m[m["star"] == 1]
    assert np.corrcoef(stars["beauty"], stars["talent"])[0, 1] < -0.1


def test_causal_rl_benchmark_declares_the_arm_that_is_actually_better():
    out = sp.causal_rl_benchmark(n_episodes=20000, confounding_strength=0.5, seed=0)
    t = out.transitions
    by_action = t.groupby("action")["reward"].mean()
    best = int(np.ravel(out.optimal_policy)[0])
    assert by_action.idxmax() == best
    # The logged contrast is confounded upward by the hidden state, so the
    # declared optimal value sits between the two logged arm means.
    assert by_action.min() < out.optimal_value < by_action.max()
    # Without confounding the logged mean of the best arm is its value.
    clean = sp.causal_rl_benchmark(n_episodes=20000, confounding_strength=0.0, seed=0)
    clean_means = clean.transitions.groupby("action")["reward"].mean()
    assert clean_means.max() == pytest.approx(clean.optimal_value, abs=0.05)


def test_cate_summary_is_the_description_of_the_fitted_effects():
    df = sp.dgp_observational(n=1000, seed=0)
    fit = sp.metalearner(
        df, y="y", treat="treatment", covariates=["x1", "x2"], learner="t"
    )
    out = sp.cate_summary(fit)
    cate = np.asarray(fit.model_info["cate"], dtype=float)
    values = {str(k).lower(): float(v) for k, v in out["CATE"].items()}
    assert values["mean"] == pytest.approx(cate.mean(), rel=1e-10)
    assert values["min"] == pytest.approx(cate.min(), rel=1e-10)
    assert values["max"] == pytest.approx(cate.max(), rel=1e-10)


# --------------------------------------------------------------------- #
#  Neural point estimates and the propensity network
# --------------------------------------------------------------------- #


def _confounded(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    d = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x1)))
    y = 1 + x1 + 0.5 * x2 + (1.0 + 0.5 * x2) * d + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def test_cfrnet_and_dragonnet_remove_observed_confounding():
    pytest.importorskip("torch")
    df = _confounded()
    raw = df.loc[df["d"] == 1, "y"].mean() - df.loc[df["d"] == 0, "y"].mean()
    assert raw > 1.5
    kw = dict(y="y", treat="d", covariates=["x1", "x2"], epochs=150, n_bootstrap=50)
    # Four seeds each: 0.96 to 1.03.
    assert sp.cfrnet(df, **kw).estimate == pytest.approx(1.0, abs=0.12)
    dragon = sp.dragonnet(df, **kw)
    assert dragon.estimate == pytest.approx(1.0, abs=0.12)
    # DragonNet's SE is the AIPW influence-function one: on the scale of
    # the efficiency bound, 0.026 here.
    assert 0.015 < dragon.se < 0.05


def test_cevae_average_effect():
    pytest.importorskip("torch")
    # Three seeds: 1.03, 1.08, 1.10 for a true 1.0; the raw contrast is 1.65.
    df = _confounded()
    out = sp.cevae(df[["x1", "x2"]].to_numpy(), df["d"].to_numpy(), df["y"].to_numpy())
    assert out.ate == pytest.approx(1.0, abs=0.3)
    assert len(out.ite) == len(df)


def test_dl_propensity_score_tracks_the_true_propensity():
    # It overfits (RMSE 0.11 against a logistic truth at n = 3000), so the
    # bar is that it orders units correctly and is right on average.
    df = _confounded(n=3000)
    e = sp.dl_propensity_score(df, treatment="d", covariates=["x1", "x2"])
    truth = 1 / (1 + np.exp(-0.8 * df["x1"].to_numpy()))
    assert np.all((e >= 0.02) & (e <= 0.98))
    assert np.corrcoef(e, truth)[0, 1] > 0.7
    assert e.mean() == pytest.approx(df["d"].mean(), abs=0.03)


# --------------------------------------------------------------------- #
#  Reports print the numbers of the fits they describe
# --------------------------------------------------------------------- #


def test_pretrends_summary_quotes_the_joint_test():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        tr = i < 100
        a = rng.normal()
        for t in range(8):
            y = a + 0.2 * t + (1.0 if (tr and t >= 4) else 0.0) + rng.normal()
            rows.append(dict(id=i, t=t, g=(4 if tr else np.nan), y=y))
    es = sp.event_study(
        pd.DataFrame(rows), y="y", treat_time="g", time="t", unit="id", window=(-4, 3)
    )
    text = sp.pretrends_summary(es)
    test = sp.pretrends_test(es)
    assert f"p = {test['pvalue']:.3f}" in text
    # The report prints the Wald form: df times the F statistic.
    assert f"chi2({test['df']}) = {test['df'] * test['statistic']:.2f}" in text
    assert "Cannot reject parallel trends" in text


def test_synth_report_quotes_the_fit(synth_panel):
    text = sp.synth_report(synth_panel, sensitivity=False, placebo=False, **SYNTH)
    fit = sp.synth(synth_panel, method="classic", placebo=False, **SYNTH)
    assert f"{fit.estimate:.4f}" in text
    assert f"{fit.se:.4f}" in text
    assert "Donor pool: 14 units" in text


def test_did_report_writes_the_summary_it_returns(staggered, tmp_path):
    out = sp.did_report(
        staggered,
        y="y",
        time="time",
        first_treat="g",
        group="unit",
        save_to=str(tmp_path),
        methods=["cs", "bjs"],
        include_sensitivity=False,
    )
    direct = sp.did_summary(
        staggered,
        y="y",
        time="time",
        first_treat="g",
        group="unit",
        methods=["cs", "bjs"],
    )
    assert out.estimate == pytest.approx(direct.estimate, rel=1e-10)
    written = {p.name for p in tmp_path.iterdir()}
    assert {"did_summary.json", "did_summary.md", "did_summary.tex"} <= written
    import json

    payload = json.loads((tmp_path / "did_summary.json").read_text(encoding="utf-8"))
    flat = json.dumps(payload)
    # The headline number in the file is the one on the returned object.
    assert f"{out.estimate:.6f}"[:8] in flat


def test_synth_recommend_is_the_comparison_winner(synth_panel):
    name = sp.synth_recommend(synth_panel, **SYNTH)
    table = sp.synth_compare(synth_panel, **SYNTH)
    assert name == table.recommended
    assert name in set(table.comparison_table["method"])


def test_rd_cate_summary_rows_are_the_three_fits(sharp_rd):
    out = sp.rd_cate_summary(sharp_rd, y="y", x="x", covs=["z"])
    table = out["comparison"].set_index("method")
    assert table.loc["Causal Forest", "estimate"] == out["forest"].estimate
    assert table.loc["Gradient Boosting", "estimate"] == out["boost"].estimate
    assert table.loc["LASSO RD", "estimate"] == out["lasso"].estimate
    # One seed of a design whose effect is 0.5: each within four SEs.
    for key in ("forest", "boost", "lasso"):
        assert abs(out[key].estimate - 0.5) <= 4.0 * out[key].se


def test_forest_support_flags_rows_outside_the_data():
    df = _forest_panel(0, 0.0)
    forest = sp.causal_forest(
        data=df,
        y="y",
        d="d",
        x=["x"],
        id="id",
        time="t",
        fe="twoway",
        n_estimators=150,
        random_state=0,
    )
    new = pd.DataFrame({"x": [0.0, 0.5, 25.0, -25.0]})
    out = sp.forest_support(forest, new)
    assert list(out["supported"]) == [True, True, False, False]
    assert list(out["n_outside_range"]) == [0, 0, 1, 1]
    assert (out["knn_ratio"].iloc[2:] > 1).all()
    # tau(x) = 1 + x: inside the support the prediction follows it.
    assert out["cate"].iloc[1] - out["cate"].iloc[0] == pytest.approx(0.5, abs=0.35)


def test_lisa_cluster_map_colours_the_two_blocks():
    gpd = pytest.importorskip("geopandas")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    from shapely.geometry import Point

    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(200, 2))
    w = sp.knn_weights(coords, k=6)
    right = coords[:, 0] > 0.5
    y = np.where(right, 3.0, -3.0) + rng.normal(scale=0.3, size=200)
    gdf = gpd.GeoDataFrame(geometry=[Point(xy) for xy in coords])
    ax = sp.lisa_cluster_map(y, w, gdf)
    faces = ax.collections[0].get_facecolor()
    assert len(faces) == 200
    high_high, low_low = to_rgba("#d62728"), to_rgba("#1f77b4")
    deep_right = coords[:, 0] > 0.7
    deep_left = coords[:, 0] < 0.3
    is_hh = np.all(np.isclose(faces, high_high), axis=1)
    is_ll = np.all(np.isclose(faces, low_low), axis=1)
    # A high block next to a low block: high-high on one side, low-low on
    # the other, and no point takes the other block's label.
    assert is_hh[deep_right].mean() > 0.8
    assert is_ll[deep_left].mean() > 0.8
    assert not is_hh[deep_left].any() and not is_ll[deep_right].any()
    import matplotlib.pyplot as plt

    plt.close("all")
