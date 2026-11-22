"""``sp.did_workflow``: the nine-step DiD protocol.

The workflow adds no estimator, so its tests are identities: every number
it reports must equal the number the public function behind that step
returns on the same panel. The statuses are checked on panels built so
that the right answer is known (a constant effect under exact parallel
trends, a treated group on its own trend, a treatment that switches off).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.workflow._degradation import WorkflowDegradedWarning

TAU = 2.0


def make_panel(
    n_units: int = 150,
    n_periods: int = 7,
    cohorts=(0, 4, 6),
    tau: float = TAU,
    trend: float = 0.0,
    noise: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Staggered panel with a constant effect ``tau``.

    ``trend`` adds a linear time trend to ever-treated units only, which
    violates parallel trends by construction.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(cohorts)[np.arange(n_units) % len(cohorts)]
    alpha = rng.normal(size=n_units)
    rows = []
    for i in range(n_units):
        for t in range(1, n_periods + 1):
            d = int(g[i] > 0 and t >= g[i])
            y = alpha[i] + 0.3 * t + tau * d + trend * t * (g[i] > 0)
            rows.append((i, t, int(g[i]), d, y + noise * rng.normal()))
    return pd.DataFrame(rows, columns=["unit", "time", "g", "d", "y"])


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture(scope="module")
def wf(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_workflow(
            panel,
            y="y",
            id="unit",
            time="time",
            cohort="g",
            target="overall",
            control_group="nevertreated",
            cluster="unit",
            jackknife=False,
        )


@pytest.fixture(scope="module")
def reference(panel):
    """The same analysis, assembled by hand from the public functions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.callaway_santanna(panel, y="y", g="g", t="time", i="unit")
        return {
            "fit": fit,
            "simple": sp.aggte(fit, type="simple"),
            "dynamic": sp.aggte(fit, type="dynamic"),
        }


def quiet(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_workflow(**kwargs)


BASE = dict(y="y", id="unit", time="time")


# --------------------------------------------------------------------- #
# The protocol itself
# --------------------------------------------------------------------- #
def test_protocol_lists_nine_steps_with_real_functions_and_references():
    protocol = sp.did_protocol()
    assert protocol["step"].tolist() == list(range(1, 10))
    assert protocol["key"].tolist() == [
        "declare",
        "design",
        "twfe",
        "estimate",
        "event_study",
        "pretrends",
        "sensitivity",
        "robustness",
        "report",
    ]
    registered = set(sp.list_functions())
    for functions in protocol["functions"]:
        for name in functions:
            assert name in registered, name
    for refs in protocol["references"]:
        bib = sp.bibtex(list(refs))
        for key in refs:
            assert "{" + key + "," in bib, key


def test_plan_only_then_one_step_at_a_time(panel):
    plan = sp.did_workflow(panel, cohort="g", through=None, **BASE)
    assert plan.verdict == "pending"
    assert plan.pending == sp.did_protocol()["key"].tolist()
    first = plan.advance()
    assert first.key == "declare" and first.status != "pending"
    plan.run("design")
    assert plan.pending[0] == "twfe"
    assert plan.step("design").status == "pass"
    assert not hasattr(plan, "estimate")  # the step has not run


# --------------------------------------------------------------------- #
# Identities: each number is the public function's number
# --------------------------------------------------------------------- #
def test_headline_is_aggte_simple(wf, reference):
    ref = reference["simple"]
    assert wf.estimate == ref.estimate
    assert wf.se == ref.se
    assert wf.ci == (float(ref.ci[0]), float(ref.ci[1]))
    assert wf.pvalue == ref.pvalue


def test_known_effect_is_inside_the_interval(wf):
    lo, hi = wf.ci
    assert lo < TAU < hi
    assert abs(wf.estimate - TAU) < 4 * wf.se


def test_path_is_uniform_bands_of_the_dynamic_aggregation(wf, reference):
    ref = sp.uniform_bands(reference["dynamic"], alpha=0.05, seed=0)
    got = wf.event_study
    for col in ("relative_time", "att", "se", "ci_lower", "ci_upper"):
        np.testing.assert_array_equal(got[col].to_numpy(), ref[col].to_numpy())
    np.testing.assert_array_equal(
        got["cband_lower"].to_numpy(), ref["cband_lower"].to_numpy()
    )
    assert wf.step("event_study").evidence["crit_uniform"] == ref.attrs["crit_uniform"]


def test_cohort_counts_along_the_path(wf):
    # Cohorts 4 and 6 on periods 1..7: e=0,1 are reached by both, e=2,3 by
    # cohort 4 only; e=-5,-4 by cohort 6 only.
    counts = wf.event_study.set_index("relative_time")["n_cohorts"]
    assert counts.loc[0] == 2 and counts.loc[1] == 2
    assert counts.loc[2] == 1 and counts.loc[3] == 1
    assert counts.loc[-5] == 1 and counts.loc[-2] == 2
    assert wf.step("event_study").status == "caution"


def test_pretrend_test_and_power_are_the_public_ones(wf, reference):
    es = reference["dynamic"]
    test = sp.pretrends_test(es, alpha=0.05)
    ev = wf.step("pretrends").evidence
    assert ev["statistic"] == test["statistic"]
    assert ev["pvalue"] == test["pvalue"]
    assert ev["df"] == test["df"]
    slope = sp.pretrends_slope_for_power(es, target_power=0.8, alpha=0.05)["slope"]
    # Not bit-identical across calls: the power integral is scipy's
    # randomised multivariate-normal CDF (abseps 1e-5), so two calls on the
    # same result differ around the eighth digit.
    assert ev["slope_power_80"] == pytest.approx(slope, rel=1e-5)
    slope = ev["slope_power_80"]
    post = wf.event_study.query("relative_time >= 0")
    # A linear violation that is zero at e = -1 biases horizon e by
    # slope * (e + 1); the average over horizons follows.
    assert ev["bias_power_80"] == pytest.approx(
        slope * (post["relative_time"] + 1).mean(), rel=1e-12
    )
    assert ev["bias_to_effect_ratio"] == pytest.approx(
        abs(ev["bias_power_80"]) / abs(post["att"].mean()), rel=1e-12
    )


def test_breakdown_values_are_breakdown_m(wf, reference):
    es = reference["dynamic"]
    n_post = int((wf.event_study["relative_time"] >= 0).sum())
    l_vec = np.ones(n_post) / n_post
    rm = sp.breakdown_m(es, method="relative_magnitude", l_vec=l_vec, seed=0)
    sd = sp.breakdown_m(es, method="smoothness", l_vec=l_vec)
    sens = wf.sensitivity
    assert sens["breakdown_rm"] == rm
    assert sens["breakdown_sd"] == sd


def test_twfe_coefficient_matches_feols_and_bacon_share(wf, panel):
    twfe = sp.feols("y ~ d | unit + time", data=panel)
    ev = wf.step("twfe").evidence
    assert ev["beta_twfe"] == pytest.approx(float(twfe.params["d"]), rel=1e-9)
    bacon = sp.bacon_decomposition(panel, y="y", treat="d", time="time", id="unit")
    assert ev["forbidden_weight_share"] == bacon["already_treated_control_weight_share"]
    assert ev["forbidden_weight_share"] > 0
    assert wf.step("twfe").status == "caution"


def test_robustness_rows_are_did_summary_and_the_other_comparison_group(wf, panel):
    table = wf.robustness
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = sp.did_summary(
            panel, y="y", time="time", first_treat="g", group="unit", cluster="unit"
        )
        notyet = sp.aggte(
            sp.callaway_santanna(
                panel, y="y", g="g", t="time", i="unit", control_group="notyettreated"
            ),
            type="simple",
        )
    est_rows = table[table["check"] == "estimator"].set_index("variant")
    for _, row in summary.detail.iterrows():
        assert est_rows.loc[row["method"], "estimate"] == row["estimate"]
    alt = table[table["check"] == "comparison group"].iloc[0]
    assert alt["variant"] == "notyettreated"
    assert alt["estimate"] == notyet.estimate and alt["se"] == notyet.se
    # constant effect, exact parallel trends: nothing should move the sign
    assert wf.step("robustness").status == "pass"
    ev = wf.step("robustness").evidence
    assert ev["min_estimate"] <= wf.estimate <= ev["max_estimate"] or (
        wf.ci[0] < ev["min_estimate"] and ev["max_estimate"] < wf.ci[1]
    )


def test_robustness_flags_an_alternative_that_loses_the_sign_conclusion():
    # A small effect on a noisy panel: the headline interval excludes zero
    # or not depending on the estimator. Whatever the draw, the step must
    # caution exactly when some alternative's interval includes zero while
    # the headline's does not.
    df = make_panel(n_units=90, tau=0.35, noise=1.5, seed=19)
    w = quiet(data=df, cohort="g", skip=["sensitivity"], jackknife=False, **BASE)
    table = w.robustness
    alt = table[table["check"].isin(["estimator", "comparison group"])]
    same_sign = np.sign(alt["estimate"]) == np.sign(w.estimate)
    includes = alt[(alt["ci_lower"] <= 0) & (alt["ci_upper"] >= 0) & same_sign]
    headline_excludes = w.ci[0] > 0 or w.ci[1] < 0
    ev = w.step("robustness").evidence
    assert ev["n_alternatives_including_zero"] == len(includes)
    if headline_excludes and len(includes):
        assert w.step("robustness").status == "caution"
        assert "includes zero under" in w.step("robustness").finding
        for variant in includes["variant"]:
            assert str(variant) in w.step("robustness").finding
    assert headline_excludes and len(
        includes
    ), "the fixture no longer exercises the branch; pick another seed"


def test_event_target_cs_equals_sun_abraham(panel, reference):
    # With never-treated controls, no covariates and a balanced panel the
    # two estimators give the same event-time path, hence the same average.
    cs = quiet(data=panel, cohort="g", target="event", through="estimate", **BASE)
    sa = quiet(
        data=panel,
        cohort="g",
        target="event",
        estimator="sa",
        through="estimate",
        **BASE,
    )
    assert cs.estimate == reference["dynamic"].estimate
    assert cs.se == reference["dynamic"].se
    assert sa.estimate == pytest.approx(cs.estimate, rel=1e-8)


def test_event_average_reproduces_aggte_overall(reference):
    import importlib

    event_average = importlib.import_module(
        "statspai.did._workflow_estimate"
    ).event_average

    es = reference["dynamic"]
    got = event_average(es, 0.05)
    assert got["estimate"] == pytest.approx(es.estimate, rel=1e-12)
    assert got["se"] == pytest.approx(es.se, rel=1e-10)


def test_group_and_calendar_targets_are_the_aggte_types(panel, reference):
    for target, agg in (("group", "group"), ("calendar", "calendar")):
        w = quiet(data=panel, cohort="g", target=target, through="estimate", **BASE)
        ref = sp.aggte(reference["fit"], type=agg)
        assert w.estimate == ref.estimate and w.se == ref.se


def test_treatment_path_gives_the_cohort_results(panel, wf):
    w = quiet(data=panel, treat="d", through="estimate", **BASE)
    assert w.estimate == wf.estimate and w.se == wf.se
    assert "derived from 'd'" in w.step("declare").evidence["cohort_source"]
    assert "_did_workflow_cohort" not in panel.columns  # caller's frame untouched


def test_date_typed_time_equals_integer_time(panel, wf):
    as_date = lambda s: pd.to_datetime("2000-01-01") + pd.to_timedelta(  # noqa: E731
        s * 30, unit="D"
    )
    dated = panel.assign(
        time=as_date(panel["time"]),
        g=as_date(panel["g"]).where(panel["g"] > 0),
    )
    w = quiet(data=dated, cohort="g", through="event_study", **BASE)
    assert w.estimate == pytest.approx(wf.estimate, rel=1e-12)
    assert w.design["n_cohorts"] == 2 and w.design["n_never_treated"] == 50
    np.testing.assert_array_equal(
        w.event_study["n_cohorts"].to_numpy(), wf.event_study["n_cohorts"].to_numpy()
    )


# --------------------------------------------------------------------- #
# Statuses on panels with a known answer
# --------------------------------------------------------------------- #
def test_declare_passes_when_stated_and_cautions_on_defaults(panel, wf):
    assert wf.step("declare").status == "pass"
    w = sp.did_workflow(panel, cohort="g", through="declare", **BASE)
    step = w.step("declare")
    assert step.status == "caution"
    for name in ("target", "control_group", "cluster"):
        assert name in step.finding and name in step.action
    choices = step.evidence["choices"].set_index("choice")
    assert choices.loc["control_group", "source"] == "default"
    assert choices.loc["estimator", "value"] == "cs"


def test_design_counts(wf):
    d = wf.design
    assert (d["n_units"], d["n_periods"], d["n_obs"]) == (150, 7, 1050)
    assert d["n_cohorts"] == 2 and d["n_treated_units"] == 100
    assert d["n_never_treated"] == 50 and d["n_always_treated"] == 0
    assert d["balanced"] and d["staggered"]
    cohorts = d["cohorts"].set_index("cohort")
    assert cohorts.loc[4, "n_pre_periods"] == 3
    assert cohorts.loc[6, "n_post_periods"] == 2
    assert wf.step("design").status == "pass"


def test_exact_parallel_trends_with_a_large_effect_passes_sensitivity(wf):
    assert wf.step("pretrends").evidence["reject"] is False
    assert wf.step("pretrends").status == "pass"
    assert wf.sensitivity["breakdown_rm"] >= 1.0
    assert wf.step("sensitivity").status == "pass"


def test_differential_trend_fails_the_pretrend_step():
    df = make_panel(n_units=300, trend=0.5, seed=1)
    w = quiet(data=df, cohort="g", through="pretrends", skip=["twfe"], **BASE)
    step = w.step("pretrends")
    assert step.evidence["reject"] is True
    assert step.status == "fail"
    assert w.verdict == "fail"
    assert "sensitivity" in step.action


def test_no_effect_gives_nothing_to_break_down():
    df = make_panel(tau=0.0, seed=3)
    w = quiet(data=df, cohort="g", through="sensitivity", skip=["twfe"], **BASE)
    assert bool(w.sensitivity["rm_grid"].iloc[0]["rejects_zero"]) is False
    assert w.step("sensitivity").status == "info"


def test_reversal_fails_the_design_and_blocks_estimation(panel):
    df = panel.copy()
    df.loc[(df["unit"] == 1) & (df["time"] == 7), "d"] = 0  # unit 1 has g = 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = sp.did_workflow(df, treat="d", **BASE)
    design = w.step("design")
    assert design.status == "fail" and design.evidence["n_reversal_units"] == 1
    assert "did_multiplegt_dyn" in design.action
    statuses = w.steps.set_index("key")["status"]
    assert statuses["estimate"] == "skipped" and statuses["robustness"] == "skipped"
    assert statuses["report"] == "info"
    assert w.verdict == "fail"
    assert not hasattr(w, "estimate")
    with pytest.raises(MethodIncompatibility, match="step 'estimate' is skipped"):
        w.estimate


def test_no_never_treated_units(panel):
    df = panel[panel["g"] > 0]
    w = quiet(data=df, cohort="g", **BASE)
    assert w.step("design").status == "fail"
    assert "notyettreated" in w.step("design").action
    w2 = quiet(
        data=df, cohort="g", control_group="notyettreated", through="estimate", **BASE
    )
    assert w2.step("design").status == "caution"
    assert np.isfinite(w2.estimate)


def test_duplicate_unit_period_rows_fail_the_design(panel):
    df = pd.concat([panel, panel.iloc[:2]], ignore_index=True)
    w = quiet(data=df, cohort="g", **BASE)
    assert w.step("design").status == "fail"
    assert w.step("design").evidence["n_duplicate_rows"] == 2


def test_single_adoption_date_has_no_twfe_diagnosis(panel):
    df = panel[panel["g"].isin([0, 4])]
    w = quiet(data=df, cohort="g", through="twfe", **BASE)
    assert w.step("twfe").status == "info"


def test_two_periods_group_dummy(panel):
    df = panel[panel["g"].isin([0, 4]) & panel["time"].isin([3, 4])].copy()
    df["grp"] = (df["g"] > 0).astype(int)
    w = quiet(data=df, treat="grp", skip=["robustness"], **BASE)
    ref = sp.aggte(
        sp.callaway_santanna(df, y="y", g="g", t="time", i="unit"), type="simple"
    )
    assert w.estimate == ref.estimate
    assert w.step("pretrends").status == "caution"  # untestable, said so
    assert w.step("sensitivity").status == "skipped"


def test_few_clusters_are_flagged_and_jackknifed():
    df = make_panel(n_units=60, seed=5)
    df["state"] = df["unit"] % 12  # cohort is constant within state (12 = 4 x 3)
    w = quiet(data=df, cohort="g", cluster="state", **BASE)
    design = w.step("design")
    assert design.evidence["n_clusters"] == 12
    assert design.evidence["cluster_grade"] == "below-evidence"
    assert design.status == "caution"
    jack = w.robustness[w.robustness["check"] == "standard error"].iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.cs_jackknife(
            df,
            y="y",
            g="g",
            time="time",
            id="unit",
            cluster="state",
            base_period="universal",
        )
    assert jack["se"] == ref.se
    assert w.step("robustness").evidence["jackknife_se_ratio"] == pytest.approx(
        ref.se / w.se, rel=1e-12
    )


# --------------------------------------------------------------------- #
# Failing loudly
# --------------------------------------------------------------------- #
def test_a_supporting_check_that_fails_is_recorded_not_hidden(panel, monkeypatch):
    import importlib

    # ``sp.did`` is a function, so the submodule is reached by name.
    ff = importlib.import_module("statspai.did.functional_form")

    def boom(*args, **kwargs):
        raise RuntimeError("no functional-form test today")

    monkeypatch.setattr(ff, "functional_form_test", boom)
    with pytest.warns(WorkflowDegradedWarning, match="functional_form_test"):
        w = sp.did_workflow(
            panel,
            cohort="g",
            target="overall",
            control_group="nevertreated",
            cluster="unit",
            skip=["sensitivity"],
            jackknife=False,
            **BASE,
        )
    assert [d["error_type"] for d in w.degradations] == ["RuntimeError"]
    assert "functional_form_test" in w.degradations[0]["section"]
    assert "outcome scale" not in set(w.robustness["check"])
    assert any("functional_form_test" in m for m in w.step("robustness").warnings)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(cohort="g", target="median"), "unknown target"),
        (dict(cohort="g", estimator="twfe"), "unknown estimator"),
        (dict(cohort="g", estimator="bjs", control_group="nevertreated"), "bjs"),
        (dict(cohort="g", estimator="sa", target="group"), "needs estimator='cs'"),
        (dict(cohort="g", treat="d"), "exactly one of cohort="),
        (dict(), "exactly one of cohort="),
        (dict(cohort="g", skip=["estimate"]), "cannot be skipped"),
        (dict(cohort="g", through="finish"), "is not a step"),
        (dict(cohort="g", event_window=(-3, -1)), "no post-treatment"),
        (dict(cohort="g", alpha=1.5), "alpha must be in"),
        (dict(cohort="g", anticipation=-1), "non-negative integer"),
        (dict(cohort="g", estimator="bjs", anticipation=1), "anticipation"),
        (dict(cohort="g", estimator="sa", jackknife=True), "jackknife=True"),
    ],
)
def test_bad_options_are_refused_with_a_hint(panel, kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match) as err:
        quiet(data=panel, **BASE, **kwargs)
    assert err.value.recovery_hint


def test_group_dummy_on_many_periods_is_refused(panel):
    df = panel.assign(grp=(panel["g"] > 0).astype(int))
    with pytest.raises(MethodIncompatibility, match="constant within every unit"):
        sp.did_workflow(df, treat="grp", **BASE)


# --------------------------------------------------------------------- #
# Reporting and export
# --------------------------------------------------------------------- #
def test_report_states_what_the_paper_owes(wf):
    items = wf.checklist.set_index("item")["statement"]
    for item in (
        "Target parameter",
        "Identifying assumption",
        "Weights",
        "Reference period",
        "Event window",
        "Standard errors",
        "Pre-trends",
        "Sensitivity",
    ):
        assert item in items.index
    assert "never-treated" in items["Identifying assumption"]
    assert "150 clusters" in items["Standard errors"]
    assert wf.report().startswith("Target parameter: ")
    bib = sp.bibtex(wf.cite().split("\n"))
    for key in wf.cite().split("\n"):
        assert "{" + key + "," in bib


def test_exports(wf, tmp_path):
    for detail in ("minimal", "standard", "agent"):
        payload = wf.to_dict(detail=detail)
        text = json.dumps(payload)  # JSON-safe at every level
        assert payload["verdict"] == wf.verdict
        assert payload["estimate"] == wf.estimate
    assert len(json.dumps(wf.to_dict(detail="minimal"))) < len(text)
    assert [s["key"] for s in wf.to_dict()["steps"]] == wf.steps["key"].tolist()
    assert "evidence" in wf.to_dict(detail="agent")["steps"][0]
    md = wf.to_markdown()
    assert md.count("\n| ") >= 10 and "**What to report**" in md
    tex = wf.to_latex(caption="DiD protocol", label="tab:did")
    assert tex.count(r"\\") == 10 and r"\label{tab:did}" in tex
    xlsx = wf.to_excel(str(tmp_path / "wf.xlsx"))
    sheets = pd.read_excel(xlsx, sheet_name=None)
    assert {"steps", "cohorts", "event_study", "sensitivity", "robustness"} <= set(
        sheets
    )
    assert len(sheets["steps"]) == 9
    docx = wf.to_word(str(tmp_path / "wf.docx"))
    assert (tmp_path / "wf.docx").stat().st_size > 0 and docx.endswith(".docx")
    summary = wf.summary()
    assert summary.count("\n[") == 9 and "Verdict" in summary
    assert [v["test"] for v in wf.violations()] == [
        s.key for s in wf._steps if s.status in ("caution", "fail")
    ]


def test_plot_draws_the_path(wf):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = wf.plot()
    assert ax.get_xlabel() == "Periods since treatment"
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


# --------------------------------------------------------------------- #
# sp.did_analysis: the diagnostics describe the reported estimate
# --------------------------------------------------------------------- #
def test_did_analysis_event_study_comes_from_the_main_estimator(panel, reference):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.did_analysis(panel, y="y", treat="g", time="time", id="unit")
    es = reference["dynamic"]
    pd.testing.assert_frame_equal(a.event_study_result.detail, es.detail)
    assert a.diagnostics["pretrend_pvalue"] == sp.pretrends_test(es)["pvalue"]
    assert "Callaway-Sant'Anna" in a.diagnostics["event_study_source"]


def test_did_analysis_reports_the_already_treated_share(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.did_analysis(
            panel, y="y", treat="g", time="time", id="unit", run_sensitivity=False
        )
    share = a.bacon["already_treated_control_weight_share"]
    assert share > 0
    assert f"{share:.1%} of the weight" in a.summary()
    assert "likely reliable" not in a.summary()


def test_did_analysis_passes_the_cluster_to_callaway_santanna():
    df = make_panel(n_units=60, seed=5)
    df["state"] = df["unit"] % 12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.did_analysis(
            df,
            y="y",
            treat="g",
            time="time",
            id="unit",
            cluster="state",
            run_bacon=False,
            run_event_study=False,
            run_sensitivity=False,
        )
        ref = sp.callaway_santanna(
            df, y="y", g="g", t="time", i="unit", clustervars="state"
        )
        unclustered = sp.callaway_santanna(df, y="y", g="g", t="time", i="unit")
    assert a.main_result.se == ref.se
    assert a.main_result.se != unclustered.se


def test_did_analysis_bjs_default_window_fits_a_short_panel():
    # Cohort 4 on periods 1..5 has three pre-periods: four leads (the old
    # default) left no reference period and the call raised.
    df = make_panel(n_periods=5, cohorts=(0, 3, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.did_analysis(df, y="y", treat="g", time="time", id="unit", method="bjs")
    leads = a.event_study_result.event_study.query("relative_time < 0")
    assert leads["relative_time"].min() == -2
