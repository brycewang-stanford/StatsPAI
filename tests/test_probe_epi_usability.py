"""Labels, messages and discoverability fixed by the 2026-11 epidemiology probe.

Each test states what a reader of the output is entitled to: a label that
describes the number next to it, a typed error with a way out, an option
that the schema an agent reads actually lists.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)


@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(11)
    n = 600
    x = rng.normal(size=n)
    a = (rng.random(n) < 1 / (1 + np.exp(-1.2 * x))).astype(int)
    y = 1.0 * a + 2.0 * x + rng.normal(size=n)
    return pd.DataFrame({"x": x, "a": a, "y": y, "age": rng.integers(30, 70, n)})


def _protocol(assignment="observational emulation"):
    return sp.target_trial_protocol(
        eligibility="age >= 40",
        treatment_strategies=["treat", "do not treat"],
        assignment=assignment,
        time_zero="baseline",
        followup_end="1 year",
        outcome="y",
        analysis_plan="IPW",
        baseline_covariates=["x"],
    )


# ---------------------------------------------------------------------------
# Target-trial emulation
# ---------------------------------------------------------------------------


def test_emulate_without_weights_is_labelled_unadjusted_and_warns(obs):
    with pytest.warns(AssumptionWarning, match="unadjusted difference in means"):
        res = sp.target_trial_emulate(
            _protocol(), obs, outcome_col="y", treatment_col="a"
        )
    elig = obs[obs["age"] >= 40]
    crude = elig.loc[elig.a == 1, "y"].mean() - elig.loc[elig.a == 0, "y"].mean()
    assert res.estimate == pytest.approx(crude, rel=1e-12)
    assert res.method.startswith("unadjusted difference in means")
    assert "IPW" not in res.method
    assert f"Method: {res.method}" in res.summary()


def test_emulate_with_weights_reports_the_analysis_plan_and_the_iptw_mean(obs):
    ps = sp.propensity_score(obs, treat="a", covariates=["x"])
    w = np.asarray(sp.ps_weights(ps, obs["a"], "ATE"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = sp.target_trial_emulate(
            _protocol(), obs, outcome_col="y", treatment_col="a", weights=w
        )
    assert res.method == "IPW"
    keep = (obs["age"] >= 40).to_numpy()
    a, y, wk = obs["a"].to_numpy()[keep], obs["y"].to_numpy()[keep], w[keep]
    hajek = np.average(y[a == 1], weights=wk[a == 1]) - np.average(
        y[a == 0], weights=wk[a == 0]
    )
    assert res.estimate == pytest.approx(hajek, rel=1e-12)


def test_emulate_randomized_protocol_does_not_warn(obs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = sp.target_trial_emulate(
            _protocol("randomization"), obs, outcome_col="y", treatment_col="a"
        )
    assert res.method.startswith("unadjusted difference in means")


def test_emulate_counts_ineligible_subjects_and_says_so(obs):
    res = sp.target_trial_emulate(
        _protocol("randomization"), obs, outcome_col="y", treatment_col="a"
    )
    n_inelig = int((obs["age"] < 40).sum())
    assert res.n_excluded == res.n_excluded_immortal == n_inelig
    assert f"n excluded (not eligible at time zero): {n_inelig}" in res.summary()
    for fmt in ("markdown", "text", "latex", "jama", "target"):
        report = sp.target_trial_report(res, fmt=fmt).lower()
        assert "prevent immortal" not in report, fmt
        assert "immortal-time prevention" not in report, fmt


def test_emulate_refuses_missing_outcomes_and_non_binary_treatment(obs):
    proto = _protocol("randomization")
    holes = obs.copy()
    holes.loc[holes.index[:5], "y"] = np.nan
    holes.loc[holes.index[:5], "age"] = 50
    with pytest.raises(DataInsufficient) as err:
        sp.target_trial_emulate(proto, holes, outcome_col="y", treatment_col="a")
    assert err.value.diagnostics["n_missing_outcome"] == 5
    recoded = obs.assign(a=obs["a"] + 1)
    with pytest.raises(MethodIncompatibility):
        sp.target_trial_emulate(proto, recoded, outcome_col="y", treatment_col="a")
    with pytest.raises(MethodIncompatibility):
        sp.target_trial_emulate(
            proto, obs, outcome_col="y", treatment_col="a", weights=np.ones(3)
        )


# ---------------------------------------------------------------------------
# Result labels
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def surv_rows():
    rng = np.random.default_rng(21)
    n = 300
    x = rng.normal(size=n)
    t = np.ceil(rng.exponential(10 * np.exp(-0.5 * x)))
    c = np.ceil(rng.uniform(2, 30, n))
    return pd.DataFrame(
        {
            "x": x,
            "time": np.minimum(t, c),
            "status": (t <= c).astype(int),
            "entry": np.zeros(n),
        }
    )


@pytest.mark.parametrize(
    "formula",
    ["time + status ~ x", "Surv(time, status) ~ x", "Surv(entry, time, status) ~ x"],
)
def test_cox_reads_the_event_from_the_formula(surv_rows, formula):
    named = sp.cox("time ~ x", data=surv_rows, event="status")
    fit = sp.cox(formula, data=surv_rows)
    # entry at 0 for every row leaves the risk sets unchanged
    np.testing.assert_allclose(
        fit.params.to_numpy(), named.params.to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        fit.std_errors.to_numpy(), named.std_errors.to_numpy(), rtol=1e-12
    )


def test_survreg_reads_the_event_from_the_formula(surv_rows):
    named = sp.survreg("time ~ x", data=surv_rows, event="status")
    fit = sp.survreg("Surv(time, status) ~ x", data=surv_rows)
    np.testing.assert_array_equal(fit.params.to_numpy(), named.params.to_numpy())
    with pytest.raises(MethodIncompatibility, match="delayed entry"):
        sp.survreg("Surv(entry, time, status) ~ x", data=surv_rows)


def test_cox_formula_errors_say_what_to_write(surv_rows):
    with pytest.raises(MethodIncompatibility) as err:
        sp.cox("time ~ x", data=surv_rows)
    assert "Surv(time, status)" in err.value.recovery_hint
    with pytest.raises(MethodIncompatibility, match="event column"):
        sp.cox("Surv(time, status) ~ x", data=surv_rows, event="entry")
    with pytest.raises(MethodIncompatibility, match="left-hand side"):
        sp.cox("Surv(time, status > 0) ~ x", data=surv_rows)


def test_finegray_result_has_std_errors():
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    t1 = rng.exponential(1 / np.exp(0.5 * x))
    t2 = rng.exponential(1.0, n)
    c = rng.exponential(3.0, n)
    t = np.minimum.reduce([t1, t2, c])
    status = np.select([t == t1, t == t2], [1, 2], 0)
    df = pd.DataFrame({"x": x, "time": t, "status": status})
    fit = sp.finegray(df, duration="time", event="status", x=["x"], cause=1)
    np.testing.assert_array_equal(fit.std_errors, fit.bse)
    np.testing.assert_allclose(fit.std_errors, np.sqrt(np.diag(fit.vcov)))


def test_bal_tab_names_the_group_whose_weights_sum_to_zero(obs):
    w = np.where(obs["a"] == 1, 1.0, 0.0)
    with pytest.raises(MethodIncompatibility) as err:
        sp.bal_tab(obs, "a", ["x"], weights=w)
    assert err.value.diagnostics["groups_with_zero_weight"] == ["0"]


def test_match_schema_lists_the_options_a_referee_asks_for():
    props = sp.function_schema("match")["parameters"]["properties"]
    for name in (
        "distance",
        "estimand",
        "n_matches",
        "caliper",
        "replace",
        "bias_correction",
        "common_support",
        "alpha",
    ):
        assert name in props, name
    import inspect

    from statspai.matching.match import MatchEstimator

    accepted = inspect.signature(MatchEstimator.__init__).parameters
    for name in (
        "distance",
        "estimand",
        "n_matches",
        "caliper",
        "replace",
        "bias_correction",
        "common_support",
        "alpha",
    ):
        assert name in accepted, name
    assert props["replace"]["default"] is accepted["replace"].default
    assert props["estimand"]["default"] == accepted["estimand"].default


def test_presso_summary_bounds_a_zero_simulated_p_value():
    rng = np.random.default_rng(0)
    n = 30
    bx = rng.uniform(0.1, 0.4, n)
    sx = np.full(n, 0.01)
    sy = np.full(n, 0.01)
    by = 0.5 * bx + rng.normal(0, 0.05, n)  # far more spread than sy allows
    res = sp.mr_presso(bx, by, sx, sy, n_boot=600, seed=1)
    assert res.global_test_pvalue == 0.0
    assert "p-value < 0.001667 (0 of 600 draws)" in res.summary()


def test_search_finds_meta_analysis_and_aft_by_the_words_a_reviewer_uses():
    for query in (
        "egger test publication bias",
        "funnel plot",
        "forest plot",
        "meta-analysis heterogeneity",
    ):
        names = [h["name"] for h in sp.search_functions(query, limit=3)]
        assert "meta_analysis" in names, query
    names = [
        h["name"] for h in sp.search_functions("weibull survival regression", limit=3)
    ]
    assert "aft" in names
