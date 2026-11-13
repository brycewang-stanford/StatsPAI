"""Reference parity for the defects found by the 2026-11 epidemiology probe.

A blind run of twenty epidemiology / biostatistics analyses against R
(same CSV bytes) compared every number a result prints. The numbers below
are the ones that did not agree, or that StatsPAI could not produce, before
the fixes of that round:

* ``sp.cox(strata=)`` reported Harrell's C over all pairs instead of the
  pairs within a stratum (R ``concordance``, Stata ``estat concordance``).
* ``CoxResult.baseline_hazard()`` of an Efron fit used Breslow increments
  (R ``basehaz``, Stata ``predict, basechazard`` use the Efron ones).
* ``sp.rmst`` took the smaller of the arms' largest times as the default
  ``tau`` and refused a larger one even when the shorter arm ends in an
  event (``survRM2::rmst2`` does neither).
* ``sp.overlap_weights`` reported the Kish effective sample size of the
  tilting function as ``effective_sample_size``.
* ``sp.svyglm`` printed a design effect of 1.0 for every coefficient
  (R ``svyglm(deff = TRUE)`` reports the design variance over the
  model-based one).
* ``KMResult.quantiles`` (median and quartiles with confidence limits),
  ``sp.logrank_test(rho=, strata=)``, ``sp.meta_analysis(method='REML',
  hksj=True)`` and ``sp.mr_mode(stderror='delta')`` are new.

Frozen reference: ``_fixtures/probe_epi_R.json`` from
``_fixtures/_generate_probe_epi_R.R`` (R 4.5.2; survival 3.8.3, survRM2
1.0.4, WeightIt 1.7.0, metafor 5.0.1, MendelianRandomization 0.10.0) on
``probe_epi_survival.csv`` / ``probe_epi_meta.csv`` written by
``_generate_probe_epi_data.py``, and the committed ``mr_ldl.csv``.

Tolerances are relative 1e-8 unless stated: every quantity is closed form
given the fitted coefficients, and the Cox coefficients of the two programs
agree to about 1e-10 (R stops its Newton iteration at a relative
log-likelihood change of 1e-9).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURES = pathlib.Path(__file__).parent / "_fixtures"
X = ["age", "ecog", "arm"]


@pytest.fixture(scope="module")
def surv():
    return pd.read_csv(_FIXTURES / "probe_epi_survival.csv")


@pytest.fixture(scope="module")
def ref():
    with open(_FIXTURES / "probe_epi_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def _nan(v):
    """R ``NA`` arrives as ``None`` (or the string 'NA')."""
    return np.array(
        [np.nan if (x is None or x == "NA") else float(x) for x in v], dtype=float
    )


# ---------------------------------------------------------------------------
# Kaplan-Meier quantiles and their confidence limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("conf_type", ["log", "log-log", "plain"])
def test_km_quantile_limits_match_survfit(surv, ref, conf_type):
    km = sp.kaplan_meier(
        surv, duration="time", event="event", group="arm", conf_type=conf_type
    )
    q = km.quantiles(ref["km_quantiles"]["probs"])
    r = ref["km_quantiles"]["by_conf_type"][conf_type]
    for g_idx, g in enumerate(ref["km_quantiles"]["groups"]):
        got = q[q["group"] == g]
        # times are observed event times: equality is exact. On this
        # fixture no step of S(t) equals 1 - p, so R's midpoint rule for
        # the estimate does not come into play.
        np.testing.assert_array_equal(
            got["time"].to_numpy(), _nan(r["quantile"][g_idx])
        )
        np.testing.assert_array_equal(
            got["ci_lower"].to_numpy(), _nan(r["lower"][g_idx])
        )
        np.testing.assert_array_equal(
            got["ci_upper"].to_numpy(), _nan(r["upper"][g_idx])
        )


def test_km_median_of_quantiles_is_median_survival(surv):
    km = sp.kaplan_meier(surv, duration="time", event="event", group="arm")
    q = km.quantiles(0.5).set_index("group")["time"].to_dict()
    assert q == km.median_survival


def test_km_summary_reports_censored_count_and_median_interval(surv):
    km = sp.kaplan_meier(
        surv, duration="time", event="event", group="arm", conf_type="log"
    )
    text = km.summary()
    n0 = int((surv["arm"] == 0).sum())
    d0 = int(surv.loc[surv["arm"] == 0, "event"].sum())
    assert f"Number censored  : {n0 - d0}" in text
    lo, hi = km.quantiles(0.5).query("group == '0'")[["ci_lower", "ci_upper"]].iloc[0]
    assert f"[{lo:.4f}, {hi:.4f}]" in text
    assert "log scale" in text


def test_km_quantiles_reject_probabilities_outside_the_unit_interval(surv):
    km = sp.kaplan_meier(surv, duration="time", event="event")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        km.quantiles([0.5, 1.0])


# ---------------------------------------------------------------------------
# Log-rank family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, kwargs, group",
    [
        ("rho0", {}, "arm"),
        ("rho1", {"rho": 1}, "arm"),
        ("rho05", {"rho": 0.5}, "arm"),
        ("strata_rho0", {"strata": "site"}, "arm"),
        ("strata_rho1", {"strata": "site", "rho": 1}, "arm"),
        ("ecog_rho1", {"rho": 1}, "ecog"),
    ],
)
def test_logrank_rho_and_strata_match_survdiff(surv, ref, key, kwargs, group):
    res = sp.logrank_test(surv, duration="time", event="event", group=group, **kwargs)
    r = ref["logrank"][key]
    assert res["test_statistic"] == pytest.approx(r["chisq"], rel=1e-10)
    levels = sorted(surv[group].unique())
    obs = [res["observed_events"][str(g)] for g in levels]
    exp = [res["expected_events"][str(g)] for g in levels]
    np.testing.assert_allclose(obs, r["obs"], rtol=1e-10)
    np.testing.assert_allclose(exp, r["exp"], rtol=1e-10)


def test_logrank_one_stratum_is_the_unstratified_test(surv):
    one = surv.assign(s=1)
    a = sp.logrank_test(one, duration="time", event="event", group="arm", rho=1)
    b = sp.logrank_test(
        one, duration="time", event="event", group="arm", rho=1, strata="s"
    )
    assert a["test_statistic"] == b["test_statistic"]
    assert b["n_strata"] == 1


# ---------------------------------------------------------------------------
# Cox: stratified concordance and the baseline hazard under each tie rule
# ---------------------------------------------------------------------------


def test_stratified_cox_concordance_counts_pairs_within_strata(surv, ref):
    fit = sp.cox("time ~ age + ecog + arm", data=surv, event="event", strata="site")
    np.testing.assert_allclose(
        fit.params.to_numpy(), ref["cox"]["coef_strata"], rtol=1e-8
    )
    assert fit.concordance == pytest.approx(ref["cox"]["strata_concordance"], rel=1e-10)
    # without strata the two definitions coincide
    plain = sp.cox("time ~ age + ecog + arm", data=surv, event="event")
    assert plain.concordance == pytest.approx(
        ref["cox"]["plain_concordance"], rel=1e-10
    )


def _hazard_at(table: pd.DataFrame, times) -> np.ndarray:
    """Right-continuous step function of the cumulative hazard."""
    t = table["time"].to_numpy()
    h = table["baseline_cumhaz"].to_numpy()
    return h[np.searchsorted(t, np.asarray(times, dtype=float), side="right") - 1]


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("basehaz_efron", {}),
        ("basehaz_breslow", {"ties": "breslow"}),
        ("basehaz_weights", {"weights": "w"}),
        ("basehaz_entry", {"start": "entry"}),
    ],
)
def test_baseline_hazard_follows_the_tie_rule_of_the_fit(surv, ref, key, kwargs):
    fit = sp.cox("time ~ age + ecog + arm", data=surv, event="event", **kwargs)
    r = ref["cox"][key]
    got = _hazard_at(fit.baseline_hazard(), r["time"])
    # basehaz lists censoring times too; the step function is evaluated there
    np.testing.assert_allclose(got, r["hazard"], rtol=1e-8, atol=1e-14)


def test_baseline_hazard_of_a_stratified_fit_is_labelled_by_stratum(surv, ref):
    fit = sp.cox("time ~ age + ecog + arm", data=surv, event="event", strata="site")
    bh = fit.baseline_hazard()
    r = ref["cox"]["basehaz_strata"]
    r_time = np.asarray(r["time"], dtype=float)
    r_haz = np.asarray(r["hazard"], dtype=float)
    r_str = np.asarray(r["strata"])
    assert sorted(bh["strata"].unique()) == [0, 1, 2]
    for level in (0, 1, 2):
        sel = r_str == f"site={level}"
        got = _hazard_at(bh[bh["strata"] == level], r_time[sel])
        np.testing.assert_allclose(got, r_haz[sel], rtol=1e-8, atol=1e-14)


def test_efron_and_breslow_baselines_coincide_without_tied_deaths():
    rng = np.random.default_rng(3)
    n = 150
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "time": rng.exponential(np.exp(-0.4 * x)) + rng.uniform(0, 1e-6, n),
            "event": (rng.random(n) < 0.8).astype(int),
        }
    )
    assert df["time"].is_unique
    efron = sp.cox("time ~ x", data=df, event="event", ties="efron")
    breslow = sp.cox("time ~ x", data=df, event="event", ties="breslow")
    np.testing.assert_allclose(
        efron.baseline_hazard()["baseline_cumhaz"],
        breslow.baseline_hazard()["baseline_cumhaz"],
        rtol=1e-10,
    )


def test_cox_summary_prints_hazard_ratios_and_counts(surv):
    fit = sp.cox("time ~ age + ecog + arm", data=surv, event="event")
    text = str(fit.summary())
    assert f"Observations: {len(surv)}   Events: {int(surv['event'].sum())}" in text
    assert "Hazard ratios, exp(coef):" in text
    hr = float(np.exp(fit.params["arm"]))
    lo, hi = np.exp(fit.conf_int().loc["arm"].to_numpy())
    row = next(
        ln for ln in text.split("\n") if ln.startswith("arm") and f"{hr:.4f}" in ln
    )
    assert f"{lo:.4f}" in row and f"{hi:.4f}" in row
    quiet = sp.cox(
        "time ~ age + ecog + arm", data=surv, event="event", hazard_ratio=False
    )
    assert "Hazard ratios" not in str(quiet.summary())


# ---------------------------------------------------------------------------
# RMST: the default tau and the largest admissible one
# ---------------------------------------------------------------------------


def test_rmst_default_tau_is_survrm2s(surv, ref):
    r = ref["rmst"]["default"]
    # arm 0 ends in an event at 68, arm 1 in a censored row at 72: the
    # curve of arm 0 is zero from 68 on, so tau can go up to 72
    assert ref["rmst"]["max_time"] == [68, 72]
    res = sp.rmst(surv, "time", "event", group="arm")
    assert res.tau == r["tau"] == 72
    tab = res.table.set_index("group")
    np.testing.assert_allclose(
        tab.loc["0", ["rmst", "se"]].astype(float), r["rmst0"], rtol=1e-10
    )
    np.testing.assert_allclose(
        tab.loc["1", ["rmst", "se"]].astype(float), r["rmst1"], rtol=1e-10
    )
    got = res.contrasts[["estimate", "ci_lower", "ci_upper", "p_value"]].to_numpy()
    np.testing.assert_allclose(got, np.asarray(r["contrasts"]), rtol=1e-9)


def test_rmst_tau_between_the_arms_last_times(surv, ref):
    r = ref["rmst"]["tau70"]
    res = sp.rmst(surv, "time", "event", group="arm", tau=70)
    tab = res.table.set_index("group")
    np.testing.assert_allclose(
        tab.loc["0", ["rmst", "se"]].astype(float), r["rmst0"], rtol=1e-10
    )
    np.testing.assert_allclose(
        tab.loc["1", ["rmst", "se"]].astype(float), r["rmst1"], rtol=1e-10
    )
    # the infinitesimal-jackknife variance equals Greenwood's without weights
    ij = sp.rmst(surv, "time", "event", group="arm", tau=70, se_method="ij")
    np.testing.assert_allclose(
        ij.table["se"].to_numpy(), res.table["se"].to_numpy(), rtol=1e-9
    )


def test_rmst_refuses_tau_beyond_a_censored_last_observation(surv):
    with pytest.raises(sp.exceptions.DataInsufficient) as err:
        sp.rmst(surv, "time", "event", group="arm", tau=73)
    assert "Use tau <= 72" in str(err.value.recovery_hint)


# ---------------------------------------------------------------------------
# Overlap-type weights: effective sample size of the weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimand, key",
    [
        ("ATE", "ATE"),
        ("ATT", "ATT"),
        ("ATC", "ATC"),
        ("ATO", "ATO"),
        ("matching", "ATM"),
    ],
)
def test_overlap_weights_ess_is_the_ess_of_the_weights(surv, ref, estimand, key):
    res = sp.overlap_weights(
        surv,
        y="time",
        treat="arm",
        covariates=["age", "ecog", "site"],
        estimand=estimand,
        n_bootstrap=2,
        seed=0,
    )
    r = ref["weightit_ess"][key]
    info = res.model_info
    # WeightIt stops its logit at a looser tolerance than the Newton fit here
    assert info["ess_treated"] == pytest.approx(r["treated"], rel=1e-7)
    assert info["ess_control"] == pytest.approx(r["control"], rel=1e-7)
    assert info["effective_sample_size"] == pytest.approx(
        r["treated"] + r["control"], rel=1e-7
    )
    assert list(res.detail["variable"]) == ["age", "ecog", "site"]


# ---------------------------------------------------------------------------
# svyglm: design effects of the coefficients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, formula, family, design_kw",
    [
        ("gaussian_strata", "time ~ age + ecog + arm", "gaussian", {"strata": "site"}),
        ("gaussian_cluster", "time ~ ecog + arm", "gaussian", {"cluster": "age"}),
        ("binomial_strata", "event ~ age + ecog + arm", "binomial", {"strata": "site"}),
        ("poisson_strata", "time ~ age + ecog + arm", "poisson", {"strata": "site"}),
    ],
)
def test_svyglm_design_effects_match_survey(surv, ref, key, formula, family, design_kw):
    design = sp.svydesign(surv, weights="w", **design_kw)
    fit = sp.svyglm(formula, design, family=family)
    r = ref["svyglm_deff"][key]
    np.testing.assert_allclose(fit.estimate.to_numpy(), r["coef"], rtol=1e-8)
    np.testing.assert_allclose(fit.std_error.to_numpy(), r["se"], rtol=1e-8)
    np.testing.assert_allclose(fit.deff.to_numpy(), r["deff"], rtol=1e-8)
    assert "DEFF" in fit.summary().columns


def test_svyglm_design_effect_is_one_under_simple_random_sampling():
    # equal weights, no strata or clusters, homoskedastic errors: the design
    # variance is the HC0 sandwich, which estimates the model-based variance
    rng = np.random.default_rng(8)
    n = 20000
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": 1 + 2 * x + rng.normal(size=n), "w": 1.0})
    fit = sp.svyglm("y ~ x", sp.svydesign(df, weights="w"))
    # sd of the ratio is about sqrt(2 / n) for the intercept and
    # sqrt(8 / n) for the slope (kurtosis of x); 0.1 is five of the latter
    np.testing.assert_allclose(fit.deff.to_numpy(), 1.0, atol=0.1)


def test_svyglm_domain_fit_reports_no_design_effect(surv):
    design = sp.svydesign(surv, weights="w", strata="site")
    fit = sp.svyglm("time ~ age + ecog", design, subpop=(surv["arm"] == 1).to_numpy())
    assert fit.deff.isna().all()
    assert "DEFF" not in fit.summary().columns


# ---------------------------------------------------------------------------
# Meta-analysis: REML and the Knapp-Hartung adjustment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def meta_data():
    return pd.read_csv(_FIXTURES / "probe_epi_meta.csv")


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("REML", {"method": "REML"}),
        ("REML_knha", {"method": "REML", "hksj": True}),
        ("DL", {"method": "DL"}),
        ("DL_knha", {"method": "DL", "hksj": True}),
    ],
)
def test_meta_analysis_matches_metafor(meta_data, ref, key, kwargs):
    res = sp.meta_analysis(meta_data["yi"], meta_data["sei"], **kwargs)
    r = ref["meta"][key]
    assert res.tau2 == pytest.approx(r["tau2"], rel=1e-9)
    assert res.estimate == pytest.approx(r["beta"], rel=1e-9)
    assert res.se == pytest.approx(r["se"], rel=1e-9)
    np.testing.assert_allclose(res.ci, r["ci"], rtol=1e-9)
    assert res.p_value == pytest.approx(r["pval"], rel=1e-8)
    assert res.q == pytest.approx(r["QE"], rel=1e-10)
    assert res.random_se == pytest.approx(r["se"], rel=1e-9)


def test_reml_tau2_solves_the_restricted_score_equation(meta_data):
    y = meta_data["yi"].to_numpy()
    v = meta_data["sei"].to_numpy() ** 2
    res = sp.meta_analysis(y, np.sqrt(v), method="REML")
    w = 1.0 / (v + res.tau2)
    mu = np.sum(w * y) / np.sum(w)
    score = np.sum(w**2 * (y - mu) ** 2) - np.sum(w) + np.sum(w**2) / np.sum(w)
    assert abs(score) <= 1e-9 * np.sum(w)


def test_reml_tau2_is_zero_for_homogeneous_studies():
    res = sp.meta_analysis([0.2, 0.2, 0.2, 0.2], [0.1, 0.2, 0.15, 0.3], method="REML")
    assert res.tau2 == 0.0
    assert res.estimate == pytest.approx(res.fixed_estimate, rel=1e-12)


def test_hksj_is_refused_for_the_fixed_effect_model(meta_data):
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.meta_analysis(meta_data["yi"], meta_data["sei"], method="fixed", hksj=True)


# ---------------------------------------------------------------------------
# Mode-based MR: delta-method ratio standard errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weighting, r_name", [("weighted", "weighted"), ("simple", "unweighted")]
)
@pytest.mark.parametrize("stderror", ["simple", "delta"])
def test_mr_mode_delta_standard_error(ref, weighting, r_name, stderror):
    g = pd.read_csv(_FIXTURES / "mr_ldl.csv")
    res = sp.mr_mode(
        g["bx"].to_numpy(),
        g["by"].to_numpy(),
        g["bxse"].to_numpy(),
        g["byse"].to_numpy(),
        method=weighting,
        stderror=stderror,
        n_boot=20000,
        seed=20261113,
    )
    r = ref["mr_mbe"][f"{r_name}_{stderror}"]
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-12)
    # two independent bootstraps of 20,000 draws; the MAD of each is within
    # about 2% of its limit, the delta and simple values differ by a factor
    # of 2 to 2.7
    assert res.se == pytest.approx(r["se"], rel=0.06)


def test_mr_dispatcher_accepts_the_four_arrays_for_every_diagnostic():
    g = pd.read_csv(_FIXTURES / "mr_ldl.csv")
    arrays = dict(
        beta_exposure=g["bx"].to_numpy(),
        se_exposure=g["bxse"].to_numpy(),
        beta_outcome=g["by"].to_numpy(),
        se_outcome=g["byse"].to_numpy(),
    )
    egger = sp.mr("egger", **arrays)
    pleio = sp.mr("pleiotropy_egger", **arrays)
    assert pleio.intercept == pytest.approx(float(egger["intercept"]), rel=1e-12)
    fstat = sp.mr("f_statistic", **arrays)
    np.testing.assert_allclose(
        fstat.per_snp_F, (arrays["beta_exposure"] / arrays["se_exposure"]) ** 2
    )
    loo = sp.mr("leave_one_out", **arrays)
    direct = sp.mr_leave_one_out(
        arrays["beta_exposure"], arrays["beta_outcome"], arrays["se_outcome"]
    )
    assert loo.summary() == direct.summary()
    with pytest.raises(TypeError):
        sp.mr("ivw", not_an_argument=1, **arrays)
