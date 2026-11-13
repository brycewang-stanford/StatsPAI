"""Defects found by the 2026-11-13 agent probe, each pinned by a true statement.

The probe ran sixteen analyses through the MCP server and twenty-nine
functions through ``sp.search_functions`` / ``sp.describe_function`` /
``sp.function_schema`` as an agent with no knowledge of the source. Every
test here states something that must hold (an identity with the direct
Python call, a number computed by hand from the data, or "the call the
layer itself recommends runs"), not that an output changed.
"""

from __future__ import annotations

import inspect
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def obs():
    rng = np.random.default_rng(20261113)
    n = 1500
    x1, x2, x3 = rng.normal(size=(3, n))
    ps = 1 / (1 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    d = (rng.random(n) < ps).astype(int)
    y = 1 + 1.5 * d + x1 + 0.5 * x2 + 0.2 * x3 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "x3": x3})


@pytest.fixture(scope="module")
def separated():
    rng = np.random.default_rng(7)
    n = 1500
    x1, x2 = rng.normal(size=(2, n))
    d = (x1 > 0).astype(int)  # treatment is a function of x1: no overlap
    y = 1 + 1.5 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def iv_frame():
    rng = np.random.default_rng(11)
    n = 800
    z, u, x = rng.normal(size=(3, n))
    d = 0.8 * z + u + rng.normal(size=n)
    y = 1 + 2 * d + x + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x})


@pytest.fixture(scope="module")
def did2x2():
    rng = np.random.default_rng(3)
    n = 300
    tr = (rng.random(n) < 0.5).astype(int)
    a = rng.normal(size=n)
    rows = []
    for t in (0, 1):
        rows.append(
            pd.DataFrame(
                {
                    "unit": np.arange(n),
                    "post": t,
                    "treated": tr,
                    "y": 1 + a + 0.5 * t + 0.3 * tr + 2.0 * tr * t + rng.normal(size=n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def staggered():
    rng = np.random.default_rng(5)
    n = 240
    g = rng.choice([0, 2013, 2016], size=n, p=[0.4, 0.3, 0.3])
    a = rng.normal(size=n)
    rows = []
    for year in range(2010, 2020):
        on = (g > 0) & (year >= g)
        rows.append(
            pd.DataFrame(
                {
                    "id": np.arange(n),
                    "year": year,
                    "first_treat": g,
                    "d": on.astype(int),
                    "y": a + 0.2 * (year - 2010) + on * 1.0 + rng.normal(size=n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def synth_panel():
    rng = np.random.default_rng(9)
    f = np.cumsum(rng.normal(size=30))
    rows = []
    for i in range(15):
        lam, mu = rng.uniform(0.5, 1.5), rng.normal(10, 2)
        for k, year in enumerate(range(1980, 2010)):
            hit = i == 0 and year >= 2000
            rows.append(
                {
                    "region": f"R{i + 1:02d}",
                    "year": year,
                    "gdp": mu + lam * f[k] + rng.normal(scale=0.3) - 5 * hit,
                    "treated": int(hit),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def firm_panel():
    rng = np.random.default_rng(13)
    n = 120
    a = rng.normal(size=n)
    rows = []
    for year in range(2001, 2009):
        x = a + rng.normal(size=n)
        rows.append(
            pd.DataFrame(
                {
                    "firm": np.arange(n),
                    "year": year,
                    "invest": x,
                    "size": rng.normal(size=n),
                    "profit": 2 * a + 0.7 * x + rng.normal(size=n),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
#  Balance: the violation reads the matched sample, not the raw one
# --------------------------------------------------------------------------- #
def test_balance_violation_reports_the_post_matching_smd(obs):
    from statspai.core._agent_summary import _max_covariate_smd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(obs, y="y", treat="d", covariates=["x1", "x2", "x3"])
    mi = r.model_info
    after = float(mi["balance_after"]["smd"].abs().max())
    before = float(
        mi["balance_before"]
        .loc[mi["balance_before"]["variable"] != "propensity_score", "smd"]
        .abs()
        .max()
    )
    assert before > 0.25 > after  # the DGP is confounded, the match balances it
    assert _max_covariate_smd(mi) == pytest.approx(after, abs=1e-12)
    assert [v for v in r.violations() if v["test"] == "balance"] == []


def test_balance_violation_still_fires_on_the_matched_sample(separated):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(separated, y="y", treat="d", covariates=["x1", "x2"])
    after = float(r.model_info["balance_after"]["smd"].abs().max())
    (v,) = [v for v in r.violations() if v["test"] == "balance"]
    assert v["value"] == pytest.approx(after, abs=1e-12)


# --------------------------------------------------------------------------- #
#  sp.audit
# --------------------------------------------------------------------------- #
def _checks(report):
    return {c["name"]: c for c in report["checks"]}


def test_audit_reads_the_placebo_inference_and_prefit_a_synth_fit_holds(synth_panel):
    r = sp.synth(
        synth_panel,
        outcome="gdp",
        unit="region",
        time="year",
        treated_unit="R01",
        treatment_time=2000,
    )
    checks = _checks(sp.audit(r))
    assert checks["placebo_inference"]["status"] == "passed"
    pre = synth_panel.query("region == 'R01' and year < 2000")["gdp"]
    ratio = r.model_info["pre_treatment_rmse"] / pre.std(ddof=1)
    # pre_treatment_rmse is stored rounded to 6 decimals
    assert checks["pretreatment_fit"]["value"] == pytest.approx(ratio, rel=1e-5)
    assert checks["pretreatment_fit"]["status"] == (
        "passed" if ratio < 0.5 else "failed"
    )


def test_audit_reads_post_matching_balance(obs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(obs, y="y", treat="d", covariates=["x1", "x2", "x3"])
    chk = _checks(sp.audit(r))["balance_after"]
    assert chk["status"] == "passed"
    assert chk["value"] == pytest.approx(r.model_info["max_abs_smd_after"])


def test_audit_family_of_ols_and_sun_abraham(firm_panel, staggered):
    ols = sp.feols(
        "profit ~ invest + size | firm + year", data=firm_panel, cluster="firm"
    )
    assert sp.audit(ols)["method_family"] == "regression"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sa = sp.sun_abraham(staggered, y="y", g="first_treat", t="year", i="id")
    report = sp.audit(sa)
    assert report["method_family"] == "did"
    assert "parallel_trends" in _checks(report)


def test_audit_two_period_did_suggests_nothing_that_cannot_run(did2x2, staggered):
    r = sp.did(did2x2, y="y", treat="treated", time="post")
    report = sp.audit(r)
    checks = _checks(report)
    # parallel trends is untested, and the audit says so ...
    for name in ("parallel_trends", "honest_did"):
        assert checks[name]["status"] == "missing"
        # ... without pointing at a function that fails on two periods
        assert checks[name]["suggest_function"] is None
        assert "two-period" in checks[name]["reason"]
    assert checks["bacon_decomposition"]["status"] == "not_applicable"
    s = report["summary"]
    assert s["passed"] + s["failed"] + s["missing"] == s["n_total"] == 2
    # the claim is true: the suggested functions do fail on this fit
    with pytest.raises(sp.exceptions.StatsPAIError):
        sp.honest_did(r)
    # a design with pre-periods keeps its suggestion
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(staggered, y="y", g="first_treat", t="year", i="id")
    assert _checks(sp.audit(cs))["honest_did"]["suggest_function"] == "sp.honest_did"


def test_audit_evidence_marks_a_run_check_and_never_overrides_the_fit(staggered):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(staggered, y="y", g="first_treat", t="year", i="id")
    base = _checks(sp.audit(cs))
    assert base["honest_did"]["status"] == "missing"
    stored_p = cs.model_info["pretrend_test"]["pvalue"]
    with_ev = _checks(
        sp.audit(
            cs,
            evidence={
                "honest_did": {"has_run": True},
                "pretrend_test": {"pvalue": 0.99},
            },
        )
    )
    assert with_ev["honest_did"]["status"] == "passed"
    # the fit's own pre-trend test is what the audit reports
    assert with_ev["parallel_trends"]["value"] == pytest.approx(stored_p)


# --------------------------------------------------------------------------- #
#  next_calls: executable, and about the model that was fitted
# --------------------------------------------------------------------------- #
def _calls(tool, args, result=None):
    from statspai.agent._enrichment import build_next_calls

    return {
        c["tool"]: c
        for c in build_next_calls(tool, result_id="r_x", base_args=args, result=result)
    }


def test_iv_followups_carry_the_exogenous_controls(iv_frame):
    calls = _calls("ivreg", {"formula": "y ~ x + (d ~ z)", "data_id": "d_1"})
    ar = calls["anderson_rubin_test"]
    assert ar["ready"] is True
    assert ar["arguments"]["endog"] == "d"
    assert ar["arguments"]["instruments"] == ["z"]
    assert ar["arguments"]["exog"] == ["x"]
    args = {k: v for k, v in ar["arguments"].items() if k != "data_id"}
    out = sp.anderson_rubin_test(data=iv_frame, **args)
    fit = sp.ivreg("y ~ x + (d ~ z)", data=iv_frame)
    # the diagnostic describes the fitted model: same 2SLS coefficient
    assert out["beta_2sls"] == pytest.approx(float(fit.params["d"]), rel=1e-10)


def test_formula_roles():
    from statspai.agent._enrichment import _formula_roles

    assert _formula_roles({"formula": "y ~ (d1 + d2 ~ z1 + z2 + z3) + x1"}) == {
        "y": "y",
        "endog": ["d1", "d2"],
        "instruments": ["z1", "z2", "z3"],
        "exog": ["x1"],
    }
    assert _formula_roles({"formula": "y ~ a + b"}) == {"y": "y", "x": ["a", "b"]}
    # a term that is not a column is not turned into a column list
    assert _formula_roles({"formula": "y ~ a + C(region)"}) == {"y": "y"}
    assert _formula_roles({"fml": "y ~ a | firm"}) == {"y": "y"}


def test_fuzzy_rd_followups_keep_the_fuzzy_design():
    calls = _calls(
        "rdrobust", {"y": "y", "x": "score", "c": 0, "fuzzy": "takeup", "data_id": "d"}
    )
    assert "rdsensitivity" not in calls  # local-randomisation scan: sharp only
    for tool in ("rdbwsensitivity", "rdplacebo"):
        assert calls[tool]["ready"] is True
        assert calls[tool]["arguments"]["fuzzy"] == "takeup"


def test_synth_followups_reuse_the_fit_arguments():
    args = {
        "outcome": "gdp",
        "unit": "region",
        "time": "year",
        "treated_unit": "R01",
        "treatment_time": 2000,
        "data_id": "d",
    }
    calls = _calls("synth", args)
    for tool in ("synth_sensitivity", "robust_synth"):
        assert calls[tool]["ready"] is True
        for key in ("outcome", "treated_unit", "treatment_time"):
            assert calls[tool]["arguments"][key] == args[key]
    sdid = _calls(
        "sdid",
        {
            "y": "gdp",
            "unit": "region",
            "time": "year",
            "treat_unit": "R01",
            "treat_time": 2000,
            "data_id": "d",
        },
    )
    assert sdid["synthdid_placebo"]["ready"] is True


def test_honest_did_is_offered_only_when_the_fit_has_a_pre_period(did2x2, staggered):
    r2 = sp.did(did2x2, y="y", treat="treated", time="post")
    assert "honest_did_from_result" not in _calls(
        "did", {"y": "y", "treat": "treated", "time": "post"}, result=r2
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(staggered, y="y", g="first_treat", t="year", i="id")
    calls = _calls(
        "callaway_santanna",
        {"y": "y", "g": "first_treat", "t": "year", "i": "id"},
        result=cs,
    )
    assert calls["honest_did_from_result"]["ready"] is True
    # and the call it stands for runs on that fit
    assert len(sp.honest_did(cs)) > 0


def test_every_followup_template_names_something_callable():
    from statspai.agent._enrichment import _FOLLOWUP_BY_TOOL, _tool_schema_index

    known = set(_tool_schema_index())
    for tool, templates in _FOLLOWUP_BY_TOOL.items():
        for t in templates:
            name = t["tool"]
            assert name in known or callable(getattr(sp, name, None)), (tool, name)


def test_dml_spec_curve_followup_is_the_adjusted_and_unadjusted_pair():
    calls = _calls(
        "dml", {"y": "y", "treat": "d", "covariates": ["x1", "x2"], "data_id": "d"}
    )
    sc = calls["spec_curve"]
    assert sc["ready"] is True
    assert sc["arguments"]["x"] == "d"
    assert sc["arguments"]["controls"] == [[], ["x1", "x2"]]


# --------------------------------------------------------------------------- #
#  Returned values an agent can read
# --------------------------------------------------------------------------- #
def test_plot_functions_come_back_as_a_png(obs):
    import matplotlib

    from statspai.agent.auto_dispatch import _figure_payload

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    out = _figure_payload((fig, ax), "rdplot")
    assert out["mime_type"] == "image/png"
    assert out["_plot_png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert out["image_bytes"] == len(out["_plot_png"])
    assert out["figsize"] == [4.0, 3.0]
    # anything that is not a figure is left to the ordinary serializer
    assert _figure_payload({"estimate": 1.0}, "regress") is None
    assert _figure_payload((1.0, 2.0), "x") is None


def test_weak_iv_confidence_set_serialises_its_interval(iv_frame):
    from statspai.agent.tools._helpers import _default_serializer

    ci = sp.anderson_rubin_ci("y", "d", ["z"], exog=["x"], data=iv_frame)
    d = ci.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["lower"] == pytest.approx(ci.lower)
    assert d["upper"] == pytest.approx(ci.upper)
    assert d["intervals"] == [[pytest.approx(ci.lower), pytest.approx(ci.upper)]]
    assert d["ci"] == [d["lower"], d["upper"]]
    # the true coefficient is 2 and the instrument is strong
    assert d["lower"] < 2.0 < d["upper"]
    payload = _default_serializer(ci)
    assert payload["lower"] == d["lower"] and payload["upper"] == d["upper"]


def test_result_card_names_the_distribution_the_interval_used(did2x2):
    from scipy import stats

    r = sp.did(did2x2, y="y", treat="treated", time="post")
    df = len(did2x2) - 4
    assert sp.result_card(r)["inference"]["reference_distribution"] == f"t({df})"
    q = stats.t.ppf(0.975, df)
    assert r.ci[0] == pytest.approx(r.estimate - q * r.se, rel=1e-10)
    assert r.ci[1] == pytest.approx(r.estimate + q * r.se, rel=1e-10)


def test_hdfe_result_payload_caps_every_per_observation_array(firm_panel):
    from statspai.agent.tools._helpers import _default_serializer, is_obs_array_key

    assert is_obs_array_key("fitted_within") and is_obs_array_key("residuals")
    assert not is_obs_array_key("coefficients")
    r = sp.hdfe_ols("profit ~ invest + size | firm + year", data=firm_panel)
    payload = _default_serializer(r)
    n = len(firm_panel)
    long_lists = [k for k, v in payload.items() if isinstance(v, list) and len(v) >= n]
    assert long_lists == []
    assert len(json.dumps(payload, default=str)) < 20_000


# --------------------------------------------------------------------------- #
#  Citations
# --------------------------------------------------------------------------- #
def test_bibtex_resolves_a_result_citation_key():
    from statspai.agent.workflow_tools import _tool_bibtex

    out = _tool_bibtex({"keys": ["callaway_santanna", "rambachan2023more"]})
    assert out["resolved_from_function"] == {
        "callaway_santanna": ["callaway2021difference"]
    }
    assert "10.1016/j.jeconom.2020.12.001" in out["bibtex"]["callaway2021difference"]
    assert out["bibtex"]["rambachan2023more"].startswith("@article{rambachan2023more")
    assert out["unknown_keys"] == [] and out["suggestions"] == {}


def test_bibtex_says_when_an_estimator_has_no_verified_reference():
    from statspai.agent.workflow_tools import _tool_bibtex

    out = _tool_bibtex({"keys": ["did_2x2", "not_a_key_or_function_zzz"]})
    assert out["no_verified_citation"] == ["did_2x2"]
    assert out["unknown_keys"] == ["not_a_key_or_function_zzz"]
    assert "did_2x2" not in out["bibtex"]


def test_robust_synth_cites_what_its_card_cites():
    from statspai.agent._enrichment import build_citations

    keys = build_citations("robust_synth")
    reference = sp.describe_function("robust_synth")["reference"]
    assert keys and all(k in reference for k in keys)
    assert "benmichael2021augmented" not in keys


# --------------------------------------------------------------------------- #
#  Cards against the functions they describe
# --------------------------------------------------------------------------- #
def test_wild_cluster_bootstrap_accepts_every_weight_type_its_schema_lists(obs):
    data = obs.assign(g=np.arange(len(obs)) % 30)
    enum = sp.function_schema("wild_cluster_bootstrap")["parameters"]["properties"][
        "weight_type"
    ]["enum"]
    assert set(enum) == {"rademacher", "mammen", "webb"}
    for value in enum:
        out = sp.wild_cluster_bootstrap(
            data,
            y="y",
            x=["d", "x1"],
            cluster="g",
            n_boot=49,
            seed=1,
            weight_type=value,
        )
        assert out["weight_type"] == value
    with pytest.raises(sp.exceptions.MethodIncompatibility) as err:
        sp.wild_cluster_bootstrap(
            data, y="y", x=["d"], cluster="g", n_boot=49, weight_type="normal"
        )
    assert "rademacher" in err.value.recovery_hint


def test_cic_schema_default_seed_is_the_signature_default():
    spec = sp.function_schema("cic")["parameters"]["properties"]["seed"]
    assert spec["default"] == inspect.signature(sp.cic).parameters["seed"].default


def _card_fields(name):
    return set(sp.describe_function(name)["result_schema"]["fields"])


def test_result_schema_fields_exist_on_the_result():
    power = sp.power("rct", n=400, effect_size=0.3)
    fields = _card_fields("power")
    assert "power" in fields and "power_val" not in fields
    assert all(hasattr(power, f) for f in fields)

    rng = np.random.default_rng(0)
    t, c = rng.exponential(1.0, 200), rng.exponential(2.0, 200)
    surv = pd.DataFrame({"time": np.minimum(t, c), "event": (t <= c).astype(int)})
    km = sp.kaplan_meier(surv, duration="time", event="event")
    fields = _card_fields("kaplan_meier")
    assert "survival_table" in fields and "tables" not in fields
    assert all(hasattr(km, f) for f in fields)


def test_decomposition_result_does_not_borrow_the_iv_test_card(obs):
    from statspai._result_contract import resolve_function

    data = obs.assign(
        ybin=(obs["y"] > obs["y"].median()).astype(int),
        age=np.where(obs["x2"] > 0, "old", "young"),
    )
    res = sp.kitagawa_decompose(data, rate="ybin", group="d", by="age")
    assert resolve_function(res) != "kitagawa_test"
    steps = res.next_steps(print_result=False)
    text = " ".join(str(s.get("action", "")) + str(s.get("reason", "")) for s in steps)
    assert "instrument" not in text.lower()
    # the decomposition itself is an identity
    d = res.to_dict()
    assert d["rate_effect"] + d["composition_effect"] + d["interaction"] == (
        pytest.approx(d["gap"], abs=1e-12)
    )


# --------------------------------------------------------------------------- #
#  Search
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query, expected_first",
    [
        ("Kaplan-Meier survival curve", {"kaplan_meier"}),
        ("Rosenbaum bounds matched pairs sensitivity", {"rosenbaum_bounds"}),
        ("zero-inflated count model", {"zip_model", "zinb"}),
    ],
)
def test_search_finds_the_function_the_query_names(query, expected_first):
    names = [m["name"] for m in sp.search_functions(query)]
    assert names[0] in expected_first
    assert expected_first <= set(names[: len(expected_first)])


# --------------------------------------------------------------------------- #
#  sp.recommend
# --------------------------------------------------------------------------- #
def test_recommend_two_period_group_design_is_the_2x2(did2x2):
    rec = sp.recommend(did2x2, y="y", treatment="treated", time="post", id="unit")
    assert rec.design == "did"
    top = rec.recommendations[0]
    assert top["function"] == "did"
    params = {k: v for k, v in top["params"].items() if k != "data"}
    fit = sp.did(did2x2, **params)
    cell = did2x2.groupby(["treated", "post"])["y"].mean()
    by_hand = (cell[1, 1] - cell[1, 0]) - (cell[0, 1] - cell[0, 0])
    assert fit.estimate == pytest.approx(by_hand, rel=1e-10)


def test_recommend_continuous_regressor_panel_is_a_panel_regression(firm_panel):
    rec = sp.recommend(
        firm_panel,
        y="profit",
        treatment="invest",
        time="year",
        id="firm",
        covariates=["size"],
    )
    assert rec.design == "panel"
    top = rec.recommendations[0]
    assert (top["function"], top["params"]["method"]) == ("panel", "fe")
    fit = sp.panel(**top["params"])
    # the firm effect is correlated with invest; within estimation recovers 0.7
    assert float(fit.params["invest"]) == pytest.approx(0.7, abs=0.1)


def test_recommend_does_not_offer_fixed_effects_for_a_unit_constant_treatment():
    rng = np.random.default_rng(2)
    n = 150
    tr = (rng.random(n) < 0.5).astype(int)
    frame = pd.concat(
        [
            pd.DataFrame(
                {"id": np.arange(n), "year": yr, "d": tr, "y": tr + rng.normal(size=n)}
            )
            for yr in range(2010, 2014)
        ],
        ignore_index=True,
    )
    rec = sp.recommend(frame, y="y", treatment="d", time="year", id="id")
    methods = [
        r["params"].get("method")
        for r in rec.recommendations
        if r["function"] == "panel"
    ]
    assert "fe" not in methods
    assert any("absorbed" in w for w in rec.warnings)
    # the claim in the warning is true: the within fit has no slope for d
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.panel(frame, "y ~ d", entity="id", time="year", method="fe")


def test_recommend_iv_cards_run_and_keep_the_controls(iv_frame):
    weak = iv_frame.assign(d=0.02 * iv_frame["z"] + iv_frame["d"] - 0.8 * iv_frame["z"])
    rec = sp.recommend(weak, y="y", treatment="d", instrument="z", covariates=["x"])
    by_fn = {r["function"]: r for r in rec.recommendations}
    assert by_fn["liml"]["params"]["x_exog"] == ["x"]
    assert by_fn["anderson_rubin_ci"]["params"]["exog"] == ["x"]
    for name in ("liml", "anderson_rubin_ci", "ivreg"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            getattr(sp, name)(**by_fn[name]["params"])  # the card is a valid call


def test_recommendation_arguments_survive_as_plain_json(synth_panel):
    from statspai.agent.tools._specs._orchestrate import (
        _recommendation_entry,
        _recommendation_next_call,
    )

    rec = sp.recommend(
        synth_panel, y="gdp", treatment="treated", time="year", id="region"
    )
    top = rec.recommendations[0]
    assert top["function"] == "synth"
    assert "np.int64" not in top["code"]
    entry = _recommendation_entry(top, list(synth_panel.columns))
    assert entry["params"]["treatment_time"] == 2000
    assert type(entry["params"]["treatment_time"]) is int
    call = _recommendation_next_call(entry)
    assert call["arguments"]["treatment_time"] == 2000
    json.dumps(call)


# --------------------------------------------------------------------------- #
#  sp.preflight
# --------------------------------------------------------------------------- #
def _status(report):
    return {c["name"]: c["status"] for c in report["checks"]}


def test_preflight_fails_matching_without_common_support(obs, separated):
    good = sp.preflight(obs, "match", y="y", treat="d", covariates=["x1", "x2", "x3"])
    assert good["verdict"] == "PASS"
    assert _status(good)["propensity_overlap"] == "passed"
    bad = sp.preflight(separated, "match", y="y", treat="d", covariates=["x1", "x2"])
    assert bad["verdict"] == "FAIL"
    assert _status(bad)["propensity_overlap"] == "failed"
    # the same data are blocked by the design-level check
    ident = sp.check_identification(
        separated, y="y", treatment="d", covariates=["x1", "x2"]
    )
    assert ident.verdict == "BLOCKERS"


def test_preflight_unknown_method_is_not_a_pass(obs):
    out = sp.preflight(obs, "some_estimator_without_checks", y="y")
    assert out["known_method"] is False
    assert out["verdict"] == "WARN"
    assert _status(out)["method_specific_checks"] == "warning"


def test_preflight_knows_the_curated_estimators(obs, synth_panel, firm_panel):
    assert sp.preflight(obs, "dml", y="y", treat="d", covariates=["x1"])["known_method"]
    ok = sp.preflight(
        synth_panel,
        "synth",
        outcome="gdp",
        unit="region",
        time="year",
        treated_unit="R01",
        treatment_time=2000,
    )
    assert ok["verdict"] == "PASS"
    assert ok["checks"][-1]["evidence"]["n_pre_periods"] == 20
    absent = sp.preflight(
        synth_panel,
        "sdid",
        y="gdp",
        unit="region",
        time="year",
        treat_unit="R99",
        treat_time=2000,
    )
    assert absent["verdict"] == "FAIL"
    no_pre = sp.preflight(
        synth_panel,
        "synth",
        outcome="gdp",
        unit="region",
        time="year",
        treated_unit="R01",
        treatment_time=1980,
    )
    assert no_pre["verdict"] == "FAIL"
    fe = sp.preflight(firm_panel, "feols", fml="profit ~ invest | firm + year")
    assert fe["verdict"] == "PASS"
    typo = sp.preflight(firm_panel, "feols", fml="profit ~ invst | firm + year")
    assert typo["verdict"] == "FAIL"
    assert typo["checks"][2]["evidence"]["missing"] == ["invst"]


# --------------------------------------------------------------------------- #
#  Translation
# --------------------------------------------------------------------------- #
def test_matchit_default_distance_translates_to_a_call_that_runs(obs):
    out = sp.from_r(
        'matchit(d ~ x1 + x2 + x3, data = df, method = "nearest", distance = "glm")'
    )
    assert out["ok"] is True
    assert out["arguments"]["distance"] == "propensity"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.match(obs, y="y", **out["arguments"])
        same = sp.match(
            obs,
            y="y",
            treat="d",
            covariates=["x1", "x2", "x3"],
            distance="propensity",
            replace=False,
            n_matches=1,
            estimand="ATT",
            m_order="largest",
        )
    assert fit.estimate == same.estimate


def test_matchit_distance_without_a_counterpart_is_reported_not_passed():
    out = sp.from_r('matchit(d ~ x1, data = df, distance = "randomforest")')
    assert "distance" not in out["arguments"]
    assert "distance" in json.dumps(out.get("untranslated_arguments", out["notes"]))
    probit = sp.from_r('matchit(d ~ x1, data = df, distance = "glm", link = "probit")')
    assert probit["arguments"]["ps_model"] == "probit"


def test_stata_data_loading_line_says_to_pass_the_data(firm_panel):
    with pytest.raises(sp.exceptions.MethodIncompatibility) as err:
        sp.stata("use firms, clear\nregress profit invest", data=firm_panel)
    assert "Remove this line" in err.value.recovery_hint
    fit = sp.stata("regress profit invest", data=firm_panel)
    direct = sp.regress("profit ~ invest", data=firm_panel)
    assert float(fit.params["invest"]) == pytest.approx(float(direct.params["invest"]))


# --------------------------------------------------------------------------- #
#  Errors
# --------------------------------------------------------------------------- #
def test_transform_data_names_the_missing_column(obs):
    from statspai.agent.workflow_tools import _tool_transform_data

    out = _tool_transform_data(
        {"operations": [{"op": "query", "expr": "x11 > 0"}]}, obs.copy()
    )
    assert out["error_kind"] == "missing_column"
    assert out["missing_column"] == "x11"
    assert "x1" in out["did_you_mean"]
    assert out["hint"] == out["recovery_hint"]


def test_transform_data_function_step_lists_the_accepted_arguments(obs):
    from statspai.agent.workflow_tools import _tool_transform_data

    out = _tool_transform_data(
        {
            "operations": [
                {"op": "function", "name": "winsor", "arguments": {"columns": ["y"]}}
            ]
        },
        obs.copy(),
    )
    accepted = [p for p in inspect.signature(sp.winsor).parameters if p != "data"]
    assert out["error_kind"] == "method_incompatibility"
    for name in accepted:
        assert name in out["hint"]


def test_error_strings_are_bounded():
    from statspai.agent.mcp_server import _ERROR_STRING_MAX, _cap_error_strings

    long = "x" * 50_000
    payload = {
        "error": long,
        "remediation": {"exception_message": long, "fix": "short"},
        "error_payload": {"message": long, "diagnostics": {"values": [long]}},
    }
    _cap_error_strings(payload)
    assert len(json.dumps(payload)) < 5 * (_ERROR_STRING_MAX + 200)
    assert payload["remediation"]["fix"] == "short"
    assert payload["error"].startswith("x" * 100) and payload["error"].endswith("x")
    assert "characters omitted" in payload["error"]


def test_match_on_a_continuous_treatment_names_a_few_values(iv_frame):
    with pytest.raises(sp.exceptions.MethodIncompatibility) as err:
        sp.match(iv_frame, y="y", treat="d", covariates=["x"])
    message = str(err.value)
    assert f"{iv_frame['d'].nunique()} distinct values" in message
    assert len(message) < 600


def test_remediation_fix_is_the_exceptions_own_hint(iv_frame):
    from statspai.agent.remediation import remediate

    try:
        sp.match(iv_frame, y="y", treat="d", covariates=["x"])
    except sp.exceptions.MethodIncompatibility as exc:
        rem = remediate(exc, context={"tool": "match"})
        assert rem["fix"] == exc.recovery_hint
        assert rem.get("category_fix") != rem["fix"]


def test_followup_to_an_optional_data_tool_carries_the_dataset(iv_frame, tmp_path):
    """``sp.liml(data=None, ...)``: ``data`` has a default, the call needs it."""
    from statspai.agent.mcp_server import handle_request

    weak = iv_frame.assign(d=0.02 * iv_frame["z"] + iv_frame["d"] - 0.8 * iv_frame["z"])
    path = tmp_path / "weak.csv"
    weak.to_csv(path, index=False)

    def call(name, **arguments):
        reply = handle_request(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                }
            )
        )
        return json.loads(reply)["result"]

    rec = call(
        "recommend",
        data_path=str(path),
        y="y",
        treatment="d",
        instrument="z",
        covariates=["x"],
    )["structuredContent"]
    first, *_rest = rec["next_calls"]
    assert first["tool"] == "liml" and first["ready"] is True
    assert first["arguments"]["data_path"] == str(path)
    out = call(first["tool"], **first["arguments"])
    assert not out.get("isError"), out["structuredContent"].get("error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = sp.liml(data=weak, y="y", x_endog=["d"], x_exog=["x"], z=["z"])
    assert out["structuredContent"]["coefficients"]["d"]["estimate"] == pytest.approx(
        float(direct.params["d"]), rel=1e-9
    )


def test_audit_keeps_its_treat_alias(obs):
    """``sp.audit(result, treat=...)`` is the house spelling of ``treatment=``.

    The alias decorator was displaced when ``evidence=`` was added (the CI
    house-style ratchet caught it); both spellings must give one report.
    """
    fit = sp.regress("y ~ d + x1 + x2", data=obs)
    by_alias = sp.audit(fit, treat="d")
    by_name = sp.audit(fit, treatment="d")
    assert by_alias == by_name
    names = {c["name"] for c in by_alias["checks"]}
    # declaring a treatment adds the observational-design checks
    assert {"overlap", "balance_after"} <= names
    assert not {"overlap", "balance_after"} & {
        c["name"] for c in sp.audit(fit)["checks"]
    }
