"""``next_steps()`` offers a result only what its own family can run.

The 2026-11-13 agent probe (``docs/dev/2026-11-13-agent-probe.md``, #35
and #36) found steps inherited from a neighbouring family:

* ``sp.regress('time ~ x + z', robust='hc1')`` after ``sp.cox`` (every
  ``EconometricResults`` subclass got the linear-model toolkit; the label
  "Cox Proportional Hazards" was also read as an RD design, because
  "hazards" contains "rd");
* the RD checklist after ``sp.bunching`` ("Kink Design");
* the sharp reduced form after a fuzzy RD, the level-jump tools after a
  kink design, and staggered estimators filled with the before / after
  indicator of a 2x2 fit;
* every failure mode registered on the producing function's card, whether
  or not it occurred.

Each test states one of: the step runs verbatim on the fit's own data, a
refit reproduces the fit's coefficients, or a step that belongs to another
family is absent.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.next_steps import _FORMULA_ROBUST_REFIT, _detect_family

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

COVS = ["age", "education", "re74", "re75"]


def _survival_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["time"] = rng.exponential(np.exp(-0.5 * df["x"]))
    df["event"] = (rng.uniform(size=n) < 0.8).astype(int)
    df["cnt"] = rng.poisson(np.exp(0.3 * df["x"]))
    df["b"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-df["x"]))).astype(int)
    return df


def _fits():
    rng = np.random.default_rng(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = sp.datasets.mpdta()
        nsw = sp.datasets.nsw_dw()
        lee = sp.datasets.lee_2008_senate()
        card = sp.datasets.card_1995()
        ca = sp.datasets.california_prop99()
    sv = _survival_frame()
    bun = pd.DataFrame(
        {"inc": np.concatenate([rng.uniform(0, 100, 5000), 50 + rng.normal(0, 1, 400)])}
    )
    two = sp.dgp_did(n_units=100, n_periods=2, seed=0)
    two["treat"] = two["first_treat"].notna().astype(int)
    two["post"] = (two["time"] >= 1).astype(int)
    mp2 = mp.assign(
        treated=((mp["first_treat"] > 0) & (mp["year"] >= mp["first_treat"])).astype(
            int
        )
    )
    u = rng.uniform(size=len(lee))
    fz = lee.assign(d=np.where(lee["x"] >= 0, u < 0.8, u < 0.1).astype(int))
    return {
        "cox": (sv, lambda: sp.cox("time ~ x + z", data=sv, event="event")),
        "logit": (sv, lambda: sp.logit("b ~ x + z", data=sv)),
        "probit": (sv, lambda: sp.probit("b ~ x + z", data=sv)),
        "poisson": (sv, lambda: sp.poisson("cnt ~ x + z", data=sv)),
        "nbreg": (sv, lambda: sp.nbreg("cnt ~ x + z", data=sv)),
        "glm": (sv, lambda: sp.glm("cnt ~ x + z", data=sv, family="poisson")),
        "feols_cluster": (
            mp2,
            lambda: sp.feols(
                "lemp ~ treated | countyreal + year",
                data=mp2,
                vcov={"CRV1": "countyreal"},
            ),
        ),
        "panel": (
            mp2,
            lambda: sp.panel(mp2, "lemp ~ treated", entity="countyreal", time="year"),
        ),
        "liml": (
            card,
            lambda: sp.liml(
                data=card, y="lwage", x_endog=["educ"], z=["nearc4"], x_exog=["exper"]
            ),
        ),
        "did_2x2": (two, lambda: sp.did(two, y="y", treat="treat", time="post")),
        "sun_abraham": (
            mp,
            lambda: sp.sun_abraham(
                mp, y="lemp", g="first_treat", t="year", i="countyreal"
            ),
        ),
        "did_imputation": (
            mp,
            lambda: sp.did_imputation(
                mp,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
            ),
        ),
        "sdid": (
            ca,
            lambda: sp.sdid(
                ca,
                outcome="cigsale",
                unit="state",
                time="year",
                treated_unit="California",
                treatment_time=1989,
            ),
        ),
        "rd_fuzzy": (fz, lambda: sp.rdrobust(fz, y="y", x="x", c=0, fuzzy="d")),
        "rkd": (lee, lambda: sp.rkd(lee, y="y", x="x", c=0)),
        "bunching": (
            bun,
            lambda: sp.bunching(bun, running_var="inc", threshold=50, n_bootstrap=20),
        ),
        "ipw": (nsw, lambda: sp.ipw(nsw, y="re78", treat="treat", covariates=COVS)),
        "ebalance": (
            nsw,
            lambda: sp.ebalance(nsw, y="re78", treat="treat", covariates=COVS),
        ),
        "mediate": (
            nsw,
            lambda: sp.mediate(
                nsw,
                y="re78",
                treat="treat",
                mediator="re75",
                covariates=["age", "education"],
                n_boot=50,
            ),
        ),
        "qreg": (nsw, lambda: sp.qreg("re78 ~ treat + age", data=nsw)),
        "tobit": (nsw, lambda: sp.tobit("re78 ~ treat + age", data=nsw)),
    }


_FITS = _fits()
_CACHE: dict = {}


def _fitted(name):
    if name not in _CACHE:
        df, fit = _FITS[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit()
            _CACHE[name] = (df, result, result.next_steps())
    return _CACHE[name]


def _actions(name):
    return [s["action"] for s in _fitted(name)[2]]


@pytest.mark.parametrize("name", sorted(_FITS))
def test_every_executable_step_runs_on_the_fit_own_data(
    name, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)  # export steps write files
    df, result, steps = _fitted(name)
    executable = [s for s in steps if s["executable"]]
    assert executable, f"{name}: no step can be run as written"
    failures = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in executable:
            try:
                eval(step["action"], {"sp": sp, "result": result, "df": df})
            except Exception as exc:  # pragma: no cover - reported below
                failures.append(f"{step['action']} -> {type(exc).__name__}: {exc}")
    capsys.readouterr()
    assert not failures, f"{name}: " + "\n".join(failures)
    for step in steps:
        assert step["executable"] == ("..." not in step["action"]), step


# ---------------------------------------------------------------------- #
#  Regression results: the linear toolkit stays with linear models
# ---------------------------------------------------------------------- #

_LINEAR_ONLY = ("sp.regress(", "sp.oster_delta(", "sp.stepwise(")


def test_cox_gets_the_hazard_checks_not_the_linear_toolkit():
    actions = _actions("cox")
    assert "result.ph_test()" in actions
    for call in _LINEAR_ONLY + ("sp.margins(",):
        assert not any(call in a for a in actions), call


@pytest.mark.parametrize("name", ["logit", "probit", "poisson", "nbreg", "glm"])
def test_likelihood_models_are_not_refitted_by_ols(name):
    actions = _actions(name)
    for call in _LINEAR_ONLY:
        assert not any(call in a for a in actions), call
    assert "sp.margins(result, data=df)" in actions


@pytest.mark.parametrize("name", sorted(_FORMULA_ROBUST_REFIT))
def test_robust_refit_is_the_same_estimator_with_the_same_coefficients(name):
    """A robust variance changes the standard errors, never the point
    estimates: the suggested refit must reproduce the fit's coefficients."""
    df, result, steps = _fitted(name)
    refits = [s["action"] for s in steps if "robust='hc1'" in s["action"]]
    lhs = "b" if name in ("logit", "probit") else "cnt"
    assert refits == [f"sp.{name}('{lhs} ~ x + z', data=df, robust='hc1')"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = eval(refits[0], {"sp": sp, "df": df})
    np.testing.assert_allclose(
        again.params.to_numpy(), result.params.to_numpy(), rtol=1e-10, atol=1e-12
    )
    assert "robust" in inspect.signature(getattr(sp, name)).parameters


def test_panel_refit_keeps_the_unit_effects():
    """``sp.regress`` on the panel formula is pooled OLS: another estimate.
    The refit keeps the within estimator and clusters on the unit."""
    df, result, steps = _fitted("panel")
    refits = [s["action"] for s in steps if s["category"] == "robustness"]
    assert refits[0] == (
        "sp.panel(df, 'lemp ~ treated', entity='countyreal', time='year', "
        "method='fe', cluster='countyreal')"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = eval(refits[0], {"sp": sp, "df": df})
    assert float(again.params["treated"]) == pytest.approx(
        float(result.params["treated"]), rel=1e-12
    )
    hausman = [a for a in _actions("panel") if "hausman_test" in a]
    assert hausman and "id='countyreal'" in hausman[0] and "time='year'" in hausman[0]


def test_feols_reads_the_outcome_from_its_formula():
    oster = [a for a in _actions("feols_cluster") if "oster_delta" in a]
    assert oster and "y='lemp'" in oster[0]


def test_column_argument_fit_has_no_formula_refit():
    # sp.liml is called with columns; there is no formula to refit.
    assert not any("~ ..." in a for a in _actions("liml"))


# ---------------------------------------------------------------------- #
#  Causal results
# ---------------------------------------------------------------------- #

_RD_ONLY = ("mccrary_test", "rd_honest", "rdbalance", "rdbwsensitivity", "rdplacebo")


def test_bunching_is_not_handed_the_rd_checklist():
    actions = _actions("bunching")
    for call in _RD_ONLY:
        assert not any(call in a for a in actions), call


def test_fuzzy_rd_followups_carry_the_treatment_column():
    actions = _actions("rd_fuzzy")
    with_outcome = [a for a in actions if "y='y'" in a]
    assert len(with_outcome) == 2
    assert all("fuzzy='d'" in a for a in with_outcome)
    # the honest interval is for the sharp jump in the outcome
    assert not any("rd_honest" in a for a in actions)


def test_kink_design_is_not_handed_the_level_jump_tools():
    df, result, steps = _fitted("rkd")
    actions = [s["action"] for s in steps]
    for call in ("rd_honest", "rdbwsensitivity", "rdplacebo"):
        assert not any(call in a for a in actions), call
    h = result.model_info["bandwidth"]
    assert f"sp.rkd(data=df, y='y', x='x', c=0, h={h / 2!r})" in actions


def test_two_period_fit_does_not_pass_its_post_indicator_as_calendar_time():
    actions = _actions("did_2x2")
    staggered = [
        a
        for a in actions
        if a.startswith(
            (
                "sp.event_study",
                "sp.bacon_decomposition",
                "sp.did_imputation",
                "sp.callaway_santanna",
            )
        )
    ]
    assert len(staggered) == 4
    assert not any("'post'" in a for a in staggered)


def test_did_imputation_unit_column_is_read_from_group():
    assert (
        "sp.event_study(data=df, y='lemp', treat_time='first_treat', "
        "time='year', unit='countyreal')"
    ) in _actions("did_imputation")


@pytest.mark.parametrize("name", ["qreg", "tobit", "bunching"])
def test_no_evalue_without_a_designated_treatment(name):
    assert not any("evalue" in a for a in _actions(name))


@pytest.mark.parametrize(
    "label, family",
    [
        ("Cox Proportional Hazards", "generic"),
        ("Generalized ordered logit", "generic"),
        ("Hurdle (Poisson, MLE)", "generic"),
        ("forward", "generic"),
        ("Effect size of a difference in means", "generic"),
        ("Difference in means (paired design)", "generic"),
        ("Bunching Estimator (Kink Design)", "generic"),
        ("CJM (2020) Density Test [rddensity::rddensity]", "rd"),
        ("Regression Kink Design (Card et al., 2015)", "rd"),
        ("Fuzzy RD Estimation", "rd"),
        ("Difference-in-Differences (2x2)", "did"),
        ("Difference in differences (TWFE, clustered)", "did"),
        ("Triple Differences (DDD)", "did"),
        ("Difference-in-discontinuities (Grembi et al. 2016)", "did"),
        ("Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)", "synth"),
    ],
)
def test_family_is_read_from_whole_words(label, family):
    assert _detect_family(label.lower()) == family


# ---------------------------------------------------------------------- #
#  Card-derived steps: failure modes that occurred, not the catalogue
# ---------------------------------------------------------------------- #


def test_failure_modes_that_did_not_occur_stay_in_the_card():
    sv = _survival_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = sp.kaplan_meier(sv, duration="time", event="event")
    card = sp.describe_function("kaplan_meier")
    remedies = {fm["remedy"] for fm in card["failure_modes"]}
    assert len(remedies) >= 2
    assert km.violations() == []
    steps = km.next_steps()
    assert not remedies & {s["action"] for s in steps}
    # the registered alternatives are still offered
    assert {s["suggest_function"] for s in steps} == set(card["alternatives"])
    assert all(a.startswith("sp.") for a in card["alternatives"])


def test_failure_mode_is_listed_when_the_result_reports_it():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        card_df = sp.datasets.card_1995()
        card_df["g"] = (card_df["exper"] // 3).astype(int)  # 8 clusters
        few = sp.hdfe_ols("lwage ~ educ | black", data=card_df, cluster="g")
        none = sp.hdfe_ols("lwage ~ educ | black", data=card_df)
    assert card_df["g"].nunique() == 8
    remedy = next(
        fm["remedy"]
        for fm in sp.describe_function("hdfe_ols")["failure_modes"]
        if "few clusters" in fm["symptom"].lower()
    )
    assert [(v["test"], v["value"]) for v in few.violations()] == [("few_clusters", 8)]
    listed = [s for s in few.next_steps() if s["action"] == remedy]
    assert len(listed) == 1 and "few_clusters" in listed[0]["reason"]
    # other failure modes of the same card did not occur: not listed
    diagnostics = [s for s in few.next_steps() if s["category"] == "diagnostics"]
    assert diagnostics == listed
    assert none.violations() == []
    assert not [s for s in none.next_steps() if s["category"] == "diagnostics"]
