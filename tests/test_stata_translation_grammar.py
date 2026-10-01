"""Stata grammar shared by every command: abbreviations, prefixes, macros.

Up to 1.34.2 ``sp.from_stata`` read options by full name only, so the
abbreviations Stata's own syntax diagrams allow were dropped with
``ok=True`` and empty notes: ``reg y x, r`` lost its robust SEs,
``cl(id)`` its clustering, and ``reghdfe y x, a(id year)`` every fixed
effect. Unknown options vanished the same way, macros were pasted into
the formula, and ``sp.stata`` ran ``... if cond`` on the full sample.

The rules tested here come from Stata's documented syntax, not from any
one do-file:

* an abbreviated command translates exactly like the spelled-out one;
* an option is honoured, reported in ``untranslated_options``, or listed
  as display-only -- never dropped without a trace;
* ``sp.stata`` refuses to run a translation that lost something.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _same_call(a, b):
    return (a["ok"], a.get("tool"), a.get("arguments")) == (
        b["ok"],
        b.get("tool"),
        b.get("arguments"),
    )


@pytest.mark.parametrize(
    "abbreviated, full",
    [
        ("reg y x, r", "regress y x, robust"),
        ("reg y x, ro", "regress y x, robust"),
        ("reg y x, vce(r)", "regress y x, vce(robust)"),
        ("reg y x, cl(id)", "regress y x, cluster(id)"),
        ("reg y x, vce(cl id)", "regress y x, vce(cluster id)"),
        ("reghdfe y x, a(id year)", "reghdfe y x, absorb(id year)"),
        ("reghdfe y x, ab(id year) cl(id)", "reghdfe y x, absorb(id year) cluster(id)"),
        (
            "reghdfe y x, a(id#year) vce(cl id)",
            "reghdfe y x, absorb(id#year) vce(cluster id)",
        ),
        (
            "ivreghdfe y (d = z), a(id) cl(id)",
            "ivreghdfe y (d = z), absorb(id) cluster(id)",
        ),
        ("ppmlhdfe y x, a(id) cl(id)", "ppmlhdfe y x, absorb(id) cluster(id)"),
        ("logit y x, vce(cl id)", "logit y x, vce(cluster id)"),
        ("ivregress 2sls y (d = z), r", "ivregress 2sls y (d = z), robust"),
    ],
)
def test_abbreviated_options_translate_like_the_full_names(abbreviated, full):
    a, b = sp.from_stata(abbreviated), sp.from_stata(full)
    assert a["ok"] and _same_call(a, b)
    assert a["untranslated_options"] == []
    assert any("abbreviations expanded" in s for s in a["semantics"])


def test_the_abbreviated_absorb_keeps_the_fixed_effects():
    out = sp.from_stata("reghdfe y x, a(id year) cl(id)")
    assert out["arguments"] == {"formula": "y ~ x | id + year", "cluster": "id"}


def test_a_full_name_wins_over_an_abbreviation_of_it():
    # ``r`` cannot be expanded to ``robust`` when ``robust`` is also written
    out = sp.from_stata("reg y x, robust r")
    assert out["arguments"]["robust"] == "hc1"
    assert out["untranslated_options"] == ["r"]


@pytest.mark.parametrize(
    "line, lost",
    [
        ("reg y x, foobar(3)", ["foobar"]),
        ("reg y x, nocons", ["noconstant"]),
        ("reghdfe y x, absorb(id) dofadjustments(none)", ["dofadjustments"]),
        ("reg y x, small", ["small"]),
        ("reg y x, vce(bootstrap)", ["vce"]),
        ("rdrobust y x, scaleregul(0)", ["scaleregul"]),
        ("rdrobust y x, vce(nncluster id)", ["vce"]),
        ("rdrobust y x, kernel(gauss)", ["kernel"]),
    ],
)
def test_an_option_that_is_not_carried_over_is_reported(line, lost):
    out = sp.from_stata(line)
    assert out["ok"]
    assert out["untranslated_options"] == lost
    assert out["notes"], "an untranslated option must come with a note"


@pytest.mark.parametrize(
    "line, shown",
    [
        ("reghdfe y x, absorb(id) noheader", ["noheader"]),
        ("logit y x, or", ["or"]),
        ("poisson y x, irr", ["irr"]),
        ("reg y x, beta", ["beta"]),
        ("rdrobust y x, all", ["all"]),
        ("rdplot y x, graph_options(title(RD))", ["graph_options"]),
        ("rddensity x, plot", ["plot"]),
        ("reghdfe y x, noabsorb", ["noabsorb"]),
    ],
)
def test_display_options_are_listed_but_do_not_count_as_lost(line, shown):
    out = sp.from_stata(line)
    assert out["untranslated_options"] == []
    assert out["ignored_display_options"] == shown


def test_level_becomes_alpha_where_the_function_takes_it():
    out = sp.from_stata("logit y x, level(90)")
    assert out["arguments"]["alpha"] == pytest.approx(0.1)
    assert out["python_code"].endswith("alpha=0.1)")
    # sp.regress has no alpha: the level is a reporting choice, not a loss
    out = sp.from_stata("reg y x, level(90)")
    assert "alpha" not in out["arguments"] and out["untranslated_options"] == []


@pytest.mark.parametrize(
    "line", ["reg y x $controls, r", "reg y x `controls'", "reghdfe y x, absorb(${fe})"]
)
def test_macros_are_refused_not_pasted_into_the_call(line):
    out = sp.from_stata(line)
    assert not out["ok"] and "macro" in out["error"]


def test_a_macro_in_a_label_option_does_not_block_the_translation():
    out = sp.from_stata("rdplot y x, title(`t')")
    assert out["ok"]


@pytest.mark.parametrize(
    "line",
    [
        "qui reg y x, robust",
        "quietly: reg y x, robust",
        "cap noi reg y x, robust",
        "eststo m1: reg y x, robust",
        "eststo: qui reg y x, robust",
        "xi: reg y x, robust",
    ],
)
def test_prefixes_that_only_change_the_output_are_peeled(line):
    out = sp.from_stata(line)
    assert _same_call(out, sp.from_stata("reg y x, robust"))
    assert any("Prefix" in s for s in out["semantics"])


@pytest.mark.parametrize(
    "line",
    [
        "by g: reg y x",
        "bys g: reg y x",
        "bysort g (t): reg y x",
        "bootstrap, reps(200): reg y x",
        "svy: reg y x",
        "permute x _b[x]: reg y x",
    ],
)
def test_prefixes_that_change_the_estimate_are_refused(line):
    out = sp.from_stata(line)
    assert not out["ok"] and "prefix" in out["error"]


def test_xtreg_needs_fe_and_robust_means_clustering_on_the_panel():
    # xtreg's default estimator is random effects, not the within estimator
    assert not sp.from_stata("xtreg y x, i(id)")["ok"]
    assert not sp.from_stata("xtreg y x, be i(id)")["ok"]
    out = sp.from_stata("xtreg y x, fe r i(id)")
    assert out["arguments"] == {"fml": "y ~ x | id", "cluster": "id"}
    # an explicit cluster variable is kept
    out = sp.from_stata("xtreg y x, fe vce(cl firm) i(id)")
    assert out["arguments"]["cluster"] == "firm"


def test_rdrobust_options_map_onto_sp_rdrobust():
    out = sp.from_stata(
        "rdrobust y x, c(1) p(2) kernel(uni) bwselect(msetwo) covs(a b) "
        "vce(cl id) masspoints(off) level(90) all"
    )
    assert out["arguments"] == {
        "y": "y",
        "x": "x",
        "c": 1.0,
        "p": 2,
        "kernel": "uniform",
        "bwselect": "msetwo",
        "covs": ["a", "b"],
        "cluster": "id",
        "masspoints": "off",
        "alpha": 0.1,
    }
    assert out["untranslated_options"] == []
    assert sp.from_stata("rdrobust y x, h(5 7)")["arguments"]["h"] == (5.0, 7.0)
    assert sp.from_stata("rdrobust y x, vce(hc2)")["arguments"]["vce"] == "hc2"


# ---------------------------------------------------------------------------
# No option is dropped without a trace
# ---------------------------------------------------------------------------

_COMMANDS = {
    "regress y x1 x2": "",
    "reghdfe y x1 x2": "absorb(id)",
    "xtreg y x1 x2": "fe i(id)",
    "ivregress 2sls y x2 (d = z)": "",
    "ivreg2 y x2 (d = z)": "",
    "ivreghdfe y x2 (d = z)": "absorb(id)",
    "logit b x1 x2": "",
    "probit b x1 x2": "",
    "poisson c x1 x2": "",
    "nbreg c x1 x2": "",
    "tobit y x1 x2": "ll(0)",
    "ppmlhdfe c x1 x2": "absorb(id)",
    "oprobit o x1 x2": "",
    "mlogit o x1 x2": "",
    "rdrobust y x1": "",
    "rddensity x1": "",
    "rdplot y x1": "",
    "heckman y x1": "select(b = x1 x2)",
    "psmatch2 b x1": "outcome(y)",
}
_OPTIONS = [
    "r",
    "robust",
    "cl(id)",
    "cluster(id)",
    "vce(cl id)",
    "vce(r)",
    "vce(bootstrap)",
    "nocons",
    "level(90)",
    "foobar(1)",
    "noheader",
    "nolog",
    "small",
    "first",
    "or",
    "irr",
    "beta",
    "all",
    "keepsin",
    "offset(e)",
    "exposure(e)",
    "iterate(50)",
]


def _signature(out):
    return (
        out.get("ok"),
        out.get("python_code"),
        tuple(out.get("notes") or []),
        tuple(out.get("semantics") or []),
        out.get("error"),
    )


@pytest.mark.parametrize("head", sorted(_COMMANDS))
def test_no_option_is_dropped_without_a_trace(head):
    """Adding an option must change the translation or be reported."""
    base = _COMMANDS[head]
    bare = sp.from_stata(f"{head}, {base}" if base else head)
    assert bare["ok"], bare
    for opt in _OPTIONS:
        out = sp.from_stata(f"{head}, {base} {opt}")
        assert _signature(out) != _signature(
            bare
        ), f"{head!r}: option {opt!r} left no trace in the translation"


# ---------------------------------------------------------------------------
# sp.stata runs what was written, or nothing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(20261001)
    n = 600
    d = pd.DataFrame(
        {
            "id": np.repeat(np.arange(60), 10),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "w": rng.uniform(0.2, 3.0, size=n),
        }
    )
    d["u"] = np.repeat(rng.normal(size=60), 10)
    d["d"] = d.z + 0.5 * d.u + rng.normal(size=n)
    d["y"] = 1 + 0.5 * d.x + (1 + d.w) * d.d + d.u + rng.normal(size=n)
    return d


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.mark.parametrize(
    "line, direct",
    [
        ("reg y x, r", lambda d: sp.regress("y ~ x", data=d, robust="hc1")),
        ("reg y x, cl(id)", lambda d: sp.regress("y ~ x", data=d, cluster="id")),
        (
            "qui reg y x, vce(cl id)",
            lambda d: sp.regress("y ~ x", data=d, cluster="id"),
        ),
        (
            "reghdfe y x, a(id) cl(id)",
            lambda d: sp.hdfe_ols("y ~ x | id", data=d, cluster="id"),
        ),
    ],
)
def test_abbreviated_commands_run_like_the_direct_call(df, line, direct):
    got, want = _quiet(sp.stata, line, data=df), _quiet(direct, df)
    assert got.params["x"] == want.params["x"]
    assert got.std_errors["x"] == want.std_errors["x"]
    # ... and the abbreviation matters: the default SEs are different
    plain = _quiet(sp.stata, line.split(",")[0].replace("qui ", ""), data=df)
    assert got.std_errors["x"] != plain.std_errors["x"]


def test_the_abbreviated_absorb_changes_the_point_estimate(df):
    fe = _quiet(sp.stata, "reghdfe y x d, a(id)", data=df)
    pooled = _quiet(sp.regress, "y ~ x + d", data=df)
    assert abs(fe.params["d"] - pooled.params["d"]) > 1e-3


@pytest.mark.parametrize(
    "line, match",
    [
        ("reg y x if z > 0", "is not applied"),
        ("reg y x in 1/100", "is not applied"),
        ("reg y x, nocons", "noconstant"),
        ("reg y x, foobar(1)", "foobar"),
        ("reghdfe y x, absorb(id) dofadjustments(none)", "dofadjustments"),
        ("reg y x, vce(bootstrap)", "vce"),
    ],
)
def test_stata_refuses_to_run_a_translation_that_lost_something(df, line, match):
    with pytest.raises(MethodIncompatibility, match=match):
        _quiet(sp.stata, line, data=df)


def test_display_options_do_not_stop_a_run(df):
    got = _quiet(sp.stata, "reg y x, r noheader beta", data=df)
    assert got.params["x"] == _quiet(sp.regress, "y ~ x", data=df).params["x"]


def test_iv_analytic_weights_are_carried_and_applied(df):
    out = sp.from_stata("ivregress 2sls y x (d = z) [aw=w], r")
    assert out["ok"] and out["arguments"]["weights"] == "w"
    got = _quiet(sp.stata, "ivregress 2sls y x (d = z) [aw=w]", data=df)
    n = len(df)
    X = np.column_stack([np.ones(n), df.x, df.d])
    Z = np.column_stack([np.ones(n), df.x, df.z])
    w = df.w.to_numpy()
    beta = np.linalg.solve(Z.T @ (X * w[:, None]), Z.T @ (df.y.to_numpy() * w))
    # just-identified weighted IV has a closed form
    assert got.params["d"] == pytest.approx(beta[2], rel=1e-10)
    unweighted = _quiet(sp.stata, "ivregress 2sls y x (d = z)", data=df)
    assert abs(got.params["d"] - unweighted.params["d"]) > 1e-3


# ---------------------------------------------------------------------------
# Options a handler reads must reach the call
# ---------------------------------------------------------------------------


def test_csdid_follows_the_documented_mapping():
    # docs/guides/callaway_santanna.md, "Migrating from Stata csdid"
    out = sp.from_stata("csdid y x1 x2, ivar(id) time(t) gvar(g)")
    assert out["arguments"] == {
        "y": "y",
        "i": "id",
        "t": "t",
        "g": "g",
        "x": ["x1", "x2"],
        "estimator": "dr",
        "base_period": "varying",
    }
    out = sp.from_stata("csdid y, ivar(id) time(t) gvar(g) method(ipw) notyet long2")
    # csdid's method(ipw) is Abadie's IPW, not the stabilised 'ipw'
    assert out["arguments"]["estimator"] == "ipw_abadie"
    assert out["arguments"]["base_period"] == "universal"
    assert out["arguments"]["control_group"] == "notyettreated"
    assert out["arguments"]["notyet_cutoff"] == "cohort"
    asinr = sp.from_stata("csdid y, ivar(id) time(t) gvar(g) notyet asinr")
    assert asinr["arguments"]["notyet_cutoff"] == "asinr"
    assert not sp.from_stata("csdid y, ivar(id) time(t) gvar(g) method(drimp)")["ok"]
    wboot = sp.from_stata("csdid y, ivar(id) time(t) gvar(g) wboot")
    assert wboot["untranslated_options"] == ["wboot"]


def test_the_csdid_line_of_parity_module_04_reproduces_stata():
    """``sp.stata`` on the line in tests/stata_parity/04_csdid.do.

    Stata 18 csdid results are the committed golden file; 1e-9 is the
    repository's reproducibility tolerance (the estimator is closed-form).
    """
    import json
    from pathlib import Path

    golden = Path(__file__).parent / "stata_parity" / "results" / "04_csdid_Stata.json"
    rows = {
        r["statistic"]: r
        for r in json.loads(golden.read_text(encoding="utf-8"))["rows"]
    }
    line = "csdid lemp, ivar(countyreal) time(year) gvar(first_treat) method(reg) long2"
    fit = _quiet(sp.stata, line, data=sp.datasets.mpdta())
    assert fit.estimate == pytest.approx(rows["simple_ATT"]["estimate"], rel=1e-9)
    assert fit.se == pytest.approx(rows["simple_ATT"]["se"], rel=1e-9)
    event = _quiet(sp.aggte, fit, type="dynamic", bstrap=False, cband=False).tidy()
    event = event.set_index("term")
    for term in ("event_-3", "event_-2", "event_+0", "event_+2"):
        assert event.loc[term, "estimate"] == pytest.approx(
            rows[term]["estimate"], rel=1e-9
        )
    # without long2 csdid uses short gaps: the pre-treatment cells differ
    short = _quiet(sp.stata, line.replace(" long2", ""), data=sp.datasets.mpdta())
    short_event = _quiet(
        sp.aggte, short, type="dynamic", bstrap=False, cband=False
    ).tidy()
    assert "event_-4" not in set(short_event["term"])


def test_did_imputation_options_reach_the_call():
    out = sp.from_stata(
        "did_imputation y id t g, horizons(0/3) pretrends(5) autosample "
        "controls(a b) cluster(id)"
    )
    assert out["arguments"] == {
        "y": "y",
        "group": "id",
        "time": "t",
        "first_treat": "g",
        "horizon": [0, 1, 2, 3],
        "pretrends": 5,
        "autosample": True,
        "controls": ["a", "b"],
        "cluster": "id",
    }
    assert sp.from_stata("did_imputation y id t g, horizons(-4(2)4)")["arguments"][
        "horizon"
    ] == [-4, -2, 0, 2, 4]
    bad = sp.from_stata("did_imputation y id t g, horizons(abc)")
    assert bad["untranslated_options"] == ["horizons"]


def _call_keywords(code):
    import ast

    call = ast.parse(code, mode="eval").body
    return {
        kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg != "data"
    }


@pytest.mark.parametrize(
    "line",
    [
        "regress y x1 x2, vce(cluster id)",
        "reghdfe y x1, absorb(id year) cluster(id)",
        "reghdfe y x1, absorb(id) keepsingletons",
        "psmatch2 d x1, outcome(y) logit ties ate",
        "xtabond y x1, i(id) t(year)",
        "xtreg y x1, fe i(id) vce(robust)",
        "ivregress 2sls y x2 (d = z), vce(cluster id)",
        "csdid y x1, ivar(id) time(t) gvar(g) method(reg) notyet",
        "did_imputation y id t g, horizons(0/2) pretrends(3)",
        "didregress (y x1) (treat), group(id) time(t) wboot",
        "rdrobust y x1, c(1) h(5) vce(cluster id) covs(a b) level(90)",
        "logit b x1, vce(cluster id) level(90)",
        "poisson c x1, exposure(e) robust",
        "ppmlhdfe c x1, absorb(id) cluster(id)",
        "xtabond y x1, lags(2) twostep vce(robust) i(id)",
        "xtabond y x1, lags(2) i(id)",
        "xtdpdsys y x1, lags(2) twostep i(id)",
        "reg y x1 [aw=w], r",
    ],
)
def test_python_code_spells_out_the_same_call_as_arguments(line):
    """The printed code and the tool-call arguments are the same call.

    An argument may be left out of the code only when it equals the sp
    function's own default.
    """
    import inspect

    out = sp.from_stata(line)
    assert out["ok"], out
    keywords = _call_keywords(out["python_code"])
    defaults = inspect.signature(inspect.unwrap(getattr(sp, out["tool"]))).parameters
    for name, value in out["arguments"].items():
        if name in ("formula", "fml"):
            continue
        if name not in keywords:
            assert name in defaults and defaults[name].default == value, (
                name,
                out["python_code"],
            )
        else:
            assert keywords[name] == value, (name, out["python_code"])


def test_xtabond_writes_stata_nonrobust_default_into_the_code():
    # sp.xtabond defaults to robust=True, Stata's xtabond to the GMM SEs
    assert "robust=False" in sp.from_stata("xtabond y x, i(id)")["python_code"]
    assert (
        "robust=True" in sp.from_stata("xtabond y x, i(id) vce(robust)")["python_code"]
    )


def test_iv_varlists_keep_factor_terms_whole():
    # c.x1##c.x2 is translated to "x1 + x2 + x1:x2"; splitting the IV varlist
    # on blanks used to turn it into "x1 + + + x2 + + + x1:x2"
    out = sp.from_stata("ivreghdfe y c.x1##c.x2 (d = z), absorb(id)")
    assert out["arguments"]["formula"] == "y ~ x1 + x2 + x1:x2 | id | d ~ z"
    out = sp.from_stata("ivregress 2sls y i.g (d = z)")
    assert out["arguments"]["formula"] == "y ~ C(g) + (d ~ z)"


def test_ivreghdfe_without_an_iv_block_is_reghdfe():
    a = sp.from_stata("ivreghdfe y x1 x2, absorb(id) cluster(id)")
    b = sp.from_stata("reghdfe y x1 x2, absorb(id) cluster(id)")
    assert a["ok"] and _same_call(a, b)


def test_keepsingletons_reaches_hdfe_ols(df):
    out = sp.from_stata("reghdfe y x, a(id) keepsin")
    assert out["arguments"]["drop_singletons"] is False
    assert out["untranslated_options"] == []
    # one observation per group for the first ten ids: they are singletons
    d = pd.concat([df[df.id >= 10], df[df.id < 10].groupby("id").head(1)])
    kept = _quiet(sp.stata, "reghdfe y x, a(id) keepsingletons", data=d)
    dropped = _quiet(sp.stata, "reghdfe y x, a(id)", data=d)
    assert kept.data_info["nobs"] - dropped.data_info["nobs"] == 10


def test_psmatch2_options_and_its_probit_default():
    out = sp.from_stata("psmatch2 d x1 x2, outcome(y) neighbor(1) logit ate ties")
    assert out["arguments"]["ties"] is True and out["arguments"]["ate"] is True
    assert out["untranslated_options"] == []
    # without `logit` Stata fits a probit score; sp.psmatch2 fits a logit
    assert sp.from_stata("psmatch2 d x, out(y)")["untranslated_options"] == ["probit"]
