"""``drdid``, ``csdid ... method(drimp)``, ``csdid_estat`` and ``jwdid`` lines
run through ``sp.stata`` against what Stata 18 printed for the same lines.

The reference file is written by ``_fixtures/_generate_did_commands_Stata.do``
(drdid 1.91, csdid 1.81, jwdid from SSC) on ``did_commands_data.csv``. Every
test hands ``sp.stata`` the command line of the do-file, so a pass says two
things at once: the line is translated to the right call, and the call
computes Stata's number.

Tolerances. Regression-only estimators agree to rounding (1e-9). Estimators
with a logit or tilting propensity score inherit the stopping rule of the
optimiser on each side: Stata's ``logit`` stops at its default ``tolerance``
while StatsPAI iterates to machine precision, which moves the estimate in
about the seventh digit, so those are held to 1e-5 (measured: at most 2e-6).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
_REF = json.loads((_FIX / "did_commands_Stata.json").read_text(encoding="utf-8"))

RTOL_REG = 1e-9
RTOL_PS = 1e-5

_TWO = "if (g == 2004 | g == 0) & (year == 2003 | year == 2005)"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(_FIX / "did_commands_data.csv")


def _run(lines: str, data: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(lines, data)


def _ref(tag: str) -> dict:
    block = _REF[tag]
    return {
        name: (b, se) for name, b, se in zip(block["names"], block["b"], block["se"])
    }


def _close(mine: float, theirs: float, rtol: float) -> None:
    assert mine == pytest.approx(theirs, rel=rtol, abs=1e-10)


# ----------------------------------------------------------------------
# drdid
# ----------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["drimp", "dripw", "reg", "stdipw", "ipw"])
@pytest.mark.parametrize("design", ["panel", "rc"])
def test_drdid_estimators(df, flag, design):
    ivar = "ivar(id) " if design == "panel" else ""
    res = _run(f"drdid y x1 x2 {_TWO}, {ivar}time(year) treatment(d04) {flag}", df)
    b, se = _ref(f"drdid_{design}_{flag}")["ATET:r1vs0.d04"]
    rtol = RTOL_REG if flag == "reg" else RTOL_PS
    _close(res.estimate, b, rtol)
    _close(res.se, se, rtol)


def test_drdid_default_is_the_improved_estimator(df):
    res = _run(f"drdid y x1 x2 {_TWO}, i(id) t(year) tr(d04)", df)
    b, se = _ref("drdid_panel_default")["ATET:r1vs0.d04"]
    assert _ref("drdid_panel_drimp")["ATET:r1vs0.d04"] == (b, se)
    _close(res.estimate, b, RTOL_PS)
    _close(res.se, se, RTOL_PS)


def test_drdid_rc1_on_repeated_cross_sections(df):
    res = _run(f"drdid y x1 x2 {_TWO}, time(year) treatment(d04) dripw rc1", df)
    b, se = _ref("drdid_rc_dripw_rc1")["ATET:r1vs0.d04"]
    _close(res.estimate, b, RTOL_PS)
    _close(res.se, se, RTOL_PS)


# ----------------------------------------------------------------------
# csdid cells
# ----------------------------------------------------------------------

_CELL = re.compile(r"g(\d+):t_(\d+)_(\d+)")


def _check_cells(res, tag: str, rtol: float, *, long2: bool = False) -> None:
    """Match csdid's ``g<cohort>:t_<a>_<b>`` cells to the result's rows.

    With short gaps the cell is dated ``b``. With ``long2`` every cell is
    compared with the period before treatment, so a pre-treatment cell is
    written ``t_<period>_<base>`` and is dated ``a``; the base period itself
    is a row of zeros in the result and absent from Stata's vector.
    """
    ref = {}
    for name, value in _ref(tag).items():
        m = _CELL.fullmatch(name)
        if not m:
            continue
        cohort, first, second = (int(v) for v in m.groups())
        ref[(cohort, first if long2 and second < cohort else second)] = value
    seen = 0
    for _, row in res.detail.iterrows():
        key = (int(row["group"]), int(row["time"]))
        if long2 and key[1] == key[0] - 1:
            assert row["att"] == 0.0
            continue
        b, se = ref[key]
        _close(row["att"], b, rtol)
        _close(row["se"], se, rtol)
        seen += 1
    assert seen == len(ref)


@pytest.mark.parametrize(
    "options, tag",
    [
        ("method(drimp)", "csdid_drimp"),
        ("method(drimp) notyet long2", "csdid_drimp_notyet_long2"),
        ("method(dripw)", "csdid_dripw"),
    ],
)
def test_csdid_cells(df, options, tag):
    res = _run(f"csdid y x1 x2, ivar(id) time(year) gvar(g) {options}", df)
    _check_cells(res, tag, RTOL_PS, long2="long2" in options)


def test_csdid_ignores_an_estimator_written_as_a_bare_option(df):
    """``csdid ..., ipw`` is not ``method(ipw)``: csdid's syntax ends in
    ``*``, which swallows the word, and the default (dripw) runs."""
    assert _REF["csdid_bare_ipw"]["b"] == _REF["csdid_dripw"]["b"]
    out = sp.from_stata("csdid y x1 x2, ivar(id) time(year) gvar(g) ipw")
    assert out["arguments"]["estimator"] == "dr"
    assert out["untranslated_options"] == []
    assert any("has no effect" in note for note in out["notes"])
    res = _run("csdid y x1 x2, ivar(id) time(year) gvar(g) ipw", df)
    _check_cells(res, "csdid_bare_ipw", RTOL_PS)


def test_drimp_differs_from_dr_and_matches_without_covariates(df):
    kw = dict(y="y", i="id", t="year", g="g", base_period="varying")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dr = sp.callaway_santanna(df, x=["x1", "x2"], estimator="dr", **kw)
        imp = sp.callaway_santanna(df, x=["x1", "x2"], estimator="drimp", **kw)
        dr0 = sp.callaway_santanna(df, estimator="dr", **kw)
        imp0 = sp.callaway_santanna(df, estimator="drimp", **kw)
    assert np.max(np.abs(dr.detail["att"] - imp.detail["att"])) > 1e-4
    np.testing.assert_allclose(dr0.detail["att"], imp0.detail["att"], rtol=1e-12)
    np.testing.assert_allclose(dr0.detail["se"], imp0.detail["se"], rtol=1e-12)


# ----------------------------------------------------------------------
# csdid_estat / estat after csdid
# ----------------------------------------------------------------------

_OVERALL = {
    "simple": "ATT",
    "group": "GAverage",
    "calendar": "CAverage",
    "event": "Post_avg",
}


def _row_name(kind: str, value: float) -> str:
    v = int(value)
    if kind == "group":
        return f"G{v}"
    if kind == "calendar":
        return f"T{v}"
    return f"Tm{-v}" if v < 0 else f"Tp{v}"


def _check_aggregate(res, kind: str, tag: str) -> None:
    ref = _ref(tag)
    b, se = ref[_OVERALL[kind]]
    _close(res.estimate, b, RTOL_PS)
    _close(res.se, se, RTOL_PS)
    if kind == "simple":
        return
    column = {"group": "group", "calendar": "time", "event": "relative_time"}[kind]
    seen = 0
    for _, row in res.detail.iterrows():
        b, se = ref[_row_name(kind, row[column])]
        _close(row["att"], b, RTOL_PS)
        _close(row["se"], se, RTOL_PS)
        seen += 1
    assert seen == len(ref) - (2 if kind == "event" else 1)


@pytest.mark.parametrize("kind", ["simple", "group", "calendar", "event"])
def test_csdid_estat(df, kind):
    res = _run(
        "csdid y x1 x2, ivar(id) time(year) gvar(g) method(dripw)\n"
        f"csdid_estat {kind}, post",
        df,
    )
    _check_aggregate(res, kind, f"csdid_dripw_{kind}")


@pytest.mark.parametrize("kind", ["simple", "group", "calendar", "event"])
def test_estat_after_csdid(df, kind):
    res = _run(
        "csdid y x1 x2, ivar(id) time(year) gvar(g) method(drimp) notyet\n"
        f"estat {kind}, post",
        df,
    )
    _check_aggregate(res, kind, f"csdid_drimp_notyet_{kind}")


def test_csdid_estat_event_window(df):
    res = _run(
        "csdid y x1 x2, ivar(id) time(year) gvar(g) method(dripw)\n"
        "csdid_estat event, window(-2 1) post",
        df,
    )
    assert sorted(res.detail["relative_time"]) == [-2, -1, 0, 1]
    _check_aggregate(res, "event", "csdid_dripw_event_window")


def test_group_average_convention_is_the_only_difference_from_aggte(df):
    """``sp.estat(result, 'group')`` holds the cohort shares fixed, as
    csdid does; ``sp.aggte`` by default does not. Same estimate, same rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(
            df, y="y", i="id", t="year", g="g", x=["x1", "x2"], base_period="varying"
        )
        fixed = sp.estat(cs, "group", print_results=False)
        r_did = sp.aggte(cs, type="group")
    assert fixed.estimate == pytest.approx(r_did.estimate, rel=1e-12)
    np.testing.assert_allclose(fixed.detail["se"], r_did.detail["se"], rtol=1e-12)
    assert fixed.se < r_did.se
    _close(fixed.se, _ref("csdid_dripw_group")["GAverage"][1], RTOL_PS)


# ----------------------------------------------------------------------
# jwdid and its estat
# ----------------------------------------------------------------------


def _jw_rows(tag: str, shift: int = 0) -> dict:
    out = {}
    for name, value in _ref(tag).items():
        m = re.match(r"(\d+)", name)
        if m:
            out[int(m.group(1)) - shift] = value
    return out


def test_jwdid_simple(df):
    res = _run("jwdid y, ivar(id) tvar(year) gvar(g)", df)
    b, se = _ref("jwdid_simple")["simple"]
    _close(res.estimate, b, RTOL_REG)
    _close(res.se, se, RTOL_REG)


@pytest.mark.parametrize(
    "kind, column, shift",
    [("group", "cohort", 0), ("calendar", "period", 0), ("event", "relative_time", 5)],
)
def test_estat_after_jwdid(df, kind, column, shift):
    res = _run(f"jwdid y, ivar(id) tvar(year) gvar(g)\nestat {kind}, post", df)
    ref = _jw_rows(f"jwdid_{kind}", shift)
    assert len(res.detail) == len(ref)
    for _, row in res.detail.iterrows():
        b, se = ref[int(row[column])]
        _close(row["att"], b, RTOL_REG)
        _close(row["se"], se, RTOL_REG)


def test_jwdid_never_lists_the_leads(df):
    res = _run("jwdid y, ivar(id) tvar(year) gvar(g) never\nestat event", df)
    ref = _jw_rows("jwdid_never_event", 5)
    rows = {int(r["relative_time"]): r for _, r in res.detail.iterrows()}
    for event, (b, se) in ref.items():
        if event == -1:  # the reference period: zero by construction
            assert b == 0.0
            continue
        _close(rows[event]["att"], b, RTOL_REG)
        _close(rows[event]["se"], se, RTOL_REG)
    assert min(rows) < 0


def test_jwdid_never_simple(df):
    res = _run("jwdid y, ivar(id) tvar(year) gvar(g) never", df)
    b, se = _ref("jwdid_never_simple")["simple"]
    _close(res.estimate, b, RTOL_REG)
    _close(res.se, se, RTOL_REG)


@pytest.mark.parametrize("kind", ["group", "calendar", "event"])
def test_headline_of_a_jwdid_aggregation_is_the_mean_of_its_rows(df, kind):
    """The headline of ``estat group | calendar | event`` is the unweighted
    mean of the rows, with the SE that mean has under the rows' joint
    covariance. Stata's number is the same mean taken through the posted
    ``e(V)``. (Up to 1.38.0 the SE reported next to that mean was the SE of
    ``estat simple``, a different quantity: 0.1064 here against 0.1168 for
    the event rows.)"""
    res = _run(f"jwdid y, ivar(id) tvar(year) gvar(g)\nestat {kind}", df)
    b, se = _ref(f"jwdid_{kind}_mean")["mean"]
    _close(res.estimate, b, RTOL_REG)
    _close(res.se, se, RTOL_REG)
    simple_se = _ref("jwdid_simple")["simple"][1]
    assert abs(res.se - simple_se) > 1e-4
    lo, hi = res.ci
    assert lo < res.estimate < hi


def test_headline_with_leads_averages_the_post_treatment_rows_only(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.jwdid(df, "y", ivar="id", tvar="year", gvar="g", never=True)
        with_leads = sp.etwfe_emfx(fit, type="event", include_leads=True)
        without = sp.etwfe_emfx(fit, type="event")
    assert (with_leads.detail["relative_time"] < 0).any()
    assert with_leads.estimate == pytest.approx(without.estimate, rel=1e-12)
    assert with_leads.se == pytest.approx(without.se, rel=1e-12)
    post = with_leads.detail[with_leads.detail["relative_time"] >= 0]
    assert with_leads.estimate == pytest.approx(post["att"].mean(), rel=1e-12)


# ----------------------------------------------------------------------
# covariates that vary over time
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, tag",
    [
        ("csdid y xt, ivar(id) time(year) gvar(g) method(dripw)", "csdid_tv_dripw"),
        (
            "csdid y xt x2, ivar(id) time(year) gvar(g) method(reg) notyet long2",
            "csdid_tv_reg_notyet_long2",
        ),
        (
            "csdid y xt, ivar(id) time(year) gvar(g) method(drimp) long2",
            "csdid_tv_drimp_long2",
        ),
    ],
)
def test_time_varying_covariate_is_read_in_the_first_period_of_each_cell(df, line, tag):
    """Each ATT(g,t) conditions on the covariate's value in the earlier of
    its two periods (csdid, and R ``did``). Up to 1.38.0 the value in the
    unit's first row of the data was used for every cell."""
    res = _run(line, df)
    _check_cells(res, tag, RTOL_PS, long2="long2" in line)


def test_row_order_does_not_matter_with_a_time_varying_covariate(df):
    kw = dict(y="y", i="id", t="year", g="g", x=["xt"], base_period="varying")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sorted_fit = sp.callaway_santanna(df, **kw)
        shuffled = sp.callaway_santanna(df.sample(frac=1.0, random_state=5), **kw)
    np.testing.assert_allclose(
        sorted_fit.detail["att"], shuffled.detail["att"], rtol=1e-10
    )


def test_constant_covariate_takes_the_same_path_as_before(df):
    """A covariate that is constant within unit gives the same cells whether
    it is read once per unit or period by period."""
    from statspai.did.callaway_santanna import _covariates_by_period

    assert _covariates_by_period(df, "id", "year", ["x1", "x2"], None) is None
    tables = _covariates_by_period(
        df, "id", "year", ["xt", "x1"], pd.Index(sorted(df["id"].unique()))
    )
    assert set(tables) == {"xt", "x1"}
    assert tables["x1"].nunique(axis=1).eq(1).all()


_R_REF = json.loads((_FIX / "did_commands_R.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_time_varying_covariate_against_r_did(df, base_period, control_group):
    """R ``did`` 2.3.0 on the same data (``_generate_did_commands_R.R``):
    the authors' implementation reads the covariate the same way."""
    ref = _R_REF[f"{base_period}_{control_group}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.callaway_santanna(
            df,
            y="y",
            i="id",
            t="year",
            g="g",
            x=["xt"],
            estimator="dr",
            base_period=base_period,
            control_group=control_group,
        )
    mine = {(int(r["group"]), int(r["time"])): r for _, r in res.detail.iterrows()}
    checked = 0
    for group, time, att, se in zip(ref["group"], ref["time"], ref["att"], ref["se"]):
        if se is None:  # the base period of a universal comparison
            assert att == 0
            continue
        row = mine[(int(group), int(time))]
        _close(row["att"], att, 1e-7)
        _close(row["se"], se, 1e-7)
        checked += 1
    assert checked >= 12


# ----------------------------------------------------------------------
# drdid, all
# ----------------------------------------------------------------------


@pytest.mark.parametrize("design", ["panel", "rc"])
def test_drdid_all(df, design):
    """Every row of ``drdid, all``. On a panel Stata also prints ``sipwra``,
    which StatsPAI does not have; the other five rows are compared."""
    ivar = "ivar(id) " if design == "panel" else ""
    res = _run(f"drdid y x1 x2 {_TWO}, {ivar}time(year) treatment(d04) all", df)
    ref = {
        name.split(":")[1]: value for name, value in _ref(f"drdid_{design}_all").items()
    }
    rows = res.detail.set_index("estimator")
    assert set(ref) - set(rows.index) == ({"sipwra"} if design == "panel" else set())
    assert len(rows) == (5 if design == "panel" else 7)
    for name, row in rows.iterrows():
        b, se = ref[name]
        _close(row["att"], b, RTOL_PS)
        _close(row["se"], se, RTOL_PS)
    _close(res.estimate, ref["drimp"][0], RTOL_PS)


# ----------------------------------------------------------------------
# did2s
# ----------------------------------------------------------------------

_DID2S_SETUP = """gen d = g > 0 & year >= g
gen relshift = cond(g > 0, year - g + 10, 0)
gen wt = 1 + x2
gen post0 = relshift == 10
gen post1 = relshift == 11
gen post2 = relshift >= 12 & g > 0
"""

# Stata demeans within unit() in single precision, which shows in the
# ninth digit; the forms without unit() agree to rounding.
_DID2S_LINES = [
    (
        "did2s y, first_stage(i.id i.year) second_stage(i.d) treatment(d) "
        "cluster(id)",
        "did2s_static",
        1e-8,
    ),
    (
        "did2s y, first_stage(i.year) second_stage(i.d) treatment(d) "
        "cluster(id) unit(id)",
        "did2s_static_unit",
        1e-6,
    ),
    (
        "did2s y, first_stage(i.id i.year) second_stage(ib0.relshift) "
        "treatment(d) cluster(id)",
        "did2s_event",
        1e-8,
    ),
    (
        "did2s y, first_stage(i.x2#i.year) second_stage(i.d) treatment(d) "
        "cluster(id) unit(id)",
        "did2s_cell_fe",
        1e-6,
    ),
    (
        "did2s y, first_stage(i.id i.year xt) second_stage(i.d) treatment(d) "
        "cluster(id)",
        "did2s_control",
        1e-8,
    ),
    (
        "did2s y [aw=wt], first_stage(i.id i.year) second_stage(i.d) "
        "treatment(d) cluster(id)",
        "did2s_weighted",
        1e-8,
    ),
    (
        "did2s y, first_stage(i.id i.year) second_stage(post0-post2) "
        "treatment(d) cluster(id)",
        "did2s_dummies",
        1e-8,
    ),
    (
        "did2s y, first_stage(i.id i.year) second_stage(d) treatment(d) " "cluster(x2)",
        "did2s_cluster_x2",
        1e-7,
    ),
]


@pytest.mark.parametrize(
    "line, tag, rtol", _DID2S_LINES, ids=[t for _, t, _ in _DID2S_LINES]
)
def test_did2s(df, line, tag, rtol):
    res = _run(_DID2S_SETUP + line, df)
    ref = {
        ("d" if name == "1.d" else name): value
        for name, value in _ref(tag).items()
        if value != (0.0, 0.0)  # Stata's base level
    }
    rows = res.detail.set_index("term")
    assert set(rows.index) == set(ref)
    for name, (b, se) in ref.items():
        _close(rows.loc[name, "estimate"], b, rtol)
        _close(rows.loc[name, "se"], se, rtol)
    if len(ref) == 1:
        _close(res.estimate, next(iter(ref.values()))[0], rtol)
    else:
        assert np.isnan(res.estimate)


def test_did2s_general_form_contains_the_original_one(df):
    """Unit and period effects with the treatment dummy as the second stage
    is what ``sp.gardner_did(first_treat=)`` has always fitted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classic = sp.gardner_did(df, y="y", group="id", time="year", first_treat="g")
        d = ((df["g"] > 0) & (df["year"] >= df["g"])).astype(int)
        general = sp.gardner_did(
            df.assign(d=d), y="y", treat="d", fe=["id", "year"], cluster="id"
        )
    assert general.estimate == pytest.approx(classic.estimate, rel=1e-10)
    assert general.se == pytest.approx(classic.se, rel=1e-8)
