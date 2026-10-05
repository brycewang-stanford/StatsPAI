"""``sp.stata``: the data-management, descriptive, survey and programming
commands of an introductory Stata text (Kohler, Kreuter and Haensch,
*Data Analysis Using Stata*, 4th ed.).

Every expectation is computed by hand or with pandas from Stata's
documented definition; the numbers Stata itself returns for the same
commands are in ``tests/reference_parity/test_kohler_kreuter_stata_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession
from statspai.exceptions import MethodIncompatibility


@pytest.fixture()
def df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    out = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "g": np.repeat([1, 2, 3], n // 3),
            "f": rng.integers(0, 2, size=n),
            "x": np.round(rng.normal(50, 10, size=n)),
            "w": np.round(rng.uniform(0.5, 3, size=n), 2),
            "psu": np.repeat(np.arange(1, 13), n // 12),
        }
    )
    out["y"] = np.round(10 + 0.5 * out["x"] + 3 * out["f"] + 4 * rng.normal(size=n))
    out["d"] = (out["y"] > out["y"].median()).astype(int)
    out["name"] = ["Meier, Hans", "Dr. Abel, Zoe", "Öl, Uwe", "a b c"] * (n // 4)
    out.loc[[3, 17, 40], "y"] = np.nan
    return out


def session(df: pd.DataFrame, *lines: str) -> StataSession:
    s = StataSession(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in lines:
            s.run(line)
    return s


# -------------------------------------------------------------- expressions
def test_grouping_functions(df):
    s = session(
        df,
        "gen a = recode(x, 40, 50, 60)",
        "gen b = irecode(x, 40, 50, 60)",
        "gen c = autocode(x, 4, 20, 80)",
    )
    x = df.x.to_numpy()
    assert s.data["a"].tolist() == np.select([x <= 40, x <= 50], [40, 50], 60).tolist()
    assert s.data["b"].tolist() == ((x > 40) * 1 + (x > 50) + (x > 60)).tolist()
    assert s.data["c"].tolist() == np.select([x <= 35, x <= 50, x <= 65],
                                             [35, 50, 65], 80).tolist()  # fmt: skip


def test_string_functions_count_bytes_and_the_u_functions_characters(df):
    s = session(
        df,
        "gen len = strlen(name)",
        "gen ulen = ustrlen(name)",
        'gen pos = strpos(name, ",")',
        "gen last = substr(name, 1, pos - 1) if pos > 0",
        "gen first = strtrim(substr(name, pos + 1, .)) if pos > 0",
        'gen up = strupper(name) + "!"',
        "gen w2 = word(name, 2)",
        "gen nw = wordcount(name)",
        'gen dr = regexm(name, "^Dr")',
        'gen clean = subinstr(name, "Dr. ", "", .)',
    )
    row = s.data.iloc[2]  # "Öl, Uwe": the umlaut is two bytes
    assert (row["len"], row["ulen"], row["pos"]) == (8, 7, 4)
    assert (row["last"], row["first"], row["w2"]) == ("Öl", "Uwe", "Uwe")
    assert s.data["up"].iloc[0] == "MEIER, HANS!"
    assert s.data["last"].iloc[3] == "" and s.data["nw"].iloc[3] == 3
    assert s.data["dr"].tolist()[:4] == [0, 1, 0, 0]
    assert s.data["clean"].iloc[1] == "Abel, Zoe"


def test_dates_clock_and_storage_helpers(df):
    s = StataSession(df)
    for line, value in (
        ("display doy(mdy(3, 15, 2020))", 75),
        ("display week(mdy(12, 31, 2020))", 52),
        ("display halfyear(mdy(7, 1, 2020))", 2),
        ('display hh(clock("14:30", "hm"))', 14),
        ('display mm(clock("14:30", "hm"))', 30),
        ('display dofc(clock("2020-03-15 14:30:00", "YMDhms"))', 21989),
        ("display hours(msofhours(3))", 3),
        ("display c(maxint)", 32740),
        ("display float(16777217)", 16777216),
        ('display real("12.5") + real("x")', np.nan),
        ("display comb(10, 3)", 120),
        ("display binomial(10, 3, .5)", 0.171875),
    ):
        s.run(line)
        assert s.output == pytest.approx(value, nan_ok=True), line
    s.run("display %td 21989")
    assert s.output == "15mar2020"
    s.run("display %tdCCYY-NN-DD 21989")
    assert s.output == "2020-03-15"
    s.run('display "mean " %5.2f 3.14159 " of " 2 + 2 " items"')
    assert s.output == "mean 3.14 of 4 items"
    s.run('display "the value is " 7')
    assert s.output == 7  # one number, with text around it


def test_string_variables_are_generated_and_replaced(df):
    s = session(df, 'gen tag = ""', 'replace tag = "low" if x < 50',
                'replace tag = "high" in 1', "gen str4 code = string(g)")  # fmt: skip
    assert s.data["tag"].iloc[0] == "high"
    assert set(s.data["tag"]) == {"", "low", "high"}
    assert s.data["code"].tolist()[:1] == ["1"]
    with pytest.raises(MethodIncompatibility, match="type mismatch"):
        s.run("replace tag = 5")


# ------------------------------------------------------ extended missing
def test_extended_missing_values_are_one_kind_until_told_apart(df):
    s = session(df, "gen z = x", "replace z = . in 1/5")
    # every missing value is `.` here: the comparisons are Stata's
    for line, count in (("count if z == .", 5), ("count if z == .a", 0),
                        ("count if z < .a", len(df)), ("count if z >= .a", 0),
                        ("count if z != .", len(df) - 5)):  # fmt: skip
        s.run(line)
        assert s.output == count, line
    s.run("mvdecode z, mv(50 = .c \\ 51 52 = .)")
    assert s.data["z"].isna().sum() == 5 + int(df.x.iloc[5:].isin([50, 51, 52]).sum())
    # z now holds `.` and `.c`, which the data keep as one
    for line in ("count if z == .", "count if z != .", "count if z == .c",
                 "tabulate z, missing", "bysort z: gen k = _N"):  # fmt: skip
        with pytest.raises(MethodIncompatibility, match="extended missing|kinds of"):
            s.run(line)
    for line in ("count if missing(z)", "count if z >= .", "count if z < ."):
        s.run(line)  # the same for every kind of missing value
    s.run("mvencode z, mv(-9)")
    assert not s.data["z"].isna().any()
    s.run("count if z == .")  # no kind is left
    assert s.output == 0


def test_a_frame_that_says_which_variables_are_coded_is_guarded(df):
    coded = df.copy()
    coded.attrs["_ext_missing"] = ["y"]
    s = StataSession(coded)
    with pytest.raises(MethodIncompatibility, match="kinds of missing"):
        s.run("count if y != .")
    s.run("recode y (missing = .), gen(y2)")  # one kind afterwards
    s.run("count if y2 == .")
    assert s.output == 3


# ----------------------------------------------------------- data steps
def test_recode_takes_the_first_rule_that_fits(df):
    s = session(
        df,
        "recode x (min/45 = 1) (46/55 = 2) (56/max = 3), gen(x3)",
        "recode g (1 2 = 0) (else = 1), gen(g2)",
        "recode f (0 = 1) (1 = 0)",
        "recode y (missing = -1) (nonmissing = 1) if g == 1, gen(seen)",
    )
    x = df.x.to_numpy()
    assert s.data["x3"].tolist() == np.select([x <= 45, x <= 55], [1, 2], 3).tolist()
    assert s.data["g2"].tolist() == (df.g == 3).astype(float).tolist()
    assert s.data["f"].tolist() == (1 - df.f).tolist()
    seen = s.data["seen"]
    assert seen[df.g != 1].isna().all() and seen.iloc[3] == -1 and seen.iloc[0] == 1


def test_quantile_groups_and_conversions(df):
    s = session(df, "xtile q = y, nquantiles(4)", "pctile p = y, nquantiles(4)",
                "tostring g, generate(gs)", "destring gs, generate(gn)")  # fmt: skip
    y = np.sort(df.y.dropna().to_numpy())
    n = len(y)
    cuts = [(y[n * k // 4 - 1] + y[n * k // 4]) / 2 if (n * k) % 4 == 0
            else y[int(np.ceil(n * k / 4)) - 1] for k in (1, 2, 3)]  # fmt: skip
    assert s.data["p"].iloc[:3].tolist() == cuts
    expected = np.searchsorted(cuts, df.y.dropna(), side="left") + 1
    assert s.data["q"].dropna().tolist() == expected.tolist()
    assert s.data["gs"].iloc[0] == "1" and s.data["gn"].tolist() == df.g.tolist()


def test_rows_and_columns(df):
    s = session(df, "order y, first", "expand 2 if id <= 3", "duplicates report id")
    assert s.data.columns[0] == "y" and len(s.data) == len(df) + 3
    assert s.output.loc[2, "observations"] == 6 and s.output.loc[2, "surplus"] == 3
    s = session(df, "contract g f", "separate _freq, by(f)")
    assert s.data["_freq"].sum() == len(df)
    assert s.data["_freq1"].notna().tolist() == (s.data["f"] == 1).tolist()
    s = session(df, 'split name, parse(",") generate(part)', "levelsof g, local(K)",
                "rename (x w) (age weight)", "drop id - f")  # fmt: skip
    assert s.data["part1"].iloc[1] == "Dr. Abel" and s.data["part2"].iloc[1] == "Zoe"
    assert s._macros.locals["K"] == "1 2 3"
    assert list(s.data.columns)[:2] == ["age", "weight"]


def test_levelsof_keeps_sixteen_digits_like_stata():
    d = pd.DataFrame({"v": np.array([8.8, 16.1, 3.0], dtype=np.float32)})
    s = session(d, "levelsof v, local(L)")
    assert s._macros.locals["L"] == "3 8.800000190734863 16.10000038146973"
    s.run("count if v == 16.10000038146973")  # no longer the stored float
    assert s.output == 0


def test_by_prefix_filters_and_runs_commands_by_group(df):
    s = session(df, "bysort g (y): keep if _n == 1")
    first = df.sort_values(["g", "y"], kind="stable").groupby("g").head(1)
    assert s.data["id"].tolist() == first["id"].tolist()
    s = session(df, "bysort g: tabulate f d, chi2")
    assert set(s.output) == {1, 2, 3}
    assert "chi2" in s.output[1].attrs["test"]


def test_checks_set_the_return_code(df):
    s = StataSession(df)
    for line, code in (
        ("capture assert y > 0 if !missing(y)", 0),
        ("capture assert y > 40", 9),
        ("capture confirm variable nosuch", 111),
        ("capture confirm new variable y", 110),
        ("capture confirm numeric variable name", 198),
        ("capture isid g", 459),
        ("capture isid id", 0),
    ):
        s.run(line)
        s.run("display _rc")
        assert s.output == code, line
    with pytest.raises(MethodIncompatibility, match="assertion is false"):
        s.run("assert y > 40")


# ------------------------------------------------------------- tables
def test_tabulate_percentages_tests_weights_and_summaries(df):
    s = session(df, "tabulate g f, row column cell expected chi2 gamma taub")
    table = s.output
    counts = pd.crosstab(df.g, df.f)
    assert table.loc[2, 1] == counts.loc[2, 1]
    assert table.attrs["row"].loc[2, 1] == pytest.approx(100 * counts.loc[2, 1] / 40)
    assert table.attrs["column"].loc["Total", 1] == 100
    assert table.attrs["expected"].loc[1, 0] == pytest.approx(
        40 * counts[0].sum() / 120
    )
    assert {"chi2", "gamma", "taub"} <= set(table.attrs["test"])
    s.run("tabulate g [aweight = w]")
    share = df.groupby("g").w.sum() / df.w.sum()
    assert s.output["Percent"].tolist() == pytest.approx((100 * share).tolist())
    assert s.output["Freq."].sum() == pytest.approx(len(df))
    s.run("tabulate g, summarize(y)")
    assert s.output.loc[1, "Mean"] == pytest.approx(df.y[df.g == 1].mean())
    assert s.output.loc["Total", "Freq."] == df.y.notna().sum()
    s.run("tab2 g f d")
    assert set(s.output) == {"g f", "g d", "f d"}
    s.run("table g, statistic(frequency) statistic(mean y)")
    assert s.output.loc[1, "frequency"] == 40  # rows, whatever y holds
    assert s.output.loc["Total", "mean y"] == pytest.approx(df.y.mean())


def test_means_follow_the_variance_stata_uses(df):
    d = df.dropna(subset=["y"])
    s = session(df, "mean y, over(g)")
    part = d.y[d.g == 2]
    row = s.output.xs("y").iloc[1]
    assert row["se"] == pytest.approx(
        part.std() / np.sqrt(len(part))
    )  # SRS in the group
    s.run("proportion f")
    p = df.f.mean()
    assert s.output["se"].iloc[1] == pytest.approx(np.sqrt(p * (1 - p) / len(df)))
    s.run("total y")
    assert s.output["se"].iloc[0] == pytest.approx(np.sqrt(len(d)) * d.y.std())
    s.run("mean y, vce(cluster psu)")
    t = (d.y - d.y.mean()).groupby(d.psu).sum() / len(d)
    m = d.psu.nunique()
    assert s.output["se"].iloc[0] == pytest.approx(np.sqrt(m / (m - 1) * (t**2).sum()))
    s.run("display _b[y]")
    assert s.output == pytest.approx(d.y.mean())


def test_svy_uses_the_design_and_leaves_lonely_strata_missing(df):
    design = sp.svydesign(df.dropna(subset=["y"]), weights="w", strata="g",
                          cluster="psu", nest=True)  # fmt: skip
    ref = sp.svymean("y", design)
    s = session(df, "svyset psu [pweight = w], strata(g)", "svy: mean y")
    assert s.output["estimate"].iloc[0] == pytest.approx(float(ref.estimate.iloc[0]))
    assert s.output["se"].iloc[0] == pytest.approx(float(ref.std_error.iloc[0]))
    s.run("svy: tabulate f d")
    assert s.output["cell"].loc["Total", "Total"] == pytest.approx(1.0)
    assert s.output["F"] > 0 and s.output["df1"] == pytest.approx(1.0)
    lonely = df.assign(g=np.where(df.psu == 1, 9, df.g))
    s = session(lonely, "svyset psu [pweight = w], strata(g)", "svy: mean y")
    assert np.isnan(s.output["se"].iloc[0])  # Stata's default: singleunit(missing)
    s.run("svyset psu [pweight = w], strata(g) singleunit(certainty)")
    s.run("svy: mean y")
    assert s.output["se"].iloc[0] > 0
    with pytest.raises(MethodIncompatibility, match="svyset"):
        session(df, "svy: mean y")


# ------------------------------------------------------ after estimation
def test_predict_diagnostics_and_the_estimation_sample(df):
    s = session(df, "regress y x f", "predict cook, cooksd", "predict rst, rstudent",
                "predict sp, stdp", "dfbeta", "count if e(sample)")  # fmt: skip
    fit = sp.regress("y ~ x + f", data=df.dropna(subset=["y"]))
    infl = sp.influence_measures(fit)
    used = df.y.notna().to_numpy()
    assert s.output == used.sum()
    assert np.allclose(s.data.loc[used, "cook"], infl["cooksd"])
    assert s.data.loc[~used, "rst"].isna().all()
    assert s.data["sp"].notna().all()  # the prediction's s.e. needs no outcome
    assert np.allclose(s.data.loc[used, "_dfbeta_1"], infl["dfbeta_x"])
    s.run("sort y")
    with pytest.raises(MethodIncompatibility, match="e\\(sample\\)"):
        s.run("count if e(sample)")


def test_logistic_postestimation(df):
    s = session(df, "logit d x f", "estat gof, group(5)")
    fit = sp.logit("d ~ x + f", data=df)
    assert s.output["statistic"] == pytest.approx(
        sp.logit_gof(fit, groups=5)["statistic"]
    )
    s.run("lroc")
    assert 0.5 < s.output["area"] < 1
    s.run("predict db, dbeta")
    assert np.allclose(s.data["db"], sp.logit_influence(fit)["dbeta"])
    s.run("display e(r2_p)")
    p = df.d.mean()
    ll0 = len(df) * (p * np.log(p) + (1 - p) * np.log(1 - p))
    assert s.output == pytest.approx(1 - fit.diagnostics["Log-Likelihood"] / ll0)
    s.run("estimates store full")
    s.run("logit d x")
    s.run("lrtest full .")
    assert s.output["df"] == 1 and s.output["chi2"] > 0
    s.run("logistic d x f")  # the same model, reported as odds ratios
    assert np.allclose(s.output.params.to_numpy(), fit.params.to_numpy(), atol=1e-8)


def test_margins_over_a_grid_and_by_factor_levels(df):
    s = session(df, "regress y c.x##i.f", "margins f, at(x = (40 60))")
    fit = sp.regress("y ~ x * C(f)", data=df.dropna(subset=["y"]))
    ref = sp.margins_at(fit, data=df.dropna(subset=["y"]),
                        at={"x": [40, 60], "f": [0, 1]})  # fmt: skip
    assert np.allclose(s.output["margin"], ref["margin"])
    assert np.allclose(s.output["se"], ref["se"])
    s.run("margins, dydx(x) at(f = (0 1))")
    assert len(s.output) == 2 and s.output["f"].tolist() == [0.0, 1.0]
    s.run("margins")
    assert s.output["margin"].iloc[0] == pytest.approx(df.y.mean())
    with pytest.raises(MethodIncompatibility, match="pwcompare"):
        s.run("margins f, pwcompare")


# ------------------------------------------------------------ programming
def test_extended_macro_functions(df):
    labelled = df.copy()
    labelled.attrs["_labels"] = {"y": "Outcome"}
    s = session(
        labelled,
        "label define yn 0 no 1 yes",
        "label values f yn",
        "local list a b c d",
        "local n : word count `list'",
        "local third : word 3 of `list'",
        "local lab : variable label y",
        "local vl : value label f",
        "local one : label (f) 1",
        'local sub : subinstr local list "b" "B"',
        "local other b c x",
        "local both : list list & other",
        "local size : list sizeof list",
        "local kind : type name",
        "local fmt : display %6.2f 2/3",
        "local i = 5",
        'local shown "`i++\'"',
    )
    got = s._macros.locals
    assert (got["n"], got["third"], got["lab"], got["vl"], got["one"]) == (
        "4", "c", "Outcome", "yn", "yes")  # fmt: skip
    assert (got["sub"], got["both"], got["size"]) == ("a B c d", "b c", "4")
    assert got["kind"].startswith("str") and got["fmt"] == ".67"
    assert (got["shown"], got["i"]) == ("5", "6")
    s.run("display \"`: word 2 of `list''\"")
    assert s.output == "b"


def test_a_program_with_syntax_runs_like_a_command(df):
    s = session(
        df,
        "program mymean, rclass",
        "syntax varlist(min=1 numeric) [if] [in] [, BY(varname) Level(real 95) "
        "noHEADer GENerate(name) *]",
        "marksample touse",
        "quietly count if `touse'",
        "return scalar N = r(N)",
        "return local spec \"`varlist'|`by'|`level'|`header'|`generate'|`options'\"",
        "end",
        "mymean y x if f == 1, by(g) lev(90) nohead foo(1)",
    )
    used = (df.f == 1) & df.y.notna() & df.x.notna()
    assert s.stored["r"]["N"] == used.sum()
    assert s.stored["r_macros"]["spec"] == "y x|g|90|noheader||foo(1)"
    assert not [c for c in s.data.columns if str(c).startswith("__temp")]
    for line, message in (
        ("mymean name", "string variables not allowed"),
        ("mymean y, by(nosuch)", "not found"),
        ("mymean", "varlist required"),
        ("mymean y, level(x)", "incorrectly specified"),
    ):
        with pytest.raises(MethodIncompatibility, match=message):
            s.run(line)


def test_command_line_tools_and_one_line_if(df):
    s = session(
        df,
        "program parts",
        "tokenize `0'",
        "local a `2'`1'",
        "gettoken first rest : 0",
        "macro shift",
        "return local out \"`a'|`first'|`1'|`3'\"",
        'if "`1\'" == "" exit',
        "return scalar reached = 1",
        "end",
        "parts x y",
    )
    assert s.stored["r_macros"]["out"] == "yx|x|y|"
    assert s.stored["r"]["reached"] == 1
    s.run("local k = 3")
    s.run("if `k' > 2 display 10")
    assert s.output == 10
    s.run("else display 20")  # not taken
    assert s.output == 10
    s.run("if `k' > 5 display 30")
    s.run("else display 40")
    assert s.output == 40


# ------------------------------------------------- the second round
def test_linear_combinations_after_mean(df):
    s = session(df, "mean y, over(g)", "lincom _b[c.y@1.g] - _b[c.y@3.g]")
    d = df.dropna(subset=["y"])
    a, b = d.y[d.g == 1], d.y[d.g == 3]
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    assert s.output["estimate"] == pytest.approx(a.mean() - b.mean())
    assert s.output["se"] == pytest.approx(se)  # independent groups
    s.run("test _b[c.y@1.g] = _b[c.y@3.g]")
    assert s.output["F"] == pytest.approx(((a.mean() - b.mean()) / se) ** 2)
    assert s.output["df_r"] == len(d) - 1
    with pytest.raises(MethodIncompatibility, match="not linear"):
        s.run("lincom _b[c.y@1.g] * _b[c.y@3.g]")


def test_nested_blocks_and_one_way_anova(df):
    s = session(df, "nestreg: regress y (x) (f w)")
    d = df.dropna(subset=["y"])
    small, big = sp.regress("y ~ x", data=d), sp.regress("y ~ x + f + w", data=d)
    rss0, rss1 = small.data_info["rss"], big.data_info["rss"]
    f2 = (rss0 - rss1) / 2 / (rss1 / big.data_info["df_resid"])
    assert s.output.loc[2, "F"] == pytest.approx(f2)
    assert s.output.loc[2, "change_r2"] == pytest.approx(
        (rss0 - rss1) / small.data_info["tss"]
    )
    s.run("display _b[w]")  # the full model is the last estimates
    assert s.output == pytest.approx(big.params["w"])
    s.run("anova y g")
    assert s.output.statistic == pytest.approx(sp.oneway(df, "y", by="g").statistic)
    with pytest.raises(MethodIncompatibility, match="testparm"):
        s.run("anova y g f")


def test_epidemiological_tables(df):
    s = session(df, "cc d f")
    t = pd.crosstab(df.d, df.f)
    a, b, c, d_ = t.loc[1, 1], t.loc[1, 0], t.loc[0, 1], t.loc[0, 0]
    out = s.output
    assert out["or"] == pytest.approx(a * d_ / (b * c))
    assert out["lb_or"] < out["or"] < out["ub_or"]
    # the exact limits put alpha / 2 in each tail of the conditional law
    from scipy import stats

    low = stats.nchypergeom_fisher(len(df), a + c, a + b, out["lb_or"])
    assert low.sf(a - 1) == pytest.approx(0.025, abs=1e-9)
    s.run("cs d f")
    r1, r0 = a / (a + c), b / (b + d_)
    assert s.output["rd"] == pytest.approx(r1 - r0)
    assert s.output["rr"] == pytest.approx(r1 / r0)


def test_statsby_and_outcome_probabilities(df):
    s = session(df, "statsby m = r(mean) n = r(N), by(g) clear: summarize y")
    assert s.data["g"].tolist() == [1, 2, 3]
    assert s.data["m"].tolist() == pytest.approx(df.groupby("g").y.mean().tolist())
    assert s.data["n"].sum() == df.y.notna().sum()
    d = df.assign(o=pd.qcut(df.x, 3, labels=False) + 1)
    s = session(d, "ologit o w f", "predict p1 p2 p3")
    total = s.data[["p1", "p2", "p3"]].sum(axis=1)
    assert np.allclose(total, 1.0)
    s = session(d, "mlogit o w f", "predict q1 q2 q3")
    assert np.allclose(s.data[["q1", "q2", "q3"]].sum(axis=1), 1.0)
    # the average probability of an outcome is its share (the score equations)
    assert s.data["q2"].mean() == pytest.approx((d.o == 2).mean(), abs=1e-6)
    with pytest.raises(MethodIncompatibility, match="one new variable per outcome"):
        s.run("predict only")
