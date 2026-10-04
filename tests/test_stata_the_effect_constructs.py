"""Stata grammar met in Huntington-Klein, *The Effect* (2nd ed.), ch. 13-21.

Each construct was either translated into a different model without a
warning or refused although Stata runs it. The numbers that pin the
estimators are in ``tests/reference_parity/test_the_effect_stata_parity.py``;
these tests pin the reading of the commands.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_expr import coefficient_key, evaluate
from statspai.agent._translation._stata_run import StataSession
from statspai.exceptions import MethodIncompatibility


def _formula(command, columns=None):
    out = sp.from_stata(command, columns=columns)
    assert out["ok"], out.get("error")
    return out["arguments"]["formula"]


# --------------------------------------------------- factor-variable notation
@pytest.mark.parametrize(
    "command, formula",
    [
        # a##b##c is the full factorial, not the main effects plus a#b#c
        (
            "reg y c.a##c.b##c.w",
            "y ~ a + b + w + a:b + a:w + b:w + a:b:w",
        ),
        # a repeated continuous variable is a power, also next to a factor
        (
            "reg y i.p##c.x##c.x",
            "y ~ C(p) + x + C(p):x + I(x**2) + C(p):I(x**2)",
        ),
        ("reg y c.x#c.x#c.x", "y ~ I(x**3)"),
        # without a prefix a variable inside # or ## is a factor
        ("reg y a##b", "y ~ C(a) + C(b) + C(a):C(b)"),
        ("reg y a##c.x", "y ~ C(a) + x + C(a):x"),
        ("reg y i.a i.b a#b", "y ~ C(a) + C(b) + C(a):C(b)"),
        ("reg y g#c.x", "y ~ C(g):x"),
    ],
)
def test_interactions(command, formula):
    assert _formula(command) == formula


def test_crossed_factors_without_main_effects_are_refused():
    out = sp.from_stata("reg y a#b")
    assert not out["ok"]
    assert "a##b" in out["error"]


@pytest.mark.parametrize(
    "command, formula",
    [
        (
            "reghdfe y d##ib3.t, a(id t)",
            "y ~ i.d + ib3.t + i.d:ib3.t | id + t",
        ),
        ("reghdfe y c.x##c.x, a(id)", "y ~ x + x:x | id"),
        ("reghdfe y x i.g#c.x, a(id)", "y ~ x + i.g:x | id"),
    ],
)
def test_reghdfe_products(command, formula):
    assert _formula(command) == formula


def test_reghdfe_slope_per_level_is_refused():
    # sp.hdfe_ols drops the base level of g inside g:x; `i.g#c.x` alone has
    # a slope for every level
    assert not sp.from_stata("reghdfe y i.g#c.x, a(id)")["ok"]


def test_margins_at_with_blanks():
    out = sp.from_stata("margins, dydx(x) at(x = 100 z=2)")
    assert out["ok"] and out["arguments"]["at"] == {"x": 100, "z": 2}


# ----------------------------------------------------- abbreviated variables
COLUMNS = ["y", "statessquireindex", "income", "incident", "leg_black"]


def test_variable_names_may_be_abbreviated():
    assert (
        _formula("reg y statessq i.leg_b", COLUMNS)
        == "y ~ statessquireindex + C(leg_black)"
    )
    # a word that fits no column is left for the estimator to report
    assert _formula("reg y zz", COLUMNS) == "y ~ zz"


def test_ambiguous_abbreviation_is_refused():
    out = sp.from_stata("reg y inc", columns=COLUMNS)
    assert not out["ok"] and "ambiguous" in out["error"]


def test_abbreviation_is_not_applied_to_expressions():
    out = sp.from_stata("test statess = inc", columns=COLUMNS)
    assert out["arguments"]["hypothesis"] == "statess = inc"


def test_abbreviated_variable_in_an_expression():
    data = pd.DataFrame({"_treated": [1.0, 0.0], "other": [3.0, 4.0]})
    np.testing.assert_array_equal(evaluate("_treat == 1", data, {}), [1.0, 0.0])
    with pytest.raises(ValueError, match="not in the data"):
        evaluate("nothing + 1", data, {})


# ------------------------------------------------------------------ date()
@pytest.mark.parametrize(
    "text, mask, days",
    [
        ("2015-07-31", "YMD", 20300),  # td(31jul2015)
        ("20150731", "YMD", 20300),
        ("7/31/2015", "MDY", 20300),
        ("31jul2015", "DMY", 20300),
        ("July 31, 2015", "MDY", 20300),
        ("1960-01-01", "YMD", 0),
    ],
)
def test_date_function(text, mask, days):
    data = pd.DataFrame({"x": [0.0]})
    assert evaluate(f'date("{text}", "{mask}")', data, {})[0] == days


def test_date_function_on_a_string_variable_and_bad_input():
    data = pd.DataFrame({"s": ["2015-08-10", "not a date", "2015-02-30"]})
    out = evaluate('date(s, "YMD")', data, {})
    assert out[0] == 20310 and np.isnan(out[1]) and np.isnan(out[2])
    with pytest.raises(ValueError, match="mask"):
        evaluate('date(s, "DM20Y")', data, {})


# -------------------------------------------------------------- data steps
def _session(data):
    return StataSession(data)


def test_encode_generate_abbreviated_to_one_letter():
    session = _session(pd.DataFrame({"s": ["b", "a", "b"], "y": [1.0, 2.0, 3.0]}))
    session.run("encode s, g(code)")
    assert list(session.data["code"]) == [2.0, 1.0, 2.0]


def test_xi_names_follow_stata():
    data = pd.DataFrame(
        {"drgi_culpability": [1, 2, 3, 2], "offense": [4, 4, 7, 9], "s": list("abca")}
    )
    session = _session(data)
    session.run("xi i.drgi_culpability, pre(cul_)")
    session.run("xi i.offense i.s")
    held = session.data
    # prefix plus the first 11 - len(prefix) characters of the name
    assert {"cul_drgi_cu_2", "cul_drgi_cu_3"} <= set(held.columns)
    assert "cul_drgi_cu_1" not in held.columns  # the smallest level is omitted
    assert list(held["cul_drgi_cu_2"]) == [0.0, 1.0, 0.0, 1.0]
    assert {"_Ioffense_7", "_Ioffense_9", "_Is_2", "_Is_3"} <= set(held.columns)


def test_weights_that_are_missing_or_zero_leave_the_sample():
    rng = np.random.default_rng(3)
    data = pd.DataFrame({"y": rng.normal(size=50), "x": rng.normal(size=50)})
    data["w"] = np.r_[np.full(10, np.nan), np.zeros(10), rng.uniform(1, 2, 30)]
    session = _session(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("reg y x [pw = w]")
    kept = sp.regress("y ~ x", data=data.iloc[20:], weights="w", robust="hc1")
    assert int(session.output.nobs) == 30
    np.testing.assert_allclose(session.output.params["x"], kept.params["x"])


def test_importance_weights_only_when_they_are_analytic_weights():
    rng = np.random.default_rng(4)
    data = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)})
    w = rng.uniform(0.5, 1.5, 40)
    data["sums_to_n"] = w * 40 / w.sum()
    data["other"] = w * 3
    session = _session(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("reg y x [iw = sums_to_n]")
        expected = sp.regress("y ~ x", data=data, weights="sums_to_n")
        np.testing.assert_allclose(
            session.output.std_errors["x"], expected.std_errors["x"]
        )
        with pytest.raises(MethodIncompatibility, match="importance weights"):
            session.run("reg y x [iw = other]")


def test_factor_covariates_of_a_list_estimator():
    d = sp.datasets.nsw_dw()
    d["agegrp"] = (d["age"] > 25) * 1 + (d["age"] > 35) * 1
    session = _session(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("psmatch2 treat i.agegrp education, outcome(re78) logit")
        by_hand = sp.psmatch2(
            data=d.assign(g1=(d.agegrp == 1) * 1.0, g2=(d.agegrp == 2) * 1.0),
            treat="treat",
            covariates=["g1", "g2", "education"],
            outcome="re78",
        )
    np.testing.assert_allclose(session.output.att, by_hand.att)
    # psmatch2 leaves the matched outcome as _<outcome>
    assert "_re78" in session.data.columns


# ---------------------------------------------------------- coefficient names
@pytest.mark.parametrize(
    "name",
    [
        "1.d#3.t",
        "3.t#1.d",
        "C(d)[T.1]:C(t)[T.3]",
        "C(d)[T.1.0]:C(t, Treatment(2))[T.3]",
        "d::1.0:t::3.0",
    ],
)
def test_coefficient_key_reads_every_spelling(name):
    assert coefficient_key(name) == "1.d#3.t"


def test_coefficient_key_of_powers_and_slopes():
    assert coefficient_key("1.p#c.x#c.x") == coefficient_key("C(p)[T.1]:I(x ** 2)")
    assert coefficient_key("g[2.0]") == "2.g"


def test_rdrobust_names_a_covariate_that_is_not_numeric():
    rng = np.random.default_rng(5)
    data = pd.DataFrame({"y": rng.normal(size=200), "x": rng.uniform(-1, 1, 200)})
    data["state"] = np.where(data["x"] > 0, "New Jersey", "Ohio")
    with pytest.raises(MethodIncompatibility, match="covariate 'state' is not numeric"):
        sp.rdrobust(data, y="y", x="x", covs=["state"])


# ------------------------------------------------------------------- table
def test_table_statistic_by_one_row_variable():
    out = sp.from_stata("table wc, stat(mean earn)")
    assert out["tool"] == "sumstats"
    assert out["arguments"] == {
        "stats": ["mean"],
        "output": "numeric",
        "vars": ["earn"],
        "by": "wc",
    }
    assert out["untranslated_options"] == []
    # two dimensions, or no statistic, are other tables
    assert not sp.from_stata("table wc hc, stat(mean earn)")["ok"]
    assert not sp.from_stata("table wc")["ok"]


# ---------------------------------------------------------------- programs
PROGRAM = """
capture program drop inner
program def inner, rclass
quietly{
    summarize x
    local m = r(mean)
    local shifted = `m' - 1
}
return scalar diff = `shifted'
end

program define outer, rclass
    inner
    return scalar diff = r(diff)
end

outer
display r(diff)
"""


def test_program_def_returns_a_local_and_may_call_another_program():
    data = pd.DataFrame({"x": np.arange(10.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sp.stata(PROGRAM, data=data) == 3.5
