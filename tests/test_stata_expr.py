"""Stata expressions, ``if`` / ``in`` qualifiers and data steps in sp.stata.

The expected values are Stata's documented semantics ([U] 12.2.1 Missing
values, [U] 13 Functions and expressions), written out by hand: a missing
value compares as larger than any number, is "true", and propagates through
arithmetic. The end-to-end check against real Stata output is the opt-in
log replay in ``tests/external_parity/test_stock_watson_4e_logs.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_expr import (
    StataExprError,
    evaluate,
    in_range_mask,
    sample_mask,
)
from statspai.exceptions import MethodIncompatibility

NA = np.nan


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "x": [1.0, NA, -2.0, 0.0, 5.0],
            "y": [2.0, 3.0, NA, 0.0, 1.0],
            "s": ["M", "F", None, "M", "m"],
            "b": [True, False, True, False, True],
        }
    )


@pytest.mark.parametrize(
    "expr, want",
    [
        # missing is larger than every number
        ("x > 0", [1, 1, 0, 0, 1]),
        ("x >= 5", [0, 1, 0, 0, 1]),
        ("x < .", [1, 0, 1, 1, 1]),
        ("x == .", [0, 1, 0, 0, 0]),
        ("x != .", [1, 0, 1, 1, 1]),
        ("x ~= .", [1, 0, 1, 1, 1]),
        # missing is true; ! of missing is false
        ("!x", [0, 0, 0, 1, 0]),
        ("x & y", [1, 1, 1, 0, 1]),
        ("x | y", [1, 1, 1, 0, 1]),
        ("x == 1 | x == 5", [1, 0, 0, 0, 1]),
        ("(x >= 0) * (x < 5)", [1, 0, 0, 1, 0]),
        # arithmetic propagates missing; x/0 is missing
        ("x + y", [3, NA, NA, 0, 6]),
        ("x / y", [0.5, NA, NA, NA, 5]),
        ("1 / 0", [NA] * 5),
        ("exp(710)", [NA] * 5),
        ("ln(x)", [0, NA, NA, NA, np.log(5)]),
        ("sqrt(x)", [1, NA, NA, 0, np.sqrt(5)]),
        # precedence: -2^2 = -(2^2); ^ associates to the left
        ("-2^2", [-4] * 5),
        ("2^3^2", [64] * 5),
        ("2^-1", [0.5] * 5),
        ("2 + 3 * 4", [14] * 5),
        ("!0 + 1", [2] * 5),
        # subscripts and system variables
        ("x[_n-1]", [NA, 1, NA, -2, 0]),
        ("x[_n+1] - x", [NA, NA, 2, 5, NA]),
        ("x[1]", [1] * 5),
        ("_n", [1, 2, 3, 4, 5]),
        ("_N", [5] * 5),
        # strings: a missing string is ""
        ('s == "M"', [1, 0, 0, 1, 0]),
        ('s != "M"', [0, 1, 1, 0, 1]),
        ('s == ""', [0, 0, 1, 0, 0]),
        # functions
        ("missing(x, y)", [0, 1, 1, 0, 0]),
        ("mi(s)", [0, 0, 1, 0, 0]),
        ("inlist(x, 1, 5)", [1, 0, 0, 0, 1]),
        ("inlist(x, 1, .)", [1, 1, 0, 0, 0]),
        ("inrange(x, 0, 5)", [1, 0, 0, 1, 1]),
        ("inrange(x, 0, .)", [1, 0, 0, 1, 1]),
        ("inrange(x, ., 0)", [0, 0, 1, 1, 0]),
        ("cond(x > 0, 10, 20)", [10, 10, 20, 20, 10]),
        ("cond(x, 10, 20, 30)", [10, 30, 10, 20, 10]),
        ("max(x, y)", [2, 3, -2, 0, 5]),
        ("min(x, y, .)", [1, 3, -2, 0, 1]),
        ("round(2.5)", [3] * 5),
        ("round(-2.5)", [-3] * 5),
        ("round(x, 2)", [2, NA, -2, 0, 6]),
        ("int(-2.7)", [-2] * 5),
        ("floor(-2.7)", [-3] * 5),
        ("mod(7, 3)", [1] * 5),
        ("abs(x)", [1, NA, 2, 0, 5]),
        # a boolean column is 0 / 1
        ("b", [1, 0, 1, 0, 1]),
        ("b == 1 & x > 0", [1, 0, 0, 0, 1]),
    ],
)
def test_expression_values(df, expr, want):
    np.testing.assert_allclose(
        evaluate(expr, df).astype(float), np.array(want, dtype=float), rtol=1e-14
    )


def test_pandas_would_disagree_on_missing(df):
    # the reason this evaluator exists
    assert sample_mask("x > 0", df).tolist() == [True, True, False, False, True]
    assert (df.x > 0).tolist() == [True, False, False, False, True]
    # `if x` keeps a missing x
    assert sample_mask("x", df).tolist() == [True, True, True, False, True]


@pytest.mark.parametrize(
    "expr",
    [
        "x = 1",  # assignment, not equality
        "e(sample)",
        "L.x > 0",
        "_b[x] > 0",
        "strlen(s) > 1",
        "z > 1",  # no such variable
        "x > .a",  # extended missing value
        's > "A"',
        "s + 1",
        "`v' > 0",
        "$g > 0",
        "x >",
        "x > 0)",
        "",
    ],
)
def test_expressions_outside_the_grammar_are_refused(df, expr):
    with pytest.raises(StataExprError):
        evaluate(expr, df)


def test_in_ranges():
    assert in_range_mask("1/3", 5).tolist() == [True, True, True, False, False]
    assert in_range_mask("4", 5).tolist() == [False, False, False, True, False]
    assert in_range_mask("-2/l", 5).tolist() == [False, False, False, True, True]
    assert in_range_mask("f/2", 5).tolist() == [True, True, False, False, False]
    for bad in ("0/3", "3/9", "4/2", "a/b"):
        with pytest.raises(StataExprError):
            in_range_mask(bad, 5)


# ----------------------------------------------------------- in sp.stata
@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(12)
    n = 400
    out = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "w": rng.normal(size=n),
            "year": rng.integers(1982, 1989, n),
            "id": np.repeat(np.arange(40), 10),
        }
    )
    out["y"] = 1 + 0.5 * out.x - 0.2 * out.w + rng.normal(size=n)
    out.loc[:14, "w"] = np.nan
    return out


def _run(text, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(text, data=data)


def test_if_qualifier_matches_filtering_first(panel):
    got = _run("reg y x if year == 1984, r", panel)
    want = sp.regress("y ~ x", data=panel[panel.year == 1984], robust="hc1")
    np.testing.assert_array_equal(got.params.to_numpy(), want.params.to_numpy())
    np.testing.assert_array_equal(got.std_errors.to_numpy(), want.std_errors.to_numpy())


def test_if_qualifier_keeps_rows_where_the_variable_is_missing(panel):
    # `if w > 0` keeps the 15 rows with missing w; y ~ x does not use w, so
    # those rows are in Stata's estimation sample
    got = _run("reg y x if w > 0", panel)
    stata_rows = panel[(panel.w > 0) | panel.w.isna()]
    pandas_rows = panel[panel.w > 0]
    assert got.nobs == len(stata_rows) == len(pandas_rows) + 15
    np.testing.assert_array_equal(
        got.params.to_numpy(), sp.regress("y ~ x", data=stata_rows).params.to_numpy()
    )


def test_in_range_and_combined_qualifiers(panel):
    got = _run("reg y x in 1/100", panel)
    assert got.nobs == 100
    both = _run("reg y x if year >= 1985 in 1/100", panel)
    assert both.nobs == int((panel.year.iloc[:100] >= 1985).sum())


def test_data_steps_feed_the_estimation(panel):
    got = _run(
        """
        set more off
        gen double x2 = x * x
        gen late = (year >= 1986)
        keep if !missing(w)
        reg y x x2 w late, r
        """,
        panel,
    )
    prepared = panel.assign(x2=panel.x**2, late=(panel.year >= 1986).astype(float))
    prepared = prepared[prepared.w.notna()]
    want = sp.regress("y ~ x + x2 + w + late", data=prepared, robust="hc1")
    np.testing.assert_allclose(
        got.params.to_numpy(), want.params.to_numpy(), rtol=1e-12
    )
    # the caller's frame is untouched
    assert "x2" not in panel.columns and len(panel) == 400


def test_generate_stores_single_precision_unless_double():
    data = pd.DataFrame({"u": np.linspace(0.1, 0.9, 9), "y": np.arange(9.0)})
    session_out = _run("gen a = u / 3\ngen double b = u / 3\nsummarize a b", data)
    exact = (data.u / 3).mean()
    single = (data.u / 3).astype(np.float32).astype(float).mean()
    assert session_out.loc["b", "Mean"] == pytest.approx(exact, rel=1e-15)
    assert session_out.loc["a", "Mean"] == pytest.approx(single, rel=1e-15)
    assert single != exact


def test_replace_sort_and_lags(panel):
    got = _run(
        """
        sort id year
        gen double lagx = x[_n-1]
        replace lagx = . if id != id[_n-1]
        reg y x lagx
        """,
        panel,
    )
    ordered = panel.sort_values(["id", "year"], kind="stable").reset_index(drop=True)
    ordered["lagx"] = ordered.groupby("id").x.shift(1)
    want = sp.regress("y ~ x + lagx", data=ordered)
    np.testing.assert_allclose(
        got.params.to_numpy(), want.params.to_numpy(), rtol=1e-12
    )
    assert got.nobs == want.nobs


def test_preserve_restore_brings_the_full_sample_back(panel):
    got = _run(
        """
        preserve
        keep if year == 1982
        reg y x
        restore
        reg y x
        """,
        panel,
    )
    assert got.nobs == len(panel)


def test_post_estimation_uses_the_rows_the_model_was_fitted_on(panel):
    data = panel.assign(d=(panel.y > 1).astype(int))
    got = _run("logit d x if year >= 1985; margins, dydx(x)", data)
    sub = data[data.year >= 1985]
    want = _run("logit d x; margins, dydx(x)", sub)
    pd.testing.assert_frame_equal(got, want)
    whole = _run("logit d x; margins, dydx(x)", data)
    assert not got.equals(whole)


@pytest.mark.parametrize(
    "text, reason",
    [
        ("reg y x if e(sample)", "not applied"),
        ("egen m = mean(x)", "egen"),
        ("use somefile.dta", "use"),
        ('gen s = "a"', "string"),
        ("replace nope = 1", "does not exist"),
        ("gen x = 1", "already exists"),
        ("gen int k = x", "truncates"),
        ("gen str3 k = x", "storage type"),
        ("keep x if year > 1984", "not both"),
        ("drop nope", "not in the data"),
        ("restore", "without a `preserve`"),
        ("gen z = x, nolabel", "options"),
    ],
)
def test_what_is_not_implemented_is_refused(panel, text, reason):
    with pytest.raises(MethodIncompatibility, match=reason):
        _run(text, panel)


def test_skipped_lines_do_not_change_the_result(panel):
    plain = _run("reg y x w, r", panel)
    noisy = _run(
        """
        set more off
        set linesize 200
        log using out.log, replace
        label var x "a regressor"
        describe
        reg y x w, r
        """,
        panel,
    )
    np.testing.assert_array_equal(
        plain.std_errors.to_numpy(), noisy.std_errors.to_numpy()
    )


# ------------------------------------------------------------------ tin()
def test_tin_on_a_datetime_time_variable():
    days = pd.date_range("1979-12-28", periods=10, freq="D")
    data = pd.DataFrame({"day": days, "v": np.arange(10.0)})
    out = _run("tsset day; summarize v if tin(01jan1980,03jan1980)", data)
    assert out.loc["v", "N"] == 3 and out.loc["v", "Mean"] == 5.0  # v = 4, 5, 6
    # an open end
    assert _run("tsset day; summarize v if tin(,31dec1979)", data).loc["v", "N"] == 4
    assert _run("tsset day; summarize v if tin(05jan1980,)", data).loc["v", "N"] == 2


def test_tin_on_period_and_numeric_time_variables():
    q = pd.DataFrame(
        {"q": pd.period_range("1999Q1", periods=8, freq="Q"), "v": np.arange(8.0)}
    )
    assert _run("tsset q; summarize v if tin(1999q3,2000q2)", q).loc["v", "N"] == 4
    assert _run("tsset q; summarize v if tin(2000,)", q).loc["v", "N"] == 4
    # a numeric time variable holding Stata daily dates (days from 1960-01-01)
    stata_days = pd.DataFrame({"d": [7305.0, 7306.0, 7307.0], "v": [1.0, 2.0, 3.0]})
    out = _run("tsset d; summarize v if tin(02jan1980,03jan1980)", stata_days)
    assert out.loc["v", "N"] == 2  # 01jan1980 is day 7305


def test_tin_needs_a_time_variable_and_a_readable_date():
    data = pd.DataFrame(
        {"day": pd.date_range("2020-01-01", periods=3), "v": [1.0, 2, 3]}
    )
    with pytest.raises(MethodIncompatibility, match="time variable"):
        _run("summarize v if tin(01jan2020,02jan2020)", data)
    with pytest.raises(MethodIncompatibility, match="cannot read the date"):
        _run("tsset day; summarize v if tin(2020q1,2020q2)", data)
