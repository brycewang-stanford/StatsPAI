"""``egen`` in ``sp.stata``.

Each function is tested against its definition in ``[D] egen``, with the
missing-value rule the manual states. The implementation was also run
beside Stata 18 on 614 rows with missing values planted in them: 27 of 30
commands agree to 1e-15 on every row, and the other three depend on the
order of rows within a group, which Stata's sort does not fix (``tag``
marks one row per group on both sides; which one is arbitrary).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession
from statspai.exceptions import MethodIncompatibility

NAN = np.nan


@pytest.fixture()
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 3, 3, 3.0],
            "h": [1, 1, 2, 1, NAN, 2, 2, 2],
            "x": [1.0, 2.0, NAN, 4.0, 6.0, NAN, NAN, NAN],
            "a": [1.0, NAN, 3.0, NAN, 5.0, 6.0, NAN, 8.0],
            "b": [2.0, NAN, 1.0, 4.0, NAN, 6.0, NAN, 0.0],
            "c": [3.0, NAN, 5.0, NAN, 7.0, 9.0, 1.0, 4.0],
        }
    )


def held(data: pd.DataFrame, *lines: str) -> pd.DataFrame:
    session = StataSession(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in lines:
            session.run(line)
    return session.data


def column(data: pd.DataFrame, line: str) -> list:
    name = line.split("=")[0].split()[-1]
    return held(data, line)[name].tolist()


def same(got: list, want: list) -> None:
    np.testing.assert_allclose(
        np.array(got, dtype=float), np.array(want, dtype=float), equal_nan=True
    )


# ---------------------------------------------------- statistics by group
def test_mean_ignores_missing_and_fills_the_group(df):
    same(
        column(df, "egen double m = mean(x), by(g)"),
        [1.5, 1.5, 1.5, 5, 5, NAN, NAN, NAN],
    )
    same(column(df, "egen double m = mean(x)"), [3.25] * 8)


def test_total_counts_missing_as_zero(df):
    same(column(df, "egen t = total(x), by(g)"), [3, 3, 3, 10, 10, 0, 0, 0])
    same(
        column(df, "egen t = total(x), by(g) missing"),
        [3, 3, 3, 10, 10, NAN, NAN, NAN],
    )
    same(column(df, "egen t = sum(x), by(g)"), [3, 3, 3, 10, 10, 0, 0, 0])


def test_count_is_zero_not_missing(df):
    same(column(df, "egen n = count(x), by(g)"), [2, 2, 2, 2, 2, 0, 0, 0])


def test_min_max_sd_median(df):
    same(column(df, "egen v = max(x), by(g)"), [2, 2, 2, 6, 6, NAN, NAN, NAN])
    same(column(df, "egen v = min(x), by(g)"), [1, 1, 1, 4, 4, NAN, NAN, NAN])
    sd = np.sqrt(0.5)
    same(
        column(df, "egen double v = sd(x), by(g)"),
        [sd, sd, sd, np.sqrt(2), np.sqrt(2), NAN, NAN, NAN],
    )
    same(
        column(df, "egen double v = median(x), by(g)"),
        [1.5, 1.5, 1.5, 5, 5, NAN, NAN, NAN],
    )
    # one value: no standard deviation
    one = pd.DataFrame({"g": [1, 2, 2.0], "x": [1.0, 2.0, 4.0]})
    same(column(one, "egen double v = sd(x), by(g)"), [NAN, np.sqrt(2), np.sqrt(2)])


def test_pctile_and_iqr_use_statas_percentile():
    d = pd.DataFrame({"x": np.arange(1.0, 11.0)})
    assert column(d, "egen p = pctile(x), p(25)") == [3.0] * 10
    assert column(d, "egen p = pctile(x)") == [5.5] * 10
    assert column(d, "egen q = iqr(x)") == [5.0] * 10  # 8 - 3


def test_expression_argument_and_if(df):
    same(
        column(df, "egen double m = mean(x * 2 + 1), by(g)"),
        [4, 4, 4, 11, 11, NAN, NAN, NAN],
    )
    # the statistic is over the selected rows, which alone receive it
    same(
        column(df, "egen double m = mean(x) if x > 1, by(g)"),
        [NAN, 2, 2, 5, 5, NAN, NAN, NAN],
    )


def test_by_prefix_equals_by_option(df):
    a = held(df, "bysort g: egen double m = mean(x)")
    b = held(df, "egen double m = mean(x), by(g)")
    same(a.sort_values("g", kind="stable")["m"].tolist(), b["m"].tolist())
    # behind the prefix the expression counts within the group
    first = held(df, "bysort g: egen double f = total(c * (_n == 1))")
    same(first["f"].tolist(), [3, 3, 3, 0, 0, 9, 9, 9])


def test_missing_by_value_is_a_group(df):
    # Stata: by() treats missing as a group of its own
    same(
        column(df, "egen n = count(c), by(h)"),
        [1, 1, 4, 1, 1, 4, 4, 4],
    )


def test_result_is_single_precision_unless_double():
    d = pd.DataFrame({"x": [0.1, 0.2]})
    single = column(d, "egen m = mean(x)")[0]
    double = column(d, "egen double m = mean(x)")[0]
    assert single == float(np.float32(double)) and single != double


# ----------------------------------------------------------- group and tag
def test_group_numbers_sorted_combinations(df):
    same(column(df, "egen id = group(g h)"), [1, 1, 2, 3, NAN, 4, 4, 4])
    same(column(df, "egen id = group(g h), missing"), [1, 1, 2, 3, 4, 5, 5, 5])
    same(column(df, "egen id = group(g)"), [1, 1, 1, 2, 2, 3, 3, 3])


def test_group_of_a_string_variable():
    d = pd.DataFrame({"s": ["b", "a", "b", "c"]})
    assert column(d, "egen id = group(s)") == [2, 1, 2, 3]


def test_tag_marks_one_row_per_group_and_is_never_missing(df):
    out = held(df, "egen t = tag(g h)", "egen tm = tag(g h), missing")
    assert out["t"].tolist() == [1, 0, 1, 1, 0, 1, 0, 0]
    assert out["tm"].tolist() == [1, 0, 1, 1, 1, 1, 0, 0]
    same(column(df, "egen t = tag(g) if x < ."), [1, 0, 0, 1, 0, 0, 0, 0])


# ----------------------------------------------------------- row functions
def test_row_functions(df):
    out = held(
        df,
        "egen double rt = rowtotal(a b c)",
        "egen double rtm = rowtotal(a b c), missing",
        "egen double rm = rowmean(a b c)",
        "egen double lo = rowmin(a b c)",
        "egen double hi = rowmax(a b c)",
        "egen k = rownonmiss(a b c)",
        "egen miss = rowmiss(a b c)",
        "egen double rs = rowsd(a b c)",
    )
    same(out["rt"].tolist(), [6, 0, 9, 4, 12, 21, 1, 12])
    same(out["rtm"].tolist(), [6, NAN, 9, 4, 12, 21, 1, 12])
    same(out["rm"].tolist(), [2, NAN, 3, 4, 6, 7, 1, 4])
    same(out["lo"].tolist(), [1, NAN, 1, 4, 5, 6, 1, 0])
    same(out["hi"].tolist(), [3, NAN, 5, 4, 7, 9, 1, 8])
    same(out["k"].tolist(), [3, 0, 3, 1, 2, 3, 1, 3])
    same(out["miss"].tolist(), [0, 3, 0, 2, 1, 0, 2, 0])
    same(out["rs"].tolist(), [1, NAN, 2, NAN, np.sqrt(2), np.sqrt(3), NAN, 4])


def test_row_varlist_ranges_and_patterns(df):
    wide = df.rename(columns={"a": "v1", "b": "v2", "c": "v3"})
    want = column(wide, "egen k = rownonmiss(v1 v2 v3)")
    assert column(wide, "egen k = rownonmiss(v1-v3)") == want
    assert column(wide, "egen k = rownonmiss(v*)") == want


def test_std(df):
    z = column(df, "egen double z = std(c)")
    c = df["c"].to_numpy()
    same(z, (c - np.nanmean(c)) / np.nanstd(c, ddof=1))


# ------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    "line, message",
    [
        ("egen s = fill(1 2)", "not implemented"),
        ("egen r = rank(x), unique", "not implemented"),
        ("egen q = cut(x), group(4)", "not implemented"),
        ("egen m = mean(x), by(nope)", "not in the data"),
        ("egen g = mean(x)", "already exists"),
        ("egen id = group(g), by(h)", "may not be combined with by"),
        ("egen m = mean(x), by(g) weird", "not implemented"),
        ("egen k = rowtotal(a zz)", "not in the data"),
        ("egen int m = mean(x)", "truncates"),
        ("egen m = mean(x", "parentheses"),
    ],
)
def test_what_is_not_implemented_is_refused(df, line, message):
    with pytest.raises(MethodIncompatibility, match=message):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.stata(line, data=df)


def test_egen_feeds_an_estimation_command():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"id": np.repeat(np.arange(40), 5)})
    d["x"] = rng.normal(size=200) + rng.normal(size=40)[d.id]
    d["y"] = 0.5 * d.x + rng.normal(size=40)[d.id] + rng.normal(size=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        within = sp.stata(
            "egen double xbar = mean(x), by(id)\negen double ybar = mean(y), by(id)\n"
            "gen double xd = x - xbar\ngen double yd = y - ybar\nreg yd xd",
            data=d,
        )
        fe = sp.regress("y ~ x + C(id)", data=d)
    assert float(within.params["xd"]) == pytest.approx(float(fe.params["x"]), rel=1e-10)


# ------------------------------------------------- the rest of the functions
# Run beside Stata 18 on the same 614 rows as the functions above: all 19
# commands agree on every row.
def test_rank_ties_share_the_average(df):
    d = pd.DataFrame({"g": [1, 1, 1, 1, 2, 2.0], "x": [3.0, 1.0, 3.0, NAN, 5.0, 4.0]})
    same(column(d, "egen r = rank(x)"), [2.5, 1, 2.5, NAN, 5, 4])
    same(column(d, "egen r = rank(x), by(g)"), [2.5, 1, 2.5, NAN, 2, 1])
    same(column(d, "egen r = rank(x), field"), [3, 5, 3, NAN, 1, 2])
    same(column(d, "egen r = rank(x), track"), [2, 1, 2, NAN, 5, 4])


def test_seq(df):
    same(column(df, "egen s = seq(), from(1) to(3)"), [1, 2, 3, 1, 2, 3, 1, 2])
    same(
        column(df, "egen s = seq(), from(10) block(2)"),
        [10, 10, 11, 11, 12, 12, 13, 13],
    )
    same(column(df, "egen s = seq(), by(g)"), [1, 2, 3, 1, 2, 1, 2, 3])


def test_anycount_anymatch_cut(df):
    same(column(df, "egen k = anycount(a b c), values(1 3)"), [2, 0, 2, 0, 0, 0, 1, 0])
    same(column(df, "egen k = anymatch(a b c), values(5/7)"), [0, 0, 1, 0, 1, 1, 0, 0])
    same(column(df, "egen k = cut(c), at(0, 4, 8)"), [0, NAN, 4, NAN, 4, NAN, 0, 4])
    same(
        column(df, "egen k = cut(c), at(0, 4, 8) icodes"),
        [0, NAN, 1, NAN, 1, NAN, 0, 1],
    )


def test_rowfirst_rowlast(df):
    same(column(df, "egen double f = rowfirst(a b c)"), [1, NAN, 3, 4, 5, 6, 1, 8])
    same(column(df, "egen double f = rowlast(a b c)"), [3, NAN, 5, 4, 7, 9, 1, 4])


def test_moments_and_modes():
    d = pd.DataFrame({"x": [1.0, 2.0, 2.0, 3.0, 10.0]})
    x = d.x.to_numpy()
    dev = x - x.mean()
    m2 = np.mean(dev**2)
    same(column(d, "egen double s = skew(x)"), [np.mean(dev**3) / m2**1.5] * 5)
    same(column(d, "egen double s = kurt(x)"), [np.mean(dev**4) / m2**2] * 5)
    same(column(d, "egen double s = mdev(x)"), [np.mean(np.abs(dev))] * 5)
    same(column(d, "egen double s = mad(x)"), [1.0] * 5)  # |x - 2| -> 0 0 1 1 8
    same(column(d, "egen s = mode(x)"), [2.0] * 5)
    two = pd.DataFrame({"x": [1.0, 1.0, 4.0, 4.0, 2.0]})
    same(column(two, "egen s = mode(x)"), [NAN] * 5)  # two modes: missing
    same(column(two, "egen s = mode(x), minmode"), [1.0] * 5)
    same(column(two, "egen s = mode(x), maxmode"), [4.0] * 5)
