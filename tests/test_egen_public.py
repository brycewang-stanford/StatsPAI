"""``sp.egen`` against Stata 18 ``egen``.

``reference_parity/_fixtures/_generate_egen_Stata.do`` runs 42 ``egen``
commands (stored as double) on ``egen_data.csv`` (60 rows, missing values
planted in the variable, the by-variables and the row variables, ties in
``x``); ``egen_Stata.csv`` holds the results. ``sp.egen`` reproduces every
column to 1e-12 relative (the values are sums and order statistics of
two-decimal numbers; the gap is floating-point summation order), with
missing exactly where Stata has missing. ``tag`` is compared by group: it
marks one row of each group on both sides, and which row is Stata's
arbitrary choice within the group's sort order.

The same commands through ``sp.stata`` give the same numbers (one
implementation; ``egen double`` keeps double precision there too).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ColumnNotFound, MethodIncompatibility

FX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"

#: Stata column -> (sp.egen arguments, the Stata command's right-hand side)
CASES = {
    "mean_g": (("mean", "x"), dict(by="g"), "mean(x), by(g)"),
    "count_g": (("count", "x"), dict(by="g"), "count(x), by(g)"),
    "sd_g": (("sd", "x"), dict(by="g"), "sd(x), by(g)"),
    "median_g": (("median", "x"), dict(by="g"), "median(x), by(g)"),
    "p25_g": (("pctile", "x"), dict(by="g", p=25), "pctile(x), p(25) by(g)"),
    "iqr_g": (("iqr", "x"), dict(by="g"), "iqr(x), by(g)"),
    "skew_g": (("skew", "x"), dict(by="g"), "skew(x), by(g)"),
    "kurt_g": (("kurt", "x"), dict(by="g"), "kurt(x), by(g)"),
    "mad_g": (("mad", "x"), dict(by="g"), "mad(x), by(g)"),
    "mdev_g": (("mdev", "x"), dict(by="g"), "mdev(x), by(g)"),
    "min_gh": (("min", "x"), dict(by=["g", "h"]), "min(x), by(g h)"),
    "max_gh": (("max", "x"), dict(by=["g", "h"]), "max(x), by(g h)"),
    "total_gh": (("total", "x"), dict(by=["g", "h"]), "total(x), by(g h)"),
    "totalm_gh": (
        ("total", "x"),
        dict(by=["g", "h"], missing=True),
        "total(x), by(g h) missing",
    ),
    "mode_a": (("mode", "a"), dict(by="g"), "mode(a), by(g)"),
    "minmode_a": (
        ("mode", "a"),
        dict(by="g", mode_ties="min"),
        "mode(a), by(g) minmode",
    ),
    "maxmode_a": (
        ("mode", "a"),
        dict(by="g", mode_ties="max"),
        "mode(a), by(g) maxmode",
    ),
    "rank_g": (("rank", "x"), dict(by="g"), "rank(x), by(g)"),
    "field_g": (("rank", "x"), dict(by="g", rank_ties="field"), "rank(x), by(g) field"),
    "track_g": (("rank", "x"), dict(by="g", rank_ties="track"), "rank(x), by(g) track"),
    "std_x": (("std", "x"), {}, "std(x)"),
    "seq_x": (
        ("seq",),
        dict(start=2, stop=4, block=2),
        "seq(), from(2) to(4) block(2)",
    ),
    "group_gh": (("group", ["g", "h"]), {}, "group(g h)"),
    "groupm_gh": (("group", ["g", "h"]), dict(missing=True), "group(g h), missing"),
    "rowtotal_abc": (("rowtotal", ["a", "b", "c"]), {}, "rowtotal(a b c)"),
    "rowtotalm_abc": (
        ("rowtotal", ["a", "b", "c"]),
        dict(missing=True),
        "rowtotal(a b c), missing",
    ),
    "rowmean_abc": (("rowmean", ["a", "b", "c"]), {}, "rowmean(a b c)"),
    "rowmin_abc": (("rowmin", ["a", "b", "c"]), {}, "rowmin(a b c)"),
    "rowmax_abc": (("rowmax", ["a", "b", "c"]), {}, "rowmax(a b c)"),
    "rowsd_abc": (("rowsd", ["a", "b", "c"]), {}, "rowsd(a b c)"),
    "rownonmiss_abc": (("rownonmiss", ["a", "b", "c"]), {}, "rownonmiss(a b c)"),
    "rowmiss_abc": (("rowmiss", ["a", "b", "c"]), {}, "rowmiss(a b c)"),
    "rowfirst_abc": (("rowfirst", ["a", "b", "c"]), {}, "rowfirst(a b c)"),
    "rowlast_abc": (("rowlast", ["a", "b", "c"]), {}, "rowlast(a b c)"),
    "anycount_ac": (
        ("anycount", ["a", "c"]),
        dict(values=[0, 3, 4]),
        "anycount(a c), values(0 3/4)",
    ),
    "anymatch_ac": (
        ("anymatch", ["a", "c"]),
        dict(values=[5]),
        "anymatch(a c), values(5)",
    ),
    "cut_at": (("cut", "x"), dict(at=[0, 6, 9, 12, 30]), "cut(x), at(0, 6, 9, 12, 30)"),
    "cut_icodes": (
        ("cut", "x"),
        dict(at=[0, 6, 9, 12, 30], icodes=True),
        "cut(x), at(0, 6, 9, 12, 30) icodes",
    ),
    "cut_group": (("cut", "x"), dict(n_groups=4), "cut(x), group(4)"),
}
#: with Stata's `if w == 1`
WHERE = {
    "mean_if": (("mean", "x"), dict(by="g"), "mean(x) if w == 1, by(g)"),
    "std_if": (("std", "x"), {}, "std(x) if w == 1"),
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FX / "egen_data.csv")


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(FX / "egen_Stata.csv").set_index("id")


def _same(got, want):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("col", sorted(CASES))
def test_sp_egen_matches_stata(col, data, ref):
    args, kw, _ = CASES[col]
    out = sp.egen(data, *args, **kw)
    assert out.index.equals(data.index)
    _same(out.to_numpy(), ref.loc[data["id"], col].to_numpy())


@pytest.mark.parametrize("col", sorted(WHERE))
def test_where_is_statas_if(col, data, ref):
    args, kw, _ = WHERE[col]
    out = sp.egen(data, *args, where=data["w"] == 1, **kw)
    _same(out.to_numpy(), ref.loc[data["id"], col].to_numpy())


def test_tag_marks_one_row_per_group_as_stata(data, ref):
    out = sp.egen(data, "tag", ["g", "h"])
    stata = ref.loc[data["id"], "tag_gh"].to_numpy()
    keys = data[["g", "h"]].fillna(-1)
    ours = out.groupby([keys.g, keys.h]).sum()
    theirs = pd.Series(stata, index=out.index).groupby([keys.g, keys.h]).sum()
    pd.testing.assert_series_equal(ours, theirs, check_dtype=False, check_names=False)
    # Stata: a row with a missing h is in no group (tag 0)
    assert out[data.h.isna()].sum() == 0


@pytest.mark.parametrize("col", sorted(CASES) + sorted(WHERE))
def test_sp_stata_egen_is_the_same_computation(col, data):
    args, kw, rhs = {**CASES, **WHERE}[col]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = sp.stata(f"egen double v = {rhs}", data=data, return_data=True)
    if col in WHERE:
        kw = {**kw, "where": data["w"] == 1}
    _same(run.data["v"].to_numpy(), sp.egen(data, *args, **kw).to_numpy())


def test_values_aligned_with_the_rows(data):
    """x may be values aligned with the rows (an expression computed in
    pandas); the result keeps the caller's index."""
    df = data.set_index(data["id"] * 10)
    a = sp.egen(df, "mean", df["x"] * 2, by="g")
    b = sp.egen(df, "mean", "x", by="g") * 2
    _same(a.to_numpy(), b.to_numpy())
    assert a.index.equals(df.index)


@pytest.mark.parametrize(
    "call, exc, match",
    [
        (lambda d: sp.egen(d, "bogus", "x"), MethodIncompatibility, "not implemented"),
        (
            lambda d: sp.egen(d, "rowmean", ["a", "b"], by="g"),
            MethodIncompatibility,
            "by",
        ),
        (lambda d: sp.egen(d, "mean", "x", p=10), MethodIncompatibility, "option"),
        (lambda d: sp.egen(d, "mean", "nope"), ColumnNotFound, "nope"),
        (lambda d: sp.egen(d, "cut", "x"), MethodIncompatibility, "at"),
        (
            lambda d: sp.egen(d, "mean", d["x"].iloc[:5].to_numpy()),
            MethodIncompatibility,
            "values",
        ),
        (
            lambda d: sp.egen(d.assign(s="t"), "mean", "s"),
            MethodIncompatibility,
            "string",
        ),
        (lambda d: sp.egen(d, "seq", stop=0), MethodIncompatibility, "to()"),
    ],
)
def test_refusals(call, exc, match, data):
    with pytest.raises(exc, match=match):
        call(data)


def test_empty_group_conventions():
    """[D] egen: count is 0 and total 0 (missing with `missing`) for a group
    with no value; mean is missing."""
    d = pd.DataFrame({"g": [1, 1, 2], "x": [np.nan, np.nan, 3.0]})
    assert sp.egen(d, "count", "x", by="g").tolist() == [0.0, 0.0, 1.0]
    assert sp.egen(d, "total", "x", by="g").tolist() == [0.0, 0.0, 3.0]
    _same(sp.egen(d, "total", "x", by="g", missing=True), [np.nan, np.nan, 3.0])
    _same(sp.egen(d, "mean", "x", by="g"), [np.nan, np.nan, 3.0])
