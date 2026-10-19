"""``sp.sumstats`` / ``sp.balance_table`` with categorical columns,
``filename=`` and strict ``output=`` (2026-10-19 audit).

A string or categorical column raised a raw ``TypeError`` from
``Series.mean``; now each level gets a row whose mean is its share, as
Stata's ``summarize i.var``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    r = np.random.default_rng(0)
    out = pd.DataFrame(
        {
            "x": r.normal(size=60),
            "s": r.choice(["a", "b", "c"], 60),
            "c": pd.Categorical(r.choice(["u", "v"], 60), categories=["v", "u"]),
            "d": np.repeat([0, 1], 30),
        }
    )
    out.loc[[3, 40], "s"] = None
    return out


def test_sumstats_level_rows_are_shares(df):
    tab = sp.sumstats(df, vars=["x", "s", "c"], output="numeric")
    shares = df["s"].value_counts(normalize=True)
    for level, share in shares.items():
        assert tab.loc[f"s: {level}", "Mean"] == pytest.approx(share, rel=1e-12)
        assert tab.loc[f"s: {level}", "N"] == df["s"].notna().sum()
    # Categorical levels keep their declared order.
    assert [i for i in tab.index if i.startswith("c: ")] == ["c: v", "c: u"]
    assert tab.loc["x", "Mean"] == pytest.approx(df.x.mean(), rel=1e-12)


def test_sumstats_by_group_shares(df):
    tab = sp.sumstats(df, vars=["s"], by="d", output="numeric")
    treated = df[df.d == 1]["s"].value_counts(normalize=True)
    for level, share in treated.items():
        assert tab.loc[f"s: {level}", ("Treated", "Mean")] == pytest.approx(share)


def test_balance_table_level_rows(df):
    tab = sp.balance_table(df, treat="d", covariates=["x", "s"], output="dataframe")
    t = df[df.d == 1]["s"].dropna()
    c = df[df.d == 0]["s"].dropna()
    diff = (t == "b").mean() - (c == "b").mean()
    assert float(tab.loc["s: b", "Diff"]) == pytest.approx(diff, abs=5e-4)


@pytest.mark.parametrize(
    "ext", [".xlsx", ".docx", ".tex", ".html", ".md", ".txt", ".csv"]
)
def test_filename_writes_the_file(df, tmp_path, ext):
    path = tmp_path / f"t{ext}"
    text = sp.sumstats(df, vars=["x", "s"], filename=str(path))
    assert path.exists() and path.stat().st_size > 0
    assert "s: a" in text  # the return is still the text table
    path2 = tmp_path / f"b{ext}"
    sp.balance_table(df, treat="d", covariates=["x"], filename=str(path2))
    assert path2.exists() and path2.stat().st_size > 0


def test_markdown_output_keeps_display_strings(df):
    md = sp.sumstats(df, vars=["x"], output="markdown")
    shown = sp.sumstats(df, vars=["x"], output="dataframe").loc["x", "Mean"]
    assert f"| {shown} |" in md.replace("  ", " ")


@pytest.mark.parametrize("bad", ["bogus", "table.pdf"])
def test_invalid_output_raises(df, bad):
    with pytest.raises(sp.MethodIncompatibility):
        sp.sumstats(df, vars=["x"], output=bad)
    with pytest.raises(sp.MethodIncompatibility):
        sp.balance_table(df, treat="d", covariates=["x"], output=bad)


def test_invalid_filename_and_test_raise(df, tmp_path):
    with pytest.raises(sp.MethodIncompatibility):
        sp.sumstats(df, vars=["x"], filename=str(tmp_path / "t.pdf"))
    with pytest.raises(sp.MethodIncompatibility):
        sp.balance_table(df, treat="d", covariates=["x"], test="chi2")
