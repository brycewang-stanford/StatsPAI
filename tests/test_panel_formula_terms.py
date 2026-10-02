"""``sp.panel`` formulas with factor, transformed and interaction terms.

A textbook two-way fixed-effects regression is written ``xtreg y d i.year,
fe``. ``sp.panel(df, "y ~ d + C(year)", method="fe")`` used to fail with
"Column 'C(year)' not found in data": the formula was split on ``+`` and
each piece looked up as a column.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n_units, n_periods = 60, 8
    out = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), n_periods),
            "year": np.tile(np.arange(2000, 2000 + n_periods), n_units),
        }
    )
    start = np.where(out.id < 20, 2003, np.where(out.id < 40, 2005, 9999))
    out["treat"] = (out.year >= start).astype(int)
    out["x"] = rng.normal(size=len(out))
    out["reg"] = out.id % 3
    out["y"] = (
        0.1 * out.id
        + 0.3 * (out.year - 2000)
        + 2 * out.treat
        + out.x
        + rng.normal(size=len(out))
    )
    return out


def _fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.panel(*args, **kwargs)


def test_year_dummies_are_the_two_way_estimator(df):
    dummies = _fit(df, "y ~ treat + x + C(year)", entity="id", time="year", method="fe")
    twoway = _fit(df, "y ~ treat + x", entity="id", time="year", method="twoway")
    absorbed = sp.feols("y ~ treat + x | id + year", df)
    assert list(dummies.params.index[:2]) == ["treat", "x"]
    assert [n for n in dummies.params.index if n.startswith("year[")] == [
        f"year[{y}]" for y in range(2001, 2008)
    ]
    for name in ("treat", "x"):
        assert dummies.params[name] == pytest.approx(twoway.params[name], rel=1e-10)
        assert dummies.params[name] == pytest.approx(
            float(absorbed.params[name]), rel=1e-10
        )


def test_interactions_and_transforms_equal_hand_built_columns(df):
    by_formula = _fit(
        df,
        "y ~ treat + treat:x + I(x**2) + C(reg):x",
        entity="id",
        time="year",
        method="fe",
    )
    built = df.assign(
        tx=df.treat * df.x,
        x2=df.x**2,
        r0x=(df.reg == 0) * df.x,
        r1x=(df.reg == 1) * df.x,
        r2x=(df.reg == 2) * df.x,
    )
    by_hand = _fit(
        built,
        "y ~ treat + tx + x2 + r0x + r1x + r2x",
        entity="id",
        time="year",
        method="fe",
    )
    pairs = {
        "treat": "treat",
        "treat:x": "tx",
        "I[x ** 2]": "x2",
        "reg[0]:x": "r0x",
        "reg[1]:x": "r1x",
        "reg[2]:x": "r2x",
    }
    assert list(by_formula.params.index) == list(pairs)
    for ours, theirs in pairs.items():
        assert by_formula.params[ours] == pytest.approx(
            by_hand.params[theirs], rel=1e-10
        )
        assert by_formula.std_errors[ours] == pytest.approx(
            by_hand.std_errors[theirs], rel=1e-10
        )


def test_plain_formulas_do_not_copy_or_change(df):
    a = _fit(df, "y ~ treat + x", entity="id", time="year", method="fe")
    assert list(a.params.index) == ["treat", "x"]


def test_twfe_is_an_alias_of_twoway(df):
    a = _fit(df, "y ~ treat + x", entity="id", time="year", method="twfe")
    b = _fit(df, "y ~ treat + x", entity="id", time="year", method="twoway")
    assert a.params["treat"] == b.params["treat"]


def test_a_factor_of_a_missing_column_raises(df):
    with pytest.raises(MethodIncompatibility, match="nope"):
        sp.panel(df, "y ~ treat + C(nope)", entity="id", time="year", method="fe")
    with pytest.raises(MethodIncompatibility, match="nope"):
        sp.panel(df, "y ~ treat + nope", entity="id", time="year", method="fe")


def test_rows_with_a_missing_factor_level_leave_the_sample(df):
    holes = df.copy()
    holes["reg"] = holes["reg"].astype(float)
    holes.loc[holes.index[:16], "reg"] = np.nan
    fit = _fit(holes, "y ~ treat + C(reg):x", entity="id", time="year", method="fe")
    ref = _fit(
        holes.dropna(subset=["reg"]),
        "y ~ treat + C(reg):x",
        entity="id",
        time="year",
        method="fe",
    )
    assert fit.params["treat"] == pytest.approx(ref.params["treat"], rel=1e-10)
