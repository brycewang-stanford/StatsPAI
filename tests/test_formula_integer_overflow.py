"""Integer columns inside generated formula terms are evaluated in float64.

numpy integer arithmetic wraps around silently. ``.dta`` files keep
``byte`` / ``int`` / ``long`` storage (``int8`` / ``int16`` / ``int32``),
and ``I(age**3)`` on those used to be computed in the column's integer
type: widening to ``int64`` (2026-10) fixed small values, but a ``long``
such as a county population (2-3 million) still overflowed ``int64`` once
cubed, and so does a plain ``int64`` column. Every assertion below compares
the integer-typed fit with the same fit on float64 columns: the design
matrices are then identical, so the coefficients agree to rounding
(tolerance 1e-10 relative; observed 0).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import StatsPAIWarning

RTOL = 1e-10


def _frames(dtype, big: bool):
    rng = np.random.default_rng(0)
    n = 500
    if big:
        age = rng.integers(1_000_000, 3_000_000, n)
        scale = 1e-18
    else:
        age = rng.integers(17, 56, n)
        scale = 1e-4
    educ = rng.integers(3, 17, n)
    eta = -1 + scale * age.astype(float) ** 3 * 0.1 + 0.01 * educ
    y = 1 + scale * age.astype(float) ** 3 + 0.05 * educ + rng.normal(0, 1, n)
    yb = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    base = pd.DataFrame({"y": y, "yb": yb, "age": age, "educ": educ})
    ints = base.astype({"age": dtype, "educ": dtype})
    floats = base.astype({"age": float, "educ": float})
    return ints, floats


CASES = [(np.int8, False), (np.int16, False), (np.int32, True), (np.int64, True)]
IDS = ["int8", "int16", "int32-large", "int64-large"]


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


@pytest.mark.parametrize("dtype, big", CASES, ids=IDS)
@pytest.mark.parametrize(
    "fit",
    [
        lambda d: sp.regress("y ~ age + I(age**3) + age:educ", data=d).params,
        lambda d: sp.regress("y ~ age + I(age^3)", data=d).params,
        lambda d: sp.glm("y ~ age + I(age**3)", data=d).params,
        lambda d: sp.logit("yb ~ I(age**3) + educ", data=d).params,
    ],
    ids=["regress", "regress-caret", "glm", "logit"],
)
def test_fit_equals_float64(dtype, big, fit):
    ints, floats = _frames(dtype, big)
    a = np.asarray(_quiet(lambda: fit(ints)), float)
    b = np.asarray(_quiet(lambda: fit(floats)), float)
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=0)


@pytest.mark.parametrize("dtype, big", CASES, ids=IDS)
def test_predict_newdata_equals_float64(dtype, big):
    ints, floats = _frames(dtype, big)
    for fit in (
        lambda: sp.regress("y ~ I(age**3) + educ", data=floats),
        lambda: sp.glm("y ~ I(age**3) + educ", data=floats),
    ):
        model = _quiet(fit)
        a = np.asarray(_quiet(lambda: model.predict(newdata=ints)), float)
        b = np.asarray(_quiet(lambda: model.predict(newdata=floats)), float)
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=0)


def _did_frames(dtype):
    rng = np.random.default_rng(1)
    df = _quiet(sp.datasets.mpdta)
    units = df["countyreal"].unique()
    pop = pd.Series(rng.integers(1_000_000, 3_000_000, len(units)), index=units)
    df["pop"] = df["countyreal"].map(pop)
    df["lemp"] = df["lemp"] + 1e-19 * df["pop"].astype(float) ** 3
    return df.astype({"pop": dtype}), df.astype({"pop": float})


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_did_covariate_formulas_equal_float64(dtype):
    ints, floats = _did_frames(dtype)

    def cs(d):
        return sp.callaway_santanna(
            d,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            x="~ I(pop**3)",
        ).estimate

    def drdid(d):
        d = d[d["year"].isin([2003, 2007]) & d["first_treat"].isin([0, 2007])].copy()
        d["tr"] = (d["first_treat"] == 2007).astype(int)
        d["post"] = (d["year"] == 2007).astype(int)
        return sp.drdid(
            d,
            y="lemp",
            group="tr",
            time="post",
            id="countyreal",
            covariates="~ I(pop**3)",
        ).estimate

    for fn in (cs, drdid):
        a = _quiet(lambda: fn(ints))
        b = _quiet(lambda: fn(floats))
        assert a == pytest.approx(b, rel=RTOL, abs=0)


def test_factor_levels_keep_integer_names():
    ints, _ = _frames(np.int64, big=False)
    ints["g"] = np.arange(len(ints)) % 3 + 1
    params = _quiet(lambda: sp.regress("y ~ C(g) + I(age**2)", data=ints).params)
    assert "C(g)[T.2]" in params.index and "C(g)[T.3]" in params.index


def test_column_in_factor_and_generated_term_warns_when_large():
    from statspai.core._formula_dtypes import float_generated_term_columns

    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "g": [2_500_000, 2_600_000, 2_700_000]})
    with pytest.warns(StatsPAIWarning, match="both as a factor"):
        out = float_generated_term_columns(df, "y ~ C(g) + I(g**3)")
    assert out["g"].dtype == np.int64  # C(g) keeps its integer level names
    # a column read only by the generated term is cast
    cast = float_generated_term_columns(df, "y ~ I(g**3)")
    assert cast["g"].dtype == np.float64


def test_wrapped_narrow_integer_column_warns():
    ints, _ = _frames(np.int8, big=False)
    with np.errstate(over="ignore"):
        ints["age3"] = ints["age"] ** 3  # int8 arithmetic: wraps around
    assert ints["age3"].min() < -100 and ints["age3"].max() > 100
    with pytest.warns(StatsPAIWarning, match="wrapped around"):
        sp.regress("y ~ age + age3", data=ints)


def test_unwrapped_narrow_integer_column_is_silent():
    ints, _ = _frames(np.int8, big=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StatsPAIWarning)
        sp.regress("y ~ age + educ", data=ints)


# ---- sp.feols / sp.fepois (fixest grammar) ----------------------------------


def _fixest_panel():
    rng = np.random.default_rng(0)
    n = 400
    pop = rng.integers(2_100_000, 2_600_000, n)  # pop**3 > 2**63
    df = pd.DataFrame(
        {
            "pop": pop.astype("int64"),
            "g": rng.integers(0, 10, n),
            "rel": rng.integers(-2, 3, n),
        }
    )
    df["y"] = (df["pop"].astype(float) / 1e6) ** 3 + rng.normal(size=n)
    return df


@pytest.mark.parametrize(
    "fml",
    [
        "y ~ I(pop**3)",
        "y ~ I(pop**3 / 1e18) | g",
        "y ~ I(pop**3 / 1e18) + i(rel, ref=0) | g",
    ],
)
@pytest.mark.parametrize("fit", ["feols", "fepois"])
def test_fixest_generated_term_equals_float64(fml, fit):
    df = _fixest_panel()
    fn = getattr(sp, fit)
    got = fn(fml, data=df).params
    want = fn(fml, data=df.assign(pop=df["pop"].astype(float))).params
    # i(rel) keeps its integer level names (rel::-2, not rel::-2.0)
    assert list(got.index) == list(want.index)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=1e-10, atol=0)


def test_fixest_stepwise_arguments_are_read():
    df = _fixest_panel()
    got = sp.feols("y ~ csw(I(pop**3 / 1e18), rel) | g", data=df)
    want = sp.feols(
        "y ~ csw(I(pop**3 / 1e18), rel) | g",
        data=df.assign(pop=df["pop"].astype(float)),
    )
    for a, b in zip(got, want):
        np.testing.assert_allclose(
            a.params.to_numpy(), b.params.to_numpy(), rtol=1e-10, atol=0
        )
