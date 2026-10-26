"""Option scan (Oct 2026) of ``statspai.survey``.

Each test states the correct behaviour for an option combination that
returned a wrong number, crashed or went unvalidated before the scan.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _frame(seed: int = 0, n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    h = np.repeat(np.arange(4), n // 4)
    psu = h * 10 + rng.integers(0, 5, n)
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "h": h,
            "psu": psu,
            "w": rng.uniform(1, 3, n),
            "x": x,
            "y": 1.0 + 0.5 * x + rng.normal(size=n),
            "dom": (rng.uniform(size=n) < 0.4).astype(float),
        }
    )
    df["b"] = (df["y"] > 1.0).astype(float)
    return df


def _design(df: pd.DataFrame) -> "sp.SurveyDesign":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.svydesign(df, weights="w", strata="h", cluster="psu", nest=True)


# S1 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,formula", [("gaussian", "y ~ x"), ("binomial", "b ~ x")]
)
def test_svyglm_with_repeated_index_labels_equals_reset_index(family, formula):
    df = _frame()
    dup = df.copy()
    dup.index = np.repeat(np.arange(len(df) // 2), 2)
    a = sp.svyglm(formula, _design(df), family=family)
    b = sp.svyglm(formula, _design(dup), family=family)
    np.testing.assert_array_equal(a.estimate.to_numpy(), b.estimate.to_numpy())
    np.testing.assert_array_equal(a.std_error.to_numpy(), b.std_error.to_numpy())


# S2 / S3 ----------------------------------------------------------------


@pytest.mark.parametrize("fn", ["svymean", "svytotal"])
def test_subpop_array_with_missing_membership_is_out_of_domain(fn):
    df = _frame()
    des = _design(df)
    mask = df["dom"].to_numpy().copy()
    mask[:10] = np.nan
    clean = np.nan_to_num(mask, nan=0.0).astype(bool)
    f = getattr(sp, fn)
    a = f("y", des, subpop=mask)
    b = f("y", des, subpop=clean)
    np.testing.assert_array_equal(a.estimate.to_numpy(), b.estimate.to_numpy())
    np.testing.assert_array_equal(a.std_error.to_numpy(), b.std_error.to_numpy())


def test_subpop_string_column_is_refused():
    df = _frame()
    df["dom_s"] = np.where(df["dom"] == 1, "yes", "no")
    with pytest.raises(MethodIncompatibility):
        sp.svymean("y", _design(df), subpop="dom_s")


# S4 ---------------------------------------------------------------------


@pytest.mark.parametrize("nest", [True, False])
def test_missing_psu_id_is_refused(nest):
    df = _frame()
    df["psu"] = df["psu"].astype(float)
    df.loc[3, "psu"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MethodIncompatibility):
            sp.svydesign(df, weights="w", strata="h", cluster="psu", nest=nest)


# S5 ---------------------------------------------------------------------


@pytest.mark.parametrize("fn", ["svymean", "svytotal"])
def test_missing_outcome_is_refused_not_nan(fn):
    df = _frame()
    df.loc[5, "y"] = np.nan
    with pytest.raises(MethodIncompatibility):
        getattr(sp, fn)("y", _design(df))


# S6 ---------------------------------------------------------------------


def test_svyglm_binomial_rejects_outcome_outside_unit_interval():
    df = _frame()
    df["b2"] = df["b"] + 1.0
    with pytest.raises(MethodIncompatibility):
        sp.svyglm("b2 ~ x", _design(df), family="binomial")


def test_svyglm_poisson_rejects_negative_outcome():
    df = _frame()
    with pytest.raises(MethodIncompatibility):
        sp.svyglm("y ~ x", _design(df), family="poisson")


# S7 ---------------------------------------------------------------------


def test_linear_calibration_refuses_missing_auxiliary():
    df = _frame()
    df.loc[0, "x"] = np.nan
    with pytest.raises(MethodIncompatibility):
        sp.linear_calibration(df, totals={"x": 10.0}, weight="w")


def test_subpop_nullable_boolean_with_missing_is_out_of_domain():
    df = _frame()
    des = _design(df)
    nullable = pd.array(df["dom"].astype(bool).to_numpy(), dtype="boolean")
    nullable[:7] = pd.NA
    clean = df["dom"].to_numpy().astype(bool)
    clean[:7] = False
    a = sp.svymean("y", des, subpop=pd.Series(nullable, index=df.index + 5))
    b = sp.svymean("y", des, subpop=clean)
    np.testing.assert_array_equal(a.estimate.to_numpy(), b.estimate.to_numpy())
    np.testing.assert_array_equal(a.std_error.to_numpy(), b.std_error.to_numpy())
