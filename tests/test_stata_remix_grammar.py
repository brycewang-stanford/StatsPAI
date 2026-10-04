"""Do-file grammar the Remix labs use and ``sp.stata`` did not read: macros
that hold a stored result, varlists in ``keep`` / ``drop``, ``collapse
(firstnm)``, ``reshape``. The reference files are Stata 18's output for the
same lines (``_fixtures/reshape_*_Stata.csv``); the macro texts are the ones
Stata printed."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession, _macro_number
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "reference_parity" / "_fixtures"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(_FIX / "did_commands_data.csv")


def _session(data: pd.DataFrame, lines: str) -> pd.DataFrame:
    session = StataSession(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in lines.strip().splitlines():
            session.run(line)
    return session.data


# ----------------------------------------------------------------------
# macros
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, text",
    [
        (0.41178860374999998184, ".41178860375"),
        (1.47311234800236223741, "1.473112348002362"),
        (1 / 3, ".3333333333333333"),
        (123456789.123456789, "123456789.1234568"),
        (-1e-7 / 3, "-3.33333333333e-08"),
        (1e20 / 3, "3.33333333333e+19"),
        (2015.0, "2015"),
        (float("nan"), "."),
    ],
)
def test_macro_text_is_stata_s(value, text):
    """What ``di "`m'"`` printed in Stata 18 after ``local m = <value>``."""
    assert _macro_number(value) == text


def test_stored_result_used_as_a_macro(df):
    out = _session(df, "su xt\ngen double c = xt - `r(mean)'")
    # Stata subtracts the macro's text, 12 digits here, not the double
    np.testing.assert_allclose(out["c"], df["xt"] - 0.41178860375, rtol=0, atol=1e-15)


def test_local_defined_from_a_result_and_from_arithmetic(df):
    out = _session(
        df,
        "su xt\nlocal m = r(mean)\nlocal k = `m' * 2\n"
        "gen double c = xt - `k'\nlocal y0 = 2000 + 5\ngen z = `y0'",
    )
    np.testing.assert_allclose(out["c"], df["xt"] - 2 * 0.41178860375, atol=1e-15)
    assert set(out["z"]) == {2005.0}


@pytest.mark.parametrize(
    "lines",
    [
        "gen q = `r(sd)'",  # nothing stored r(sd)
        "local w : word 1 of a b\ngen q = `w'",  # an extended function
    ],
)
def test_macros_the_session_cannot_know_are_still_refused(df, lines):
    with pytest.raises(MethodIncompatibility, match="macro"):
        sp.stata(lines, df)


def test_macros_that_read_the_data_take_statas_value(df):
    # `local k = exp` is evaluated as Stata evaluates it (2026-10): _N is
    # the number of observations, a variable is its value in observation 1
    assert sp.stata("local k = _N\ndisplay `k'", df) == len(df)
    assert sp.stata("local k = xt\ndisplay `k'", df) == pytest.approx(
        float(np.float64(df["xt"].iloc[0])), rel=1e-12
    )


def test_text_macros_are_unchanged(df):
    res = sp.stata('local v "xt x1"\nreg y `v\'', df)
    assert list(res.params.index) == ["Intercept", "xt", "x1"]


# ----------------------------------------------------------------------
# keep / drop varlists
# ----------------------------------------------------------------------


def test_drop_and_keep_expand_ranges_and_wildcards(df):
    out = _session(
        df, "gen a = 1\ngen b = 2\ngen c = 3\ngen t_1 = 1\ngen t_2 = 2\ndrop a-c t_*"
    )
    assert list(out.columns) == list(df.columns)
    out = _session(df, "keep id x1-y")
    assert list(out.columns) == ["id", "x1", "x2", "y"]
    with pytest.raises(MethodIncompatibility, match="matches no variable"):
        sp.stata("drop zz*\nreg y xt", df)
    with pytest.raises(MethodIncompatibility, match="comes after"):
        sp.stata("drop y-x1\nreg y xt", df)


# ----------------------------------------------------------------------
# collapse
# ----------------------------------------------------------------------


def test_collapse_first_takes_the_row_and_firstnm_skips_missing(df):
    """Stata: ``first`` is the first observation, missing or not;
    ``firstnm`` the first non-missing one. ``first`` used to skip missing."""
    out = _session(
        df,
        "gen double ym = y if year != 2001\n"
        "collapse (first) f=ym (firstnm) fn=ym (last) l=ym (lastnm) ln=ym, by(id)",
    )
    assert out["f"].isna().all()
    second = df[df["year"] == 2002].set_index("id")["y"]
    last = df[df["year"] == 2006].set_index("id")["y"]
    np.testing.assert_allclose(out.set_index("id")["fn"], second)
    np.testing.assert_allclose(out.set_index("id")["l"], last)
    np.testing.assert_allclose(out.set_index("id")["ln"], last)
    # Stata's `su`: fn has mean .2947777, sd 1.487003 over the 400 units
    assert out["fn"].mean() == pytest.approx(0.2947777, abs=5e-8)
    assert out["fn"].std() == pytest.approx(1.487003, abs=5e-7)


# ----------------------------------------------------------------------
# reshape
# ----------------------------------------------------------------------

_RESHAPE_SETUP = (
    "keep if id <= 3 & year <= 2003\n"
    "drop d04\n"
    "replace xt = . if id == 2 & year == 2002\n"
    "drop if id == 3 & year == 2001\n"
)


def _same(mine: pd.DataFrame, path: Path) -> None:
    theirs = pd.read_csv(path)
    assert list(mine.columns) == list(theirs.columns)
    np.testing.assert_allclose(
        mine.to_numpy(dtype=float), theirs.to_numpy(dtype=float), atol=1e-6
    )


def test_reshape_wide_and_back(df):
    wide = _session(df, _RESHAPE_SETUP + "reshape wide y xt, i(id) j(year)")
    _same(wide, _FIX / "reshape_wide_Stata.csv")
    long = _session(
        df,
        _RESHAPE_SETUP
        + "reshape wide y xt, i(id) j(year)\nreshape long y xt, i(id) j(year)",
    )
    _same(long, _FIX / "reshape_long_Stata.csv")
    # the unit-period that was dropped comes back as a row of missing values
    row = long[(long["id"] == 3) & (long["year"] == 2001)]
    assert len(row) == 1 and row[["y", "xt"]].isna().all(axis=None)


@pytest.mark.parametrize(
    "line, words",
    [
        ("reshape wide y, i(id) j(year)", "not constant within"),
        ("reshape wide y xt, i(g) j(year)", "not unique within"),
        ("reshape long y, i(id) j(year)", "already exists"),
        ("reshape wide y xt, i(id) j(year) string", "not implemented"),
        ("reshape error", "are implemented"),
    ],
)
def test_reshape_refusals(df, line, words):
    with pytest.raises(MethodIncompatibility, match=words):
        sp.stata("drop d04\n" + line + "\nreg y xt", df)


# ----------------------------------------------------------------------
# reghdfe factor variables, did2s spacing
# ----------------------------------------------------------------------


def test_reghdfe_factor_terms_are_written_for_hdfe_ols():
    out = sp.from_stata("reghdfe y i.g x, absorb(id) vce(cluster id)")
    assert out["arguments"]["formula"] == "y ~ i.g + x | id"
    out = sp.from_stata("reghdfe y ib1940.g, absorb(id year) noconstant")
    assert out["arguments"]["formula"] == "y ~ ib1940.g | id + year"
    assert out["untranslated_options"] == []
    assert "noconstant" in out["ignored_display_options"]
    assert not sp.from_stata("reghdfe y i.g#c.x, absorb(id)")["ok"]


def test_feols_base_level_is_checked(df):
    with pytest.raises(ValueError, match="not a level"):
        sp.hdfe_ols("y ~ ib1999.g | year", data=df)
    with pytest.raises(ValueError, match="base level"):
        sp.hdfe_ols("y ~ xt | ib2003.g", data=df)
    default = sp.hdfe_ols("y ~ i.g + xt | year", data=df)
    explicit = sp.hdfe_ols("y ~ ib0.g + xt | year", data=df)
    np.testing.assert_allclose(default.params, explicit.params, rtol=1e-12)


def test_did2s_factor_written_with_a_space():
    out = sp.from_stata(
        "did2s y, first_stage(i.year) second_stage(ib0. rel) treatment(d) "
        "cluster(id) unit(id)"
    )
    assert out["ok"] and out["arguments"]["second_stage"] == ["ib0.rel"]


# ----------------------------------------------------------------------
# regression adjustment
# ----------------------------------------------------------------------


def test_g_computation_by_arm_guards(df):
    c = df[df["year"] == 2005]
    kw = dict(y="y", treat="x2", covariates=["x1", "xt"])
    with pytest.raises(MethodIncompatibility, match="by_arm=True"):
        sp.g_computation(c, se_method="analytic", **kw)
    with pytest.raises(MethodIncompatibility, match="by_arm=True"):
        sp.g_computation(c, ps_covariates=["x1"], **kw)
    with pytest.raises(MethodIncompatibility, match="se_method"):
        sp.g_computation(c, by_arm=True, se_method="robust", **kw)
    with pytest.raises(MethodIncompatibility, match="binary treatment"):
        sp.g_computation(c, by_arm=True, estimand="dose_response", **kw)
    boot = sp.g_computation(c, by_arm=True, n_boot=200, seed=1, **kw)
    exact = sp.g_computation(c, by_arm=True, se_method="analytic", **kw)
    assert boot.estimate == exact.estimate
    assert boot.se == pytest.approx(exact.se, rel=0.15)


def test_by_arm_equals_the_two_regressions_written_out(df):
    c = df[df["year"] == 2005]
    X = np.column_stack([np.ones(len(c)), c["x1"], c["xt"]])
    y = c["y"].to_numpy()
    arm = c["x2"].to_numpy() == 1
    b1 = np.linalg.lstsq(X[arm], y[arm], rcond=None)[0]
    b0 = np.linalg.lstsq(X[~arm], y[~arm], rcond=None)[0]
    res = sp.g_computation(
        c, "y", "x2", ["x1", "xt"], by_arm=True, se_method="analytic"
    )
    assert res.estimate == pytest.approx(float(np.mean(X @ (b1 - b0))), rel=1e-12)
    att = sp.g_computation(
        c, "y", "x2", ["x1", "xt"], estimand="ATT", by_arm=True, se_method="analytic"
    )
    assert att.estimate == pytest.approx(float(np.mean(X[arm] @ (b1 - b0))), rel=1e-12)
