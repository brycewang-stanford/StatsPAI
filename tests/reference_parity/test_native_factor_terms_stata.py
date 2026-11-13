"""Factor terms on the native HDFE engine, against Stata 18.

Agent probe 2026-11-13, item #33: ``sp.from_stata("xtreg y x i.year, fe")``
gives ``sp.feols('y ~ x + C(year) | id')``. The native engine learnt
``C(factor)`` the same day; a named base level (``ib2005.year`` ->
``C(year, Treatment(2005))``) and a factor interacted with a continuous
term (``i.year#c.x``, ``c.x##i.year`` -> ``C(year):x``, ``x:C(year)``)
still needed the optional pyfixest, so those do-file lines raised
``MissingDependencyError``.

Reference: Stata 18 MP ``xtreg, fe`` and ``reghdfe`` 6.x on the bundled
castle-doctrine panel, ``_fixtures/_generate_factor_terms_Stata.do`` ->
``_fixtures/factor_terms_Stata.csv``. Every coefficient and standard
error is held to 1e-8 relative (absolute 1e-10 for the coefficients that
are zero to rounding); the fits are closed-form OLS after the within
transformation, and agree to 1e-9 or better on macOS.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.fixest import _native

FIX = Path(__file__).resolve().parent / "_fixtures"
DATA = Path(__file__).resolve().parents[2] / "src" / "statspai" / "datasets" / "data"


def _reference() -> dict:
    frame = pd.read_csv(
        FIX / "factor_terms_Stata.csv", skipinitialspace=True, dtype={"name": str}
    )
    return {
        (r.tag, r.kind, r.name): float(r.value) for r in frame.itertuples(index=False)
    }


STATA = _reference()


@pytest.fixture(scope="module")
def castle() -> pd.DataFrame:
    return pd.read_csv(DATA / "castle_2013.csv")


_LEVEL = re.compile(
    r"C\(year(?:, Treatment\(\d+\))?\)\[(?:T\.)?(?P<lev>\d+)\](?::(?P<var>\w+))?\Z"
)
_VAR_FIRST = re.compile(r"(?P<var>\w+):C\(year\)\[T\.(?P<lev>\d+)\]\Z")
_HDFE = re.compile(r"year::(?P<lev>\d+)\Z")


def _stata_name(term: str) -> str:
    """StatsPAI's coefficient label in Stata's ``e(b)`` spelling."""
    for pattern in (_LEVEL, _VAR_FIRST, _HDFE):
        m = pattern.match(term)
        if m:
            var = m.groupdict().get("var")
            return f"{m.group('lev')}.year" + (f"#c.{var}" if var else "")
    return term


CASES = {
    "xtreg_iyear_iid": "xtreg l_homicide post unemployrt i.year, fe",
    "xtreg_iyear_cluster": (
        "xtreg l_homicide post unemployrt i.year, fe vce(cluster sid)"
    ),
    "xtreg_ib2005_cluster": (
        "xtreg l_homicide post unemployrt ib2005.year, fe vce(cluster sid)"
    ),
    "xtreg_year_by_c_cluster": (
        "xtreg l_homicide post i.year#c.unemployrt, fe vce(cluster sid)"
    ),
    "xtreg_c_hash_year_cluster": (
        "xtreg l_homicide post c.unemployrt##i.year, fe vce(cluster sid)"
    ),
    "reghdfe_iyear_cluster": (
        "reghdfe l_homicide post unemployrt i.year, absorb(sid) vce(cluster sid)"
    ),
    "reghdfe_iyear_iid": "reghdfe l_homicide post unemployrt i.year, absorb(sid)",
}


def _run(command: str, data: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(f"xtset sid year\n{command}", data=data)


@pytest.mark.parametrize("tag", sorted(CASES))
def test_translated_command_matches_stata(castle, tag):
    fit = _run(CASES[tag], castle)
    if tag.startswith("xtreg"):
        # the translator's sp.feols call is answered without pyfixest
        assert fit.model_info["backend"] == "statspai-native"
    # Stata's e(b) also lists the base level (0, flagged "b") and _cons
    estimated = {
        name
        for (t, kind, name) in STATA
        if t == tag
        and kind == "se"
        and name not in ("N", "_cons")
        and STATA[(t, "se", name)] > 0
    }
    ours = {}
    for term in fit.params.index:
        name = _stata_name(str(term))
        if name not in estimated:
            # the first level of i.year#c.x is reported as "2000b.year#c.x"
            name = name.replace(".year", "b.year", 1)
        ours[name] = str(term)
    assert set(ours) == estimated
    for name, term in ours.items():
        assert float(fit.params[term]) == pytest.approx(
            STATA[(tag, "b", name)], rel=1e-8, abs=1e-10
        ), name
        assert float(fit.std_errors[term]) == pytest.approx(
            STATA[(tag, "se", name)], rel=1e-8
        ), name
    n_obs = getattr(fit, "n_obs", None)
    if not isinstance(n_obs, (int, np.integer)):
        n_obs = fit.data_info["nobs"]
    assert n_obs == STATA[(tag, "s", "N")]


@pytest.mark.parametrize(
    "command, python_code",
    [
        (
            "xtreg y x ib2003.year, fe",
            "sp.feols('y ~ x + C(year, Treatment(2003)) | <panel_id>', data=df)",
        ),
        (
            "xtreg y x i.year#c.w, fe",
            "sp.feols('y ~ x + C(year):w | <panel_id>', data=df)",
        ),
        (
            "xtreg y c.x##i.year, fe",
            "sp.feols('y ~ x + C(year) + x:C(year) | <panel_id>', data=df)",
        ),
    ],
)
def test_the_translations_this_file_pins(command, python_code):
    assert sp.from_stata(command)["python_code"] == python_code


@pytest.mark.parametrize(
    "rhs",
    [
        "post + unemployrt + C(year)",
        "post + unemployrt + C(year, Treatment(2005))",
        "post + C(year):unemployrt",
        "post + unemployrt + C(year):unemployrt",
        "post + unemployrt + C(year) + unemployrt:C(year)",
        # written out of order: main effects still come first
        "unemployrt:C(year) + C(year) + post + unemployrt",
    ],
)
def test_design_is_the_one_formulaic_builds(castle, rhs):
    """Same columns, names and order as formulaic (what the pyfixest
    delegate hands to its solver), and the least-squares-dummy-variable
    coefficients of that design."""
    formulaic = pytest.importorskip("formulaic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols(f"l_homicide ~ {rhs} | sid", data=castle)
    assert fit.model_info["backend"] == "statspai-native"
    design = formulaic.model_matrix(rhs, castle)
    names = [c for c in design.columns if c != "Intercept"]
    assert list(fit.params.index) == names
    X = np.column_stack(
        [design[names].to_numpy(float), pd.get_dummies(castle["sid"]).to_numpy(float)]
    )
    beta = np.linalg.lstsq(X, castle["l_homicide"].to_numpy(float), rcond=None)[0]
    np.testing.assert_allclose(
        fit.params.to_numpy(), beta[: len(names)], rtol=1e-8, atol=1e-10
    )


def _plan(fml: str, data: pd.DataFrame):
    return _native.plan(
        fml,
        data,
        vcov=None,
        cluster=None,
        weights=None,
        ssc=None,
        fixef_rm="none",
        extra={},
    )


def test_terms_the_engine_does_not_implement_are_refused_not_misread(castle):
    """Anything outside the implemented set goes to the delegate with a
    reason; it is never fitted as something else."""
    df = castle.assign(region=np.where(castle["south"] == 1, "s", "n"))
    native = _plan("l_homicide ~ post + C(year):unemployrt | sid", df)
    assert isinstance(native, _native.Plan)
    refused = {
        # factor by factor
        "l_homicide ~ post + C(year):C(region) | sid": "syntax",
        # a base level that is not in the data
        "l_homicide ~ post + C(year, Treatment(1999)) | sid": "reference level",
        # the same factor term twice
        "l_homicide ~ post + C(year) + C(year) | sid": "syntax",
        # a factor on a float column
        "l_homicide ~ post + C(unemployrt) | sid": "dtype",
    }
    for fml, reason in refused.items():
        out = _plan(fml, df)
        assert isinstance(out, str) and reason in out, (fml, out)
