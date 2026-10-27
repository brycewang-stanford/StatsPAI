"""``sp.eivreg`` against Stata 18 ``eivreg``.

Reference: ``_fixtures/eivreg_stata.txt`` written by
``_fixtures/_generate_eivreg_Stata.do`` (StataNow 18 MP, 2026-10-28) on
``_fixtures/eivreg_data.csv`` (``_generate_eivreg_data.py``). Cases: two
error-prone regressors, one, analytic weights, frequency weights, and
analytic weights with zero-weight rows. The estimator and its sandwich
variance are closed form on both sides, so the tolerance is the CLAUDE.md
5.1 default of 1e-6; observed 5e-15 at worst.
"""

from __future__ import annotations

import pathlib
import warnings
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import IdentificationFailure

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6
FORMULA = "y ~ x1 + x2 + z"
TERMS = {"x1": "x1", "x2": "x2", "z": "z", "_cons": "Intercept"}
CASES = {
    "plain": {"reliability": {"x1": 0.8, "x2": 0.75}},
    "one": {"reliability": {"x1": 0.9}},
    "aw": {"reliability": {"x1": 0.8, "x2": 0.75}, "weights": "w"},
    "fw": {
        "reliability": {"x1": 0.8, "x2": 0.75},
        "weights": "fw",
        "weights_as": "frequency",
    },
}


@lru_cache(maxsize=1)
def _stata() -> Dict[str, float]:
    out: Dict[str, float] = {}
    text = (_FIX / "eivreg_stata.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            key, value = line.split()
            out[key] = float(value)
    return out


def _data() -> pd.DataFrame:
    df = pd.read_csv(_FIX / "eivreg_data.csv")
    df["fw"] = np.ceil(df["w"])
    return df


@pytest.mark.parametrize("case", sorted(CASES))
def test_coefficients_and_standard_errors_match_stata(case):
    ref = _stata()
    fit = sp.eivreg(FORMULA, _data(), **CASES[case])
    for stata_name, name in TERMS.items():
        assert fit.params[name] == pytest.approx(
            ref[f"{case}_b_{stata_name}"], rel=RTOL
        )
        assert fit.std_errors[name] == pytest.approx(
            ref[f"{case}_se_{stata_name}"], rel=RTOL
        )
    cov = fit.data_info["var_cov"]
    assert cov[1, 2] == pytest.approx(ref[f"{case}_cov_x1_x2"], rel=RTOL)


@pytest.mark.parametrize("case", sorted(CASES))
def test_model_statistics_match_stata(case):
    ref = _stata()
    fit = sp.eivreg(FORMULA, _data(), **CASES[case])
    assert fit.nobs == int(ref[f"{case}_N"])
    assert fit.data_info["df_resid"] == ref[f"{case}_df_r"]
    assert fit.diagnostics["F-statistic"] == pytest.approx(ref[f"{case}_F"], rel=RTOL)
    assert fit.diagnostics["R-squared"] == pytest.approx(ref[f"{case}_r2"], rel=RTOL)
    assert fit.diagnostics["Root MSE"] == pytest.approx(ref[f"{case}_rmse"], rel=RTOL)


def test_zero_analytic_weights_leave_the_sample_as_in_stata():
    ref = _stata()
    df = _data()
    df.loc[:2, "w"] = 0.0
    fit = sp.eivreg(FORMULA, df, reliability={"x1": 0.8, "x2": 0.75}, weights="w")
    assert fit.nobs == int(ref["aw0_N"]) == 77
    assert fit.params["x1"] == pytest.approx(ref["aw0_b_x1"], rel=RTOL)
    assert fit.std_errors["x1"] == pytest.approx(ref["aw0_se_x1"], rel=RTOL)


def test_without_reliability_the_point_estimate_is_ols():
    ref = _stata()
    df = _data()
    fit = sp.eivreg(FORMULA, df)
    assert fit.params["x1"] == pytest.approx(ref["none_b_x1"], rel=RTOL)
    ols = sp.regress(FORMULA, data=df)
    np.testing.assert_allclose(fit.params.values, ols.params.values, rtol=1e-12)


def test_reliability_floor_is_where_stata_refuses():
    """Stata fits at 0.4360 and stops with r(399) at 0.4359: the floor is the
    R-squared of x1 on x2, z and y (0.435934 on these data)."""
    ref = _stata()
    assert ref["rc_x1_0.4360"] == 0 and ref["rc_x1_0.4359"] == 399
    df = _data()
    sp.eivreg(FORMULA, df, reliability={"x1": 0.4360})
    for r in (0.4359, 0.10):
        with pytest.raises(IdentificationFailure):
            sp.eivreg(FORMULA, df, reliability={"x1": r})
