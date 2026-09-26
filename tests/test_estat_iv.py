"""``sp.estat`` IV tests read what ``sp.ivreg`` computed.

``estat endogenous`` / ``estat overid`` looked for ``model_info`` keys
(``wu_hausman``, ``sargan_stat``) that no IV estimator writes -- sp.ivreg stores
``"Hausman F-stat"`` / ``"Sargan statistic"`` (``"Hansen J statistic"`` under a
robust vcov) in ``diagnostics`` -- so both always answered "not found", and
``estat(result, "all")`` matched ``model_type`` exactly against ``"iv"`` while
sp.ivreg labels its fits ``"IV-2SLS"``, so no IV test ever ran.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    n = 500
    z1, z2, u = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    d = 0.6 * z1 + 0.4 * z2 + 0.5 * u + rng.normal(size=n)
    return pd.DataFrame({"y": 1 + 0.5 * d + u, "d": d, "z1": z1, "z2": z2})


def _fit(df, formula, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ivreg(formula, data=df, **kw)


def test_endogeneity_test_reports_the_fit_statistic(df):
    r = _fit(df, "y ~ (d ~ z1 + z2)")
    out = sp.estat(r, "endogenous", print_results=False)
    assert "error" not in out
    assert out["statistic"] == pytest.approx(float(r.diagnostics["Hausman F-stat"]))
    assert out["pvalue"] < 0.05  # d is endogenous by construction


@pytest.mark.parametrize("kw, key", [({}, "Sargan"), ({"robust": "hc1"}, "Hansen J")])
def test_overid_test_reports_sargan_or_hansen(df, kw, key):
    r = _fit(df, "y ~ (d ~ z1 + z2)", **kw)
    out = sp.estat(r, "overid", print_results=False)
    assert "error" not in out
    assert out["statistic"] == pytest.approx(float(r.diagnostics[f"{key} statistic"]))
    assert out["df"] == 1


def test_all_runs_the_iv_tests(df):
    r = _fit(df, "y ~ (d ~ z1 + z2)")
    tests = [o["test"] for o in sp.estat(r, "all", print_results=False)]
    assert any("endogeneity" in t for t in tests)
    assert any("over-identification" in t for t in tests)
    assert any("First-stage" in t for t in tests)


def test_exactly_identified_model_has_no_overid_statistic(df):
    r = _fit(df, "y ~ (d ~ z1)")
    assert "error" in sp.estat(r, "overid", print_results=False)
