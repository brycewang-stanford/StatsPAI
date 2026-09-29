"""Weak-IV inference with absorbed, nested fixed effects vs Stata ``ivreghdfe``.

Reference: Stata 18, ``ivreghdfe y (d = z) w1, absorb(year unit rXy)
cluster(unit) ffirst`` on ``_fixtures/weakiv_nested_fe.csv`` (90 units, 12
years; ``year`` nested in ``rXy``, ``unit`` nested in the clusters), from
``_fixtures/_generate_weakiv_nested_fe_Stata.do``. Two errors found on the Web
of Power replication (QJE 2023, Table 4 col. 4):

* the cluster-robust Anderson-Rubin statistic was the *score* form -- the
  cluster meat built from ``y - b0 d`` rather than the reduced-form residual
  -- which disagrees with the reduced-form test whenever the instrument
  matters (p = 0.068 against ``ivreghdfe``'s 0.0246);
* the absorbed-FE charge ignored FEs nested in other FEs, so ``year`` inside
  ``prefecture x year`` cost 14 extra degrees of freedom and every
  small-sample-scaled statistic (AR F, effective F, the IV SE) came out
  ~1.6% off.

Both now follow ``ivreg2`` / ``reghdfe``. Tolerance 1e-7 relative (the
demeaning tolerance on both sides).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
FE = ["year", "unit", "rXy"]


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "weakiv_nested_fe_Stata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "weakiv_nested_fe.csv")


def test_anderson_rubin_matches_ivreghdfe(ref, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ar = sp.anderson_rubin_test(
            data,
            y="y",
            endog="d",
            instruments=["z"],
            exog=["w1"],
            absorb=FE,
            cluster="unit",
        )
    assert ar["ar_stat"] == pytest.approx(ref["ar_f"], rel=1e-7)
    assert ar["ar_pvalue"] == pytest.approx(ref["ar_p"], rel=1e-6)
    assert ar["ar_df"] == (1, ref["N_clust"] - 1)
    assert ar["effective_F"] == pytest.approx(ref["kp_f"], rel=1e-7)


def test_iv_absorb_se_matches_ivreghdfe(ref, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.iv("y ~ (d ~ z) + w1", data=data, absorb=FE, cluster="unit")
    assert float(r.params["d"]) == pytest.approx(ref["b"], rel=1e-7)
    assert float(r.std_errors["d"]) == pytest.approx(ref["se"], rel=1e-7)


def test_nested_fe_charge_matches_reghdfe(ref, data):
    from statspai.inference._dof import absorbed_dof_charge

    charge, nested = absorbed_dof_charge(
        data[FE], FE, [data[c].nunique() for c in FE], data[["unit"]]
    )
    assert nested == ["unit"]
    assert charge == ref["df_a"]
