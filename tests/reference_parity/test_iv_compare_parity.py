"""``sp.iv_compare`` against R ``ivmodel`` and Stata ``ivregress`` / ``ivreg2`` / ``jive``.

Data: ``_fixtures/rd_iv_ivw.csv`` (800 rows, one endogenous regressor,
three instruments, two controls; the first stage is weak, F = 2.8, so the
k-class estimators are far apart and a mix-up between them would show).

References, all on the same file:

* R ``ivmodel`` 1.9.1 (GPL-2, black box): ``KClass(k = 1)``, ``LIML``,
  ``Fuller(b = 1)``, estimate and conventional standard error
  (``_generate_iv_compare_R.R`` -> ``iv_compare_R.json``).
* Stata 18 ``ivregress 2sls`` / ``ivregress liml`` and ``ivreg2,
  fuller(1)`` 4.1.12 with ``small``, default and robust VCE, and the
  first-stage F of ``estat firststage``
  (``_generate_iv_compare_Stata.do`` -> ``iv_compare_Stata.json``).
* Stata ``jive`` (st0108) ``ujive1``, default and robust, and
  ``weakivtest`` for the effective F: both already in
  ``rd_iv_Stata.json``.

Tolerance 1e-9 relative on every estimate, standard error and F
(observed <= 1e-11).

One reference divergence, recorded and not copied: with
``heteroSE = TRUE`` ``ivmodel`` reports 0.54824 for LIML and 0.45060 for
Fuller where Stata's robust VCE with ``small`` gives 0.54979 and 0.45173.
For 2SLS ``ivmodel``'s number is the HC0 sandwich and StatsPAI's
``robust='hc0'`` equals it. For ``k != 1`` the two packages use different
sandwich forms; StatsPAI follows Stata, which is what
``test_iv_stata_commands_parity.py`` already pins.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "iv_compare_R.json").read_text(encoding="utf-8"))
STATA = json.loads((_FIX / "iv_compare_Stata.json").read_text(encoding="utf-8"))
OLD = json.loads((_FIX / "rd_iv_Stata.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "rd_iv_ivw.csv")
FORMULA = "y ~ (d ~ z1 + z2 + z3) + x1 + x2"
REL = 1e-9


def _table(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the weak-instrument warning
        tab = sp.iv_compare(FORMULA, data=DATA, **kw)
    assert list(tab["status"]) == ["ok"] * 4
    return tab.set_index("method")


def test_fixtures_are_the_reference_outputs():
    assert R["versions"]["ivmodel"] == "1.9.1"
    assert STATA["versions"]["ivreg2"] == "4.1.12"
    # the estimators are far apart on these data
    b = {k: v["b"] for k, v in R["homoskedastic"].items()}
    assert b["liml"] < 0.06 < 0.17 < b["2sls"] < b["fuller"]


@pytest.mark.parametrize("method", ["2sls", "liml", "fuller"])
def test_conventional_rows_match_ivmodel_and_stata(method):
    tab = _table()
    for ref in (R["homoskedastic"][method], STATA[f"{method}_unadjusted"]):
        assert tab.loc[method, "estimate"] == pytest.approx(ref["b"], rel=REL)
        assert tab.loc[method, "SE"] == pytest.approx(ref["se"], rel=REL)


@pytest.mark.parametrize("method", ["2sls", "liml", "fuller"])
def test_robust_rows_match_stata(method):
    tab = _table(robust="hc1")
    ref = STATA[f"{method}_robust"]
    assert tab.loc[method, "estimate"] == pytest.approx(ref["b"], rel=REL)
    assert tab.loc[method, "SE"] == pytest.approx(ref["se"], rel=REL)


def test_hc0_two_stage_least_squares_matches_ivmodel_heteroSE():
    tab = _table(robust="hc0")
    ref = R["heteroSE"]["2sls"]
    assert tab.loc["2sls", "SE"] == pytest.approx(ref["se"], rel=REL)


def test_jive_row_matches_stata_jive():
    ref = OLD["jive"]["ujive1"]
    tab = _table()
    assert tab.loc["jive", "estimate"] == pytest.approx(ref["b"][0], rel=REL)
    assert tab.loc["jive", "SE"] == pytest.approx(ref["se"][0], rel=REL)
    ref = OLD["jive"]["ujive1_robust"]
    tab = _table(robust="hc0")
    assert tab.loc["jive", "SE"] == pytest.approx(ref["se"][0], rel=REL)


def test_first_stage_and_effective_f_match_stata():
    tab = _table()
    assert tab["first_stage_F"].nunique() == 1
    assert tab.loc["2sls", "first_stage_F"] == pytest.approx(
        STATA["first_stage_F_unadjusted"], rel=REL
    )
    tab = _table(robust="hc1")
    assert tab.loc["2sls", "first_stage_F"] == pytest.approx(
        STATA["first_stage_F_robust"], rel=REL
    )
    # Montiel Olea-Pflueger effective F, Stata weakivtest after ivreg2, robust
    assert tab.loc["2sls", "effective_F"] == pytest.approx(
        OLD["weakivtest"]["ivw3_hc"], rel=REL
    )


def test_intervals_are_wald_intervals_at_the_normal_quantile():
    tab = _table(alpha=0.1)
    z = 1.6448536269514722
    for method in tab.index:
        row = tab.loc[method]
        assert row["CI lower"] == pytest.approx(row["estimate"] - z * row["SE"])
        assert row["CI upper"] == pytest.approx(row["estimate"] + z * row["SE"])
