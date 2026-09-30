"""``sp.oster_bounds(absorb=...)`` against ``psacalc`` after ``xtreg, fe``.

Empirical papers run ``xtreg y d x i.ind#i.year, fe`` and then ``psacalc``
with ``R_max = 1.3 * e(r2_a)``. ``oster_bounds`` could not: its data mode
had no fixed effects (the controls there are thousands of dummies), its
summary mode only had Oster's first-order approximation, and nothing gave
``xtreg``'s ``e(r2_a)`` -- which charges the panel means unless the VCE is
clustered on a variable the panel is nested in, and counts the dummies at
their exact rank given the panel effect.

Reference: Stata 18 ``xtreg y x z i.year i.city#i.year, fe [vce(cluster
city)]`` + ``psacalc`` 2.1 on ``_fixtures/reghdfe_fitstats.csv``, from
``_generate_oster_absorb_Stata.do``. ``r2_a`` / ``R_max`` and the
bias-adjusted beta agree to 1e-12; delta to 1e-8 (``psacalc`` finds it by
a root search).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "oster_absorb_Stata.json").read_text(encoding="utf-8"))
CASES = {"plain": ("1.3*r2", None), "cluster": ("1.3*r2_a", "city")}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "reghdfe_fitstats.csv")


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_psacalc_after_xtreg_fe(data, case):
    rule, cluster = CASES[case]
    r = sp.oster_bounds(
        data,
        y="y",
        treat="x",
        controls=["z"],
        absorb="id",
        absorb_controls="year + city#year",
        r_max=rule,
        cluster=cluster,
        delta=-1,
    )
    ref = REF[case]
    assert r["method"] == "exact"
    np.testing.assert_allclose(r["r2_a_long"], ref["r2_a"], rtol=1e-12)
    np.testing.assert_allclose(r["r_max"], ref["rmax"], rtol=1e-12)
    np.testing.assert_allclose(r["beta_adjusted"], ref["beta"], rtol=1e-10)
    np.testing.assert_allclose(r["delta_for_zero"], ref["delta"], rtol=1e-8)


def test_moments_give_the_exact_solution(data):
    full = sp.oster_bounds(
        data,
        y="y",
        treat="x",
        controls=["z"],
        absorb="id",
        absorb_controls="year + city^year",
        r_max=0.4,
    )
    inp = full["inputs"]
    summ = sp.oster_bounds(
        beta_short=inp["beta_o"],
        r2_short=inp["r_o"],
        beta_long=inp["beta_t"],
        r2_long=inp["r_t"],
        r_max=0.4,
        moments={k: inp[k] for k in ("sigma_yy", "sigma_xx", "t_x")},
    )
    assert summ["method"] == "exact"
    assert summ["delta_for_zero"] == pytest.approx(full["delta_for_zero"], rel=1e-12)
    assert summ["beta_adjusted"] == pytest.approx(full["beta_adjusted"], rel=1e-12)
    approx = sp.oster_bounds(
        beta_short=inp["beta_o"],
        r2_short=inp["r_o"],
        beta_long=inp["beta_t"],
        r2_long=inp["r_t"],
        r_max=0.4,
    )
    assert approx["method"] == "approximate"


def test_bad_rule_is_rejected(data):
    with pytest.raises(ValueError, match="r_max"):
        sp.oster_bounds(data, y="y", treat="x", controls=["z"], r_max="1.3*foo")
