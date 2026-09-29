"""Weighted shift-share inference: ``weights=`` for ``sp.ssaggregate``,
``sp.bartik`` and ``sp.shift_share_se``.

The IV / shift-share functions accepted no weights, so a design estimated
with regional population weights (ADH, AER 2013, ``[aw = timepwt48]``) could
not be reproduced, and its AKM inference not computed. References on
``_fixtures/shiftshare_weighted_{loc,shares,shocks}.csv`` (400 locations, 25
industries, lognormal weights):

* R 4.5.2, ``ShiftShareSE`` 1.1.0 ``ivreg_ss.fit(..., w = w)`` and
  ``reg_ss.fit(..., w = w)`` -- coefficient and the Homoscedastic / EHW / AKM
  / AKM0 rows (``_generate_shiftshare_weighted_R.R``);
* Stata 18, ``ivregress 2sls y c1 c2 (x = z) [aw = w], vce(robust) small``
  for ``sp.bartik``'s weighted HC1 (``_generate_shiftshare_weighted_Stata.do``).

All agree to 1e-12. The BHJ shock-level IV with ``l_weights`` equals the
weighted location-level IV (the Borusyak-Hull-Jaravel equivalence).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "shiftshare_weighted_R.json").read_text(encoding="utf-8"))
ST = json.loads((_FIX / "shiftshare_weighted_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    loc = pd.read_csv(_FIX / "shiftshare_weighted_loc.csv")
    W = pd.read_csv(_FIX / "shiftshare_weighted_shares.csv", header=None).to_numpy()
    g = pd.read_csv(_FIX / "shiftshare_weighted_shocks.csv")["g"].to_numpy()
    return loc, W, g


@pytest.mark.parametrize("mode", ["iv", "ols"])
def test_ssaggregate_weighted_matches_shiftshare_se(data, mode):
    loc, W, g = data
    ref = R[mode]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "iv":
            r = sp.ssaggregate(
                loc,
                y="y",
                x="x",
                shares=W,
                shocks=g,
                controls=["c1", "c2"],
                weights="w",
            )
            coef = r.params["x"]
        else:
            r = sp.ssaggregate(
                loc, y="y", x="z", shares=W, controls=["c1", "c2"], weights="w"
            )
            coef = r.params["z"]
    assert coef == pytest.approx(ref["beta"], rel=1e-12)
    for row in ("Homoscedastic", "EHW", "AKM"):
        assert r.diagnostics[f"SE ({row})"] == pytest.approx(ref["se"][row], rel=1e-12)
    assert r.diagnostics["CI lower (AKM0, 95%)"] == pytest.approx(
        ref["ci_l"]["AKM0"], rel=1e-10
    )
    assert r.diagnostics["CI upper (AKM0, 95%)"] == pytest.approx(
        ref["ci_r"]["AKM0"], rel=1e-10
    )
    if mode == "iv":
        # BHJ equivalence with l_weights
        assert r.diagnostics["beta (BHJ shock-level)"] == pytest.approx(coef, rel=1e-10)


def test_bartik_weighted_matches_ivregress_and_akm(data):
    loc, W, g = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sp.bartik(
            loc,
            y="y",
            endog="x",
            shares=pd.DataFrame(W),
            shocks=pd.Series(g),
            covariates=["c1", "c2"],
            weights="w",
        )
        ss = sp.shift_share_se(b, shares=W)
    assert b.params["x"] == pytest.approx(ST["b"], rel=1e-12)
    assert b.std_errors["x"] == pytest.approx(ST["se"], rel=1e-12)
    assert b.std_errors["c1"] == pytest.approx(ST["se_c1"], rel=1e-12)
    assert ss.diagnostics["SE (AKM)"] == pytest.approx(R["iv"]["se"]["AKM"], rel=1e-12)


def test_unit_weights_change_nothing(data):
    loc, W, g = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.ssaggregate(loc, y="y", x="x", shares=W, shocks=g, controls=["c1"])
        b = sp.ssaggregate(
            loc.assign(one=3.0),
            y="y",
            x="x",
            shares=W,
            shocks=g,
            controls=["c1"],
            weights="one",
        )
    assert a.params["x"] == pytest.approx(b.params["x"], rel=1e-12)
    assert a.diagnostics["SE (AKM)"] == pytest.approx(
        b.diagnostics["SE (AKM)"], rel=1e-12
    )


def test_negative_weights_raise(data):
    loc, W, g = data
    with pytest.raises(sp.MethodIncompatibility, match="non-negative"):
        sp.ssaggregate(
            loc.assign(w=-loc.w), y="y", x="x", shares=W, shocks=g, weights="w"
        )
