"""``sp.rdrobust`` / ``sp.rdbwselect`` ``masspoints=`` vs R rdrobust 4.0.0.

On a running variable with heavy ties (a 0.02 grid, ~97% of observations
tied) R's default ``masspoints='adjust'`` changes the data-driven bandwidth:
the reference bandwidth uses the unique-value count and a ``bwcheck = 10``
floor engages. ``'check'`` only warns and ``'off'`` treats x as continuous,
so both give the unadjusted bandwidths. StatsPAI always applied the
adjustment; the option makes the other two reachable. Reference:
``_fixtures/_generate_rd_masspoints_R.R``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "rd_masspoints_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "rd_masspoints.csv", float_precision="round_trip")
RTOL = 1e-8


@pytest.mark.parametrize("mp", ["adjust", "check", "off"])
@pytest.mark.parametrize("bs", ["mserd", "msetwo", "cerrd"])
def test_rdbwselect(mp, bs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdbwselect(D, y="y", x="x", bwselect=bs, masspoints=mp)
    ref = R[f"bw_{mp}_{bs}"]
    row = out.iloc[0]
    for k in ("h_left", "h_right", "b_left", "b_right"):
        assert row[k] == pytest.approx(ref[k], rel=RTOL), k
    assert out.attrs["detected"] and out.attrs["masspoints"] == mp


@pytest.mark.parametrize("mp", ["adjust", "check", "off"])
def test_rdrobust(mp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rdrobust(D, y="y", x="x", masspoints=mp)
    ref = R[f"rd_{mp}"]
    mi = r.model_info
    assert mi["bandwidth_h"] == pytest.approx(ref["h"], rel=RTOL)
    assert mi["bandwidth_b"] == pytest.approx(ref["b"], rel=RTOL)
    assert mi["conventional"]["estimate"] == pytest.approx(ref["coef"], rel=RTOL)
    assert mi["conventional"]["se"] == pytest.approx(ref["se_conv"], rel=RTOL)
    assert r.estimate == pytest.approx(ref["coef_bc"], rel=RTOL)
    assert r.se == pytest.approx(ref["se_rb"], rel=RTOL)


def test_check_warns_and_bad_value_raises():
    with pytest.warns(UserWarning, match="mass points detected"):
        sp.rdbwselect(D, y="y", x="x", masspoints="check")
    with pytest.raises(sp.MethodIncompatibility, match="masspoints"):
        sp.rdbwselect(D, y="y", x="x", masspoints="yes")
