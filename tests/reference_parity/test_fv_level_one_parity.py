"""``i.f#1.d`` translated by ``sp.from_stata`` reproduces Stata.

Stata codes ``i.visit#1.treat`` (and ``ibn.visit#1.treat``) as one column per
level of ``visit``, every level kept: the visit indicator times
``treat == 1``. The translator writes it as ``i.visit#c.treat`` for
``sp.hdfe_ols`` and ``C(visit):treat`` for ``sp.regress``. A lone
``i.visit#c.x1`` is likewise a slope for every level.

Reference: Stata 18 MP, reghdfe 6.13.1, on the committed CSV
(``_fixtures/_generate_fv_level_one_Stata.do``). Tolerance rel 1e-9: the
CSV holds 6-decimal values read identically by both sides, and the largest
gap observed is 7e-12 (iterative demeaning in reghdfe).
"""

from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

DATA = Path(__file__).parent / "_fixtures" / "fv_level_one.csv"

# reghdfe savings ibn.visit#1.treat ibn.visit#c.x1, absorb(visit week)
#   vce(cluster hhid)
RHD = {
    "visit::2#treat": (-0.0685204440812609, 0.165040536195178),
    "visit::3#treat": (0.668277625510619, 0.16018641652568),
    "visit::4#treat": (0.128998291131529, 0.15172673811621),
    "visit::2#x1": (0.465696380134155, 0.0849858932586468),
    "visit::3#x1": (0.425653493918422, 0.0855192415544983),
    "visit::4#x1": (0.811391529099768, 0.0804156284462854),
}
# regress savings i.visit#1.treat x1 i.visit, vce(cluster hhid)
REG = {
    "C(visit)[2]:treat": (-0.0846935299785676, 0.165326694495392),
    "C(visit)[3]:treat": (0.658519053981254, 0.162721796525365),
    "C(visit)[4]:treat": (0.189053378039017, 0.155791364396154),
}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def _run(cmd, df):
    t = sp.from_stata(cmd)
    assert t["ok"], t
    return t, eval(t["python_code"], {"sp": sp, "df": df})


def test_reghdfe_level_one_and_lone_slope(df):
    t, r = _run(
        "reghdfe savings ibn.visit#1.treat ibn.visit#c.x1, "
        "absorb(visit week) vce(cluster hhid)",
        df,
    )
    assert "i.visit#c.treat" in t["python_code"]
    assert any("0/1 indicator" in s for s in t["semantics"])
    for name, (b, se) in RHD.items():
        assert r.params[name] == pytest.approx(b, rel=1e-9)
        assert r.std_errors[name] == pytest.approx(se, rel=1e-9)


def test_regress_level_one(df):
    _, r = _run("regress savings i.visit#1.treat x1 i.visit, vce(cluster hhid)", df)
    for name, (b, se) in REG.items():
        assert r.params[name] == pytest.approx(b, rel=1e-9)
        assert r.std_errors[name] == pytest.approx(se, rel=1e-9)


def test_other_levels_are_still_refused():
    t = sp.from_stata("regress y i.visit#2.treat x")
    assert not t["ok"]
