"""sp.gologit (generalized ordered logit / probit) against Stata's
user-written ``gologit2`` 3.2.8, used as a black box (outputs only).

Reference ``tests/fixtures/ldv_margins/stata_gologit2.do`` on the 600 rows
of ``_data.py``. gologit2 stops at Stata ``ml``'s default tolerance, so the
comparison is to 1e-6 (the estimates agree to ~3e-7).
"""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "ldv_margins"
_spec = importlib.util.spec_from_file_location("_gol_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_gologit2.json").read_text(encoding="utf-8"))
X = ["x1", "x2", "g2", "g3"]


@pytest.fixture(scope="module")
def df():
    return _mod.make_ldv_frame()


@pytest.mark.parametrize(
    "tag,kw",
    [
        ("npl", {}),
        ("pl", {"pl": True}),
        ("npl_x1", {"npl": ["x1"]}),
        ("pl_x2g2", {"pl": ["x2", "g2"]}),
        ("probit_npl_x1", {"npl": ["x1"], "link": "probit"}),
        ("npl_x1_robust", {"npl": ["x1"], "vce": "robust"}),
    ],
)
def test_matches_gologit2(df, tag, kw):
    r = sp.gologit(df, y="yo", x=X, **kw)
    ref = STATA[tag]
    names = list(r.params.index)
    assert names[0] == "0:x1" and names[-1] == "2:_cons"  # gologit2's layout
    np.testing.assert_allclose(r.params, [ref[k][0] for k in names], rtol=1e-6)
    np.testing.assert_allclose(r.std_errors, [ref[k][1] for k in names], rtol=1e-6)


def test_margins_match_gologit2(df):
    r = sp.gologit(df, y="yo", x=X, npl=["x1"])
    m = sp.margins(r, data=df)
    ref = {}
    for name, (b, se) in STATA["npl_x1_ame"].items():
        var, out = name.split(":")
        ref[(var, int(re.match(r"(\d+)", out).group(1)) - 1)] = (b, se)
    got = np.array([[row["dy/dx"], row["se"]] for _, row in m.iterrows()])
    want = np.array([ref[(row.variable, int(row.outcome))] for _, row in m.iterrows()])
    np.testing.assert_allclose(got, want, rtol=2e-6)
    # the outcome probabilities sum to one, so each variable's effects sum to 0
    np.testing.assert_allclose(m.groupby("variable")["dy/dx"].sum(), 0, atol=1e-10)


def test_parallel_lines_is_ologit(df):
    g = sp.gologit(df, y="yo", x=X, pl=True)
    o = sp.ologit(data=df, y="yo", x=X)
    np.testing.assert_allclose(g.params[[f"0:{v}" for v in X]], o.params[X], rtol=1e-7)
    cuts = o.params[["/cut1", "/cut2", "/cut3"]].to_numpy()
    np.testing.assert_allclose(
        g.params[["0:_cons", "1:_cons", "2:_cons"]], -cuts, rtol=1e-7
    )


def test_bad_inputs(df):
    with pytest.raises(MethodIncompatibility, match="pl= or npl="):
        sp.gologit(df, y="yo", x=X, pl=["x1"], npl=["x2"])
    with pytest.raises(MethodIncompatibility, match="regressors"):
        sp.gologit(df, y="yo", x=X, npl=["nope"])
    with pytest.raises(DataInsufficient, match="three categories"):
        sp.gologit(df, y="s", x=X)
