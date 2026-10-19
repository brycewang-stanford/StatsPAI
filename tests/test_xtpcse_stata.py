"""sp.xtpcse against Stata 18 MP ``xtpcse`` (23 option combinations).

Reference ``tests/fixtures/xtpcse/stata_xtpcse.do`` on a balanced panel
(10 x 20) and an unbalanced one with interior gaps. Matching on the
unbalanced panel pinned how Stata forms the common AR(1) coefficient: each
panel's coefficient is weighted by its number of consecutive pairs (one
more with ``np1``), which is ``T_i - 1`` only for a panel without gaps.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "xtpcse"
_spec = importlib.util.spec_from_file_location("_xtpcse_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_xtpcse.json").read_text(encoding="utf-8"))

OPTIONS = {
    "default": {},
    "hetonly": {"hetonly": True},
    "independent": {"independent": True},
    "nmk": {"nmk": True},
    "ar1": {"correlation": "ar1"},
    "psar1": {"correlation": "psar1"},
    "ar1_dw": {"correlation": "ar1", "rhotype": "dw"},
    "ar1_tscorr": {"correlation": "ar1", "rhotype": "tscorr"},
    "ar1_freg": {"correlation": "ar1", "rhotype": "freg"},
    "ar1_hetonly": {"correlation": "ar1", "hetonly": True},
    "pairwise": {"pairwise": True},
    "ar1_np1": {"correlation": "ar1", "np1": True},
    "ar1_pairwise": {"correlation": "ar1", "pairwise": True},
}
CASES = [
    (panel, opt)
    for panel in ("tscs", "tscs_unbal")
    for opt in OPTIONS
    if f"{panel}_{opt}" in STATA
]


@pytest.fixture(scope="module")
def frames():
    return {"tscs": _mod.make_tscs(), "tscs_unbal": _mod.make_unbalanced()}


@pytest.mark.parametrize("panel,opt", CASES, ids=[f"{p}-{o}" for p, o in CASES])
def test_matches_stata_xtpcse(frames, panel, opt):
    r = sp.xtpcse(
        frames[panel], y="y", x=["x1", "x2"], id="id", time="t", **OPTIONS[opt]
    )
    ref = STATA[f"{panel}_{opt}"]
    names = ["x1", "x2", "_cons"]
    np.testing.assert_allclose(r.params[names], [ref[k][0] for k in names], rtol=1e-12)
    np.testing.assert_allclose(
        r.std_errors[names], [ref[k][1] for k in names], rtol=1e-12
    )
    assert r.data_info["nobs"] == ref["N"][0]
    if opt.startswith("ar1"):
        np.testing.assert_allclose(r.model_info["rho"], ref["rho"][0], rtol=1e-12)
    if opt == "psar1":  # Stata stores the panel coefficients as e(rho)
        got = list(r.model_info["rho"].values())
        np.testing.assert_allclose(got, ref["rho"][0], rtol=1e-12)


def test_independent_is_ols_with_n_divisor(frames):
    df = frames["tscs"]
    r = sp.xtpcse(df, y="y", x=["x1", "x2"], id="id", time="t", independent=True)
    o = sp.regress("y ~ x1 + x2", data=df)
    np.testing.assert_allclose(
        r.params[["x1", "x2"]], o.params[["x1", "x2"]], rtol=1e-12
    )
    n, k = len(df), 3
    np.testing.assert_allclose(
        r.std_errors[["x1", "x2"]],
        o.std_errors[["x1", "x2"]] * np.sqrt((n - k) / n),
        rtol=1e-10,
    )


def test_casewise_without_common_period_is_refused(frames):
    df = frames["tscs"]
    df = df[~((df["id"] == 1) & (df["t"] <= 10)) & ~((df["id"] == 2) & (df["t"] > 10))]
    with pytest.raises(DataInsufficient, match="pairwise"):
        sp.xtpcse(df, y="y", x="x1", id="id", time="t")


def test_bad_options_are_refused(frames):
    df = frames["tscs"]
    with pytest.raises(MethodIncompatibility, match="alternatives"):
        sp.xtpcse(df, y="y", x="x1", id="id", time="t", hetonly=True, independent=True)
    with pytest.raises(MethodIncompatibility, match="correlation"):
        sp.xtpcse(df, y="y", x="x1", id="id", time="t", correlation="ar2")
