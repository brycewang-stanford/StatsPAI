"""sp.intreg (interval regression) and sp.margins after it, against Stata 18.

Reference: ``tests/fixtures/ldv_margins/stata_margins.do`` (the ``intreg``
block) on ``_data.py``'s 600 rows: 91 exact values, 106 left-censored,
55 right-censored, 343 interval-coded, 5 rows with no bounds (dropped).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "ldv_margins"
_spec = importlib.util.spec_from_file_location("_ldv_data_i", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_margins.json").read_text(encoding="utf-8"))
FORMULA = "ilo + ihi ~ x1 + x2 + C(g)"
# ours: const, 2.g, 3.g, x1, x2, lnsigma; Stata: x1 x2 1b.g 2.g 3.g _cons
_ORDER = [3, 4, 1, 2, 0, 5]


@pytest.fixture(scope="module")
def df():
    return _mod.make_ldv_frame()


def _stata(tag):
    return np.array([v for v in STATA[tag].values() if v[1] > 0])


@pytest.mark.parametrize(
    "kw,tag",
    [
        ({}, "intreg_e"),
        ({"vce": "robust"}, "intreg_rob_e"),
        ({"weights": "w"}, "intreg_pw_e"),
    ],
)
def test_coefficients_match_stata(df, kw, tag):
    r = sp.intreg(df, formula=FORMULA, **kw)
    st = r._ml_state
    want = _stata(tag)  # the last row is /lnsigma
    np.testing.assert_allclose(st["theta"][_ORDER], want[:, 0], rtol=1e-9)
    se = np.sqrt(np.diag(st["vcov"]))
    np.testing.assert_allclose(se[_ORDER], want[:, 1], rtol=1e-9)
    np.testing.assert_allclose(r.params["sigma"], np.exp(want[-1, 0]), rtol=1e-9)


def test_counts_and_loglik_match_stata(df):
    mi = sp.intreg(df, formula=FORMULA).model_info
    # Stata: e(N_unc) e(N_lc) e(N_rc) e(N_int) e(ll)
    assert (
        mi["n_uncensored"],
        mi["n_left_censored"],
        mi["n_right_censored"],
        mi["n_interval"],
    ) == (91, 106, 55, 343)
    np.testing.assert_allclose(mi["log_likelihood"], -864.383563075362, rtol=1e-12)


@pytest.mark.parametrize(
    "tag,predict,weights,rtol",
    [
        ("intreg_xb", None, None, 2e-6),
        ("intreg_e0", "e(0,.)", None, 2e-6),
        ("intreg_ystar", "ystar(-1,3)", None, 2e-6),
        # Stata's numerical dy/dx of pr(0,2) is off by ~2e-6 here; ours is
        # the closed form to 1e-12 (test below)
        ("intreg_pr02", "pr(0,2)", None, 5e-6),
        ("intreg_pw_e0", "e(0,.)", "w", 2e-6),
    ],
)
def test_margins_match_stata(df, tag, predict, weights, rtol):
    r = sp.intreg(df, formula=FORMULA, weights=weights)
    m = sp.margins(r, data=df, predict=predict).set_index("variable")
    m = m.loc[["x1", "x2", "2.g", "3.g"]]
    want = _stata(tag)
    np.testing.assert_allclose(m["dy/dx"], want[:, 0], rtol=rtol)
    np.testing.assert_allclose(m["se"], want[:, 1], rtol=rtol)
    assert m.attrs["n"] == 595  # the rows without bounds are not averaged


def test_pr_margin_equals_closed_form(df):
    r = sp.intreg(df, formula=FORMULA)
    d = df[df[["ilo", "ihi"]].notna().any(axis=1)]
    X = np.column_stack([np.ones(len(d)), d.g == 2, d.g == 3, d.x1, d.x2]).astype(float)
    th, V = r._ml_state["theta"], r._ml_state["vcov"]

    def phi(a):
        return np.exp(-a * a / 2) / np.sqrt(2 * np.pi)

    def ame(t):  # d Pr(0 < y < 2) / d x1
        s = np.exp(t[-1])
        xb = X @ t[:-1]
        return np.mean(phi(-xb / s) - phi((2 - xb) / s)) * t[3] / s

    J = np.array(
        [np.imag(ame(th + 1j * 1e-30 * np.eye(6)[i])) / 1e-30 for i in range(6)]
    )
    m = sp.margins(r, data=df, predict="pr(0,2)", variables=["x1"])
    np.testing.assert_allclose(m["dy/dx"], [ame(th)], rtol=1e-10)
    np.testing.assert_allclose(m["se"], [np.sqrt(J @ V @ J)], rtol=1e-8)


def test_tobit_is_the_one_limit_special_case(df):
    # censored at 0 from below: lo missing and hi = 0; exact otherwise
    d = df.assign(lo=np.where(df.yt <= 0, np.nan, df.yt), hi=df.yt)
    a = sp.intreg(d, y_lower="lo", y_upper="hi", x=["x1", "x2"])
    b = sp.tobit(d, y="yt", x=["x1", "x2"], ll=0)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-7)


def test_inverted_bounds_are_refused(df):
    d = df.assign(lo=df.ihi + 1.0)
    with pytest.raises(MethodIncompatibility, match="ihi"):
        sp.intreg(d, y_lower="lo", y_upper="ihi", x=["x1"])


def test_formula_and_columns_agree(df):
    a = sp.intreg(df, formula="ilo + ihi ~ x1 + x2")
    b = sp.intreg(df, y_lower="ilo", y_upper="ihi", x=["x1", "x2"])
    np.testing.assert_allclose(a.params, b.params, rtol=1e-10)
