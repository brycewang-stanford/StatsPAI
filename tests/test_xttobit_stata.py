"""sp.xttobit / sp.xtintreg (random-effects tobit and interval regression)
against Stata 18 MP.

Reference ``tests/fixtures/xttobit/stata_xttobit.do`` on 120 unbalanced
panels. The panel likelihood is an integral over the random effect; Stata
approximates it by adaptive Gauss-Hermite quadrature, 12 nodes by default,
and so do we (30 nodes by default, centred at the mode). Both converge to
the exact integral, so the comparison is against Stata with 40 nodes,
where its quadrature error is far below the tolerance; the 12-node default
is compared at the looser tolerance its own quadrature error allows.
xtintreg's standard errors are compared with Stata's non-adaptive fit
(``intmethod(ghermite) intpoints(100)``), because its adaptive-quadrature
Hessian is not the curvature of its own likelihood (see the test).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "xttobit"
_spec = importlib.util.spec_from_file_location("_xttobit_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_xttobit.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df():
    return _mod.make_panel()


def _ref(tag, eq):
    """Stata's coefficients of one equation, in our order (x..., _cons)."""
    r = STATA[tag]
    return [r[f"{eq}:x1"], r[f"{eq}:x2"], r[f"{eq}:_cons"]]


def _check(fit, tag, eq, rtol_b, rtol_se):
    want = np.array(_ref(tag, eq))
    np.testing.assert_allclose(
        fit.params[["x1", "x2", "_cons"]], want[:, 0], rtol=rtol_b
    )
    np.testing.assert_allclose(
        fit.std_errors[["x1", "x2", "_cons"]], want[:, 1], rtol=rtol_se
    )
    r = STATA[tag]
    for ours, theirs in [("sigma_u", "e(sigma_u)"), ("sigma_e", "e(sigma_e)")]:
        np.testing.assert_allclose(fit.params[ours], r[theirs][0], rtol=rtol_b)
    np.testing.assert_allclose(fit.model_info["rho"], r["e(rho)"][0], rtol=rtol_b)
    np.testing.assert_allclose(
        fit.model_info["log_likelihood"], r["e(ll)"][0], rtol=rtol_b
    )
    assert fit.model_info["n_groups"] == r["e(N_g)"][0]


@pytest.mark.parametrize(
    "tag,kw",
    [("tobit_q40", {"ll": 0, "ul": 4}), ("tobit_ll_q40", {"ll": 0})],
)
def test_xttobit_matches_stata(df, tag, kw):
    fit = sp.xttobit(df, y="y", x=["x1", "x2"], id="id", **kw)
    _check(fit, tag, "y", rtol_b=1e-6, rtol_se=1e-6)
    # LR test against the pooled tobit, as xttobit's e(chi2_c)
    np.testing.assert_allclose(
        fit.model_info["lr_sigma_u0"], STATA[tag]["e(chi2_c)"][0], rtol=1e-7
    )


def test_xttobit_stata_default_quadrature(df):
    # Stata's 12-node default differs from its 40-node fit by ~3e-10
    fit = sp.xttobit(df, y="y", x=["x1", "x2"], id="id", ll=0, ul=4, intpoints=12)
    _check(fit, "tobit_q12", "y", rtol_b=1e-6, rtol_se=1e-6)


def test_xtintreg_matches_stata(df):
    fit = sp.xtintreg(df, y_lower="lo", y_upper="hi", x=["x1", "x2"], id="id")
    # Estimates and log-likelihood against the default (mvaghermite) fit.
    # Its reported standard errors come from xtintreg's analytic Hessian,
    # which under adaptive quadrature is not the curvature of its own
    # likelihood: central differences of Stata's e(ll) (from() iterate(0),
    # 40 mvaghermite nodes) give se(sigma_e) = 0.0385837, the reported value
    # is 0.0387655 (+0.47%). The curvature itself is checked below.
    _check(fit, "intreg_q40", "lo", rtol_b=1e-6, rtol_se=1e-2)


def test_xtintreg_se_matches_stata_nonadaptive(df):
    # With non-adaptive quadrature Stata's reported standard errors equal
    # the curvature of its likelihood, and match ours (estimates to 1e-10
    # because 100 plain nodes are accurate here, where 40 adaptive ones
    # leave a 1e-7 quadrature error in Stata's default fit)
    fit = sp.xtintreg(df, y_lower="lo", y_upper="hi", x=["x1", "x2"], id="id")
    _check(fit, "intreg_gh100", "lo", rtol_b=1e-9, rtol_se=1e-8)
    r = STATA["intreg_gh100"]
    for ours, theirs in [("sigma_u", "/sigma_u"), ("sigma_e", "/sigma_e")]:
        np.testing.assert_allclose(fit.std_errors[ours], r[theirs][1], rtol=1e-8)


def test_xttobit_is_xtintreg_with_tobit_coding(df):
    a = sp.xttobit(df, y="y", x=["x1", "x2"], id="id", ll=0)
    lo = df["y"].where(df["y"] > 0)
    b = sp.xtintreg(
        df.assign(L=lo, H=df["y"]), y_lower="L", y_upper="H", x=["x1", "x2"], id="id"
    )
    np.testing.assert_allclose(a.params, b.params, rtol=1e-10)


def test_without_limits_is_refused(df):
    with pytest.raises(MethodIncompatibility, match="ll="):
        sp.xttobit(df, y="y", x=["x1"], id="id")
