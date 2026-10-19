"""fixef / predict / marginaleffects after fits with absorbed fixed effects.

Against R ``fixest`` 0.14 and ``marginaleffects`` 0.32 (both GPL-3, used as
black boxes). Fixtures: ``_fixtures/_generate_fixest_postestimation_data.py``
then ``Rscript _fixtures/_generate_fixest_postestimation_R.R``.

Before this, ``result.fixef()`` did not exist, ``predict(newdata)`` left the
effects out (and returned the linear index of a Poisson fit), and
``sp.avg_slopes`` after ``sp.fepois`` returned the coefficients (the link
was not recorded, so the model was read as linear): 0.286 against
marginaleffects' 0.275 on the audit data.

Tolerances. The effects are recovered from the fitted index by alternating
projections run to 1e-13 relative; the fits themselves (IRLS for the GLMs)
agree with fixest to about 1e-8, so values are compared at ``rtol=1e-6``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
R = json.loads((FIX / "fixest_postestimation_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(FIX / "fixest_postestimation_data.csv")
TOL = dict(rtol=1e-6, atol=1e-8)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def fits():
    d = DATA
    return {
        "ols1": _quiet(sp.feols, "y ~ x1 + x2 | f1", d),
        "ols2": _quiet(sp.feols, "y ~ x1 + x2 | f1 + f2", d),
        "ols3": _quiet(sp.feols, "y ~ x1 + x2 | f1 + f2 + f3", d),
        "int": _quiet(sp.feols, "y ~ x1 * x2 | f1 + f2", d),
        "iv": _quiet(sp.feols, "yiv ~ x1 | f1 | d1 ~ z1 + z2", d),
        "pois": _quiet(sp.fepois, "yp ~ x1 + x2 | f1 + f2", d),
        "logit": _quiet(sp.feglm, "yb ~ x1 + x2 | f1", d, family="logit"),
    }


def _newdata():
    nd = DATA.iloc[:6].copy()
    nd.loc[nd.index[1], "f1"] = 99
    return nd


@pytest.mark.parametrize("name", ["ols1", "ols2", "ols3", "iv", "pois", "logit"])
def test_fixef_matches_fixest(fits, name):
    got = fits[name].fixef()
    want = R["fixef"][name]
    assert list(got) == list(want)
    for fe, ref in want.items():
        s = got[fe]
        assert [str(v) for v in s.index] == ref["level"]
        np.testing.assert_allclose(s.to_numpy(), ref["value"], **TOL)


@pytest.mark.parametrize("name", ["ols2", "ols3", "iv", "pois", "logit"])
def test_predict_newdata_adds_the_effects(fits, name):
    with pytest.warns(Warning, match="never saw"):
        got = fits[name].predict(newdata=_newdata())
    want = np.array([np.nan if v is None else v for v in R["predict"][name]])
    assert np.isnan(got[1]) and np.isnan(want[1])
    np.testing.assert_allclose(np.delete(got, 1), np.delete(want, 1), **TOL)


def test_predict_link_scale(fits):
    with pytest.warns(Warning, match="never saw"):
        got = fits["pois"].predict(_newdata(), what="link")
    want = np.array([np.nan if v is None else v for v in R["predict"]["pois_link"]])
    np.testing.assert_allclose(np.delete(got, 1), np.delete(want, 1), **TOL)


@pytest.mark.parametrize("name", ["ols2", "ols3", "pois", "logit", "iv"])
def test_in_sample_effects_reproduce_the_fit(fits, name):
    """Identity: predicting the estimation rows returns the fitted values."""
    res = fits[name]
    pred = res.predict(DATA)
    if name == "iv":  # no stored fitted values: y - residual
        fitted = DATA["yiv"].to_numpy() - np.asarray(res.data_info["residuals"])
    else:
        fitted = np.asarray(res.predict())
    np.testing.assert_allclose(pred, fitted, rtol=1e-9, atol=1e-9)


def test_avg_slopes_linear_keeps_standard_errors(fits):
    got = sp.avg_slopes(fits["int"])
    ref = R["avg_slopes"]["int"]
    np.testing.assert_allclose(got["estimate"], ref["estimate"], **TOL)
    # marginaleffects differentiates numerically (see the marginaleffects test)
    np.testing.assert_allclose(got["std_error"], ref["std.error"], rtol=2e-4)
    cmp = sp.avg_comparisons(fits["int"])
    np.testing.assert_allclose(
        cmp["estimate"], R["avg_comparisons"]["int"]["estimate"], **TOL
    )


@pytest.mark.parametrize("name", ["pois", "logit"])
def test_avg_slopes_nonlinear_include_the_effects(fits, name):
    got = sp.avg_slopes(fits[name], vcov=False)
    np.testing.assert_allclose(
        got["estimate"], R["avg_slopes"][name]["estimate"], **TOL
    )
    assert "std_error" not in got.columns
    # sp.margins (Stata-style) gives the same average marginal effects; its
    # standard errors would ignore the effects' uncertainty, so it refuses
    # them unless vcov=False (tests/test_margins_fe_refusal.py)
    m = _quiet(sp.margins, fits[name], DATA, vcov=False)
    np.testing.assert_allclose(m["dy/dx"], R["avg_slopes"][name]["estimate"], **TOL)
    assert "se" not in m.columns


@pytest.mark.parametrize("name", ["ols2", "pois", "logit"])
def test_avg_predictions(fits, name):
    got = sp.avg_predictions(fits[name], vcov=False)
    np.testing.assert_allclose(got["estimate"], R["avg_predictions"][name], **TOL)


def test_avg_comparisons_poisson(fits):
    got = sp.avg_comparisons(fits["pois"], vcov=False)
    np.testing.assert_allclose(
        got["estimate"], R["avg_comparisons"]["pois"]["estimate"], **TOL
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda f: sp.avg_slopes(f["pois"]),
        lambda f: sp.avg_predictions(f["ols2"]),
        lambda f: sp.predictions(f["logit"]),
        lambda f: sp.avg_slopes(f["int"], slope="eyex"),
    ],
)
def test_standard_errors_that_ignore_the_effects_are_refused(fits, call):
    with pytest.raises(MethodIncompatibility, match="vcov=False"):
        call(fits)


def test_unseen_level_in_margins_is_refused(fits):
    with pytest.raises(MethodIncompatibility, match="never saw"):
        sp.avg_slopes(fits["pois"], newdata=_newdata(), vcov=False)


def test_pyfixest_backend_predictions_agree():
    """backend='pyfixest': the effects are recovered from pyfixest's fitted
    index on its estimation data, as for a native fit. pyfixest's IRLS /
    demeaning stops earlier than the native engine (its own ``fixef()`` was
    5e-7 from fixest's here), hence ``atol=2e-6``."""
    pytest.importorskip("pyfixest")
    tol = dict(rtol=0, atol=2e-6)
    res = _quiet(sp.feols, "y ~ x1 + x2 | f1 + f2", DATA, backend="pyfixest")
    with pytest.warns(Warning, match="never saw"):
        got = res.predict(newdata=_newdata())
    want = np.array([np.nan if v is None else v for v in R["predict"]["ols2"]])
    np.testing.assert_allclose(np.delete(got, 1), np.delete(want, 1), **tol)
    got = sp.avg_predictions(res, vcov=False)
    np.testing.assert_allclose(got["estimate"], R["avg_predictions"]["ols2"], **tol)
    # identity: x'b plus the effects of the row reproduces the fitted values
    fe = res.fixef()
    s = DATA["f1"].map(fe["f1"]) + DATA["f2"].map(fe["f2"])
    xb = DATA[["x1", "x2"]].to_numpy() @ res.params.to_numpy()
    np.testing.assert_allclose(xb + s, res.predict(), **tol)


def test_hypotheses_joint_matches_marginaleffects():
    h = R["hypotheses"]
    g = _quiet(sp.glm, "yb ~ x1 + x2 + d1", data=DATA, family="binomial")
    jf = sp.hypotheses(g, joint=["x1", "x2"])
    assert jf["df1"].iloc[0] == h["joint_f"]["df1"]
    assert jf["df2"].iloc[0] == h["joint_f"]["df2"]
    np.testing.assert_allclose(jf["statistic"], h["joint_f"]["statistic"], rtol=1e-6)
    jc = sp.hypotheses(g, joint=["x1", "x2", "d1"], joint_test="chisq")
    np.testing.assert_allclose(
        jc["statistic"], h["joint_chisq"]["statistic"], rtol=1e-6
    )
    ja = sp.hypotheses(g, joint=True)
    np.testing.assert_allclose(ja["statistic"], h["joint_all"]["statistic"], rtol=1e-6)
    hv = sp.hypotheses(g, ["x1 = 0", "x2 = 0"])
    np.testing.assert_allclose(hv["estimate"], h["vector"]["estimate"], rtol=1e-6)
    np.testing.assert_allclose(hv["std_error"], h["vector"]["std.error"], rtol=1e-6)
    # a joint test written as restrictions is the same Wald statistic
    jr = sp.hypotheses(g, ["x1 = 0", "x2 = 0"], joint=True)
    np.testing.assert_allclose(jr["statistic"], jf["statistic"], rtol=1e-12)


@pytest.mark.parametrize("kind", ["DK", "NW"])
def test_panel_hac_matches_fixest(kind):
    """Driscoll-Kraay / Newey-West with the ids given (pyfixest computes them)."""
    pytest.importorskip("pyfixest")
    res = _quiet(
        sp.feols,
        "y ~ x1 + x2 | id",
        DATA,
        vcov=kind,
        vcov_kwargs={"time_id": "t", "panel_id": "id", "lag": 2},
    )
    np.testing.assert_allclose(res.std_errors, R["hac"][kind], rtol=1e-8)


def test_panel_hac_without_ids_is_a_typed_error():
    fit = _quiet(sp.feols, "y ~ x1 + x2 | id", DATA)
    with pytest.raises(MethodIncompatibility, match="vcov_kwargs"):
        sp.vcov(fit, "DK")
    with pytest.raises(MethodIncompatibility, match="lag"):
        sp.feols("y ~ x1 | id", DATA, vcov="NW", vcov_kwargs={"time_id": "t"})
