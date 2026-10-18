"""``sp.feglm(family='logit' | 'probit')`` on its native engine vs R ``fixest``.

32 cells: family {logit, probit} x absorbed effects {firm, firm + year} x
vcov {iid, hetero, cluster g, cluster g + year} x weights {none, w}, plus
Poisson (``sp.fepois``) clustered on g + year with and without weights. Reference:
``_fixtures/_generate_feglm_native_R.R`` (fixest 0.14, ``glm.tol`` and
``fixef.tol`` 1e-10 so the reference is the optimum, not fixest's default
stopping point) on ``feglm_native_data.csv``, whose firm effects include
singletons and an all-zero firm, so fixest's perfect-fit rule drops rows.

Observed: coefficients 1.3e-13, standard errors 1.0e-6 (both sides stop
iterating at a tolerance; the SEs depend on the fixed effects, which the
1e-10 fixest stop leaves ~1e-7 from the optimum). Budget 1e-6 for
coefficients and 5e-6 for SEs (CLAUDE.md section 5.1: same estimator,
iterative solver on both sides).
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "feglm_native_R.json").read_text(encoding="utf-8"))
RTOL_COEF = 1e-6
RTOL_SE = 5e-6  # iterative on both sides; observed <= 1.0e-6

FES = {"one": "d ~ x1 + x2 | firm", "multi": "d ~ x1 + x2 | firm + year"}
VCOV = {
    "iid": "iid",
    "hetero": "hetero",
    "cr1": {"CRV1": "g"},
    "twoway": {"CRV1": "g + year"},
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "feglm_native_data.csv")


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("vcov", sorted(VCOV))
@pytest.mark.parametrize("fe", sorted(FES))
@pytest.mark.parametrize("family", ["logit", "probit"])
def test_native_feglm_matches_fixest(data, family, fe, vcov, weighted):
    kw = {"weights": "w"} if weighted else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feglm(FES[fe], data=data, family=family, vcov=VCOV[vcov], **kw)
    assert fit.model_info["backend"] == "statspai-native"
    ref = R[f"{family}_{fe}_{vcov}_{'w' if weighted else 'none'}"]
    assert fit.data_info["nobs"] == ref["nobs"]
    np.testing.assert_allclose(fit.params[["x1", "x2"]], ref["coef"], rtol=RTOL_COEF)
    np.testing.assert_allclose(fit.std_errors[["x1", "x2"]], ref["se"], rtol=RTOL_SE)
    # fixest's GLM p-values are normal, not t. Compared on the log scale:
    # at p ~ 1e-19 an SE difference of 1e-6 moves p by 1e-4 relative.
    np.testing.assert_allclose(
        np.log(fit.pvalues[["x1", "x2"]].to_numpy()), np.log(ref["pvalue"]), rtol=1e-5
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_native_fepois_twoway_matches_fixest(data, weighted):
    """Two-way clustering with fixest's eigenvalue fix (pyfixest's GLM
    covariance skips it, which moved the standard errors by 5%)."""
    # fixest's default sample: all-zero groups and singletons dropped
    # (sp.fepois keeps singletons unless fixef_rm='singleton').
    kw = {"weights": "w"} if weighted else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.fepois(
            FES["multi"],
            data=data,
            vcov=VCOV["twoway"],
            fixef_rm="singleton",
            **kw,
        )
    assert fit.model_info["backend"] == "statspai-native"
    ref = R[f"poisson_multi_twoway_{'w' if weighted else 'none'}"]
    assert fit.data_info["nobs"] == ref["nobs"]
    np.testing.assert_allclose(fit.params[["x1", "x2"]], ref["coef"], rtol=RTOL_COEF)
    np.testing.assert_allclose(fit.std_errors[["x1", "x2"]], ref["se"], rtol=RTOL_SE)


def test_perfect_fit_drop_warns(data):
    with pytest.warns(UserWarning, match="only 0 or only 1"):
        sp.feglm(FES["one"], data=data, family="logit", vcov="iid")
