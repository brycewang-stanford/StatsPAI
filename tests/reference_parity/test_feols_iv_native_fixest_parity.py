"""``sp.feols`` IV with absorbed effects on its native engine vs R ``fixest``.

24 cells: formula {just-identified, over-identified, two endogenous
regressors} x vcov {iid, hetero, cluster g, cluster g + year} x weights
{none, w}, with firm and year effects and 1% singleton firms kept
(``fixef.rm = "none"``, ``sp.feols``'s default). Reference:
``_fixtures/_generate_feols_iv_native_R.R`` (fixest 0.14) on
``feols_iv_native_data.csv``. The first-stage F of a single endogenous
regressor is fixest's ``ivwald1`` (Wald statistic under the model's
variance).

pyfixest cannot serve as the reference here: it does not fit several
endogenous regressors, and its first stage drops singletons that the IV
model keeps, so its robust / clustered first-stage F differs from fixest's
by 1% on this panel.

Both sides demean iteratively; observed agreement is ~1e-10 on
coefficients and standard errors. Budget 1e-6 (CLAUDE.md section 5.1).
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "feols_iv_native_R.json").read_text(encoding="utf-8"))
RTOL = 1e-6

FMLS = {
    "just": "y ~ x2 | firm + year | x1 ~ z",
    "over": "y ~ x2 | firm + year | x1 ~ z + z2",
    "two": "y ~ x2 | firm + year | x1 + x3 ~ z + z2",
}
VCOV = {
    "iid": "iid",
    "hetero": "hetero",
    "cr1": {"CRV1": "g"},
    "twoway": {"CRV1": "g + year"},
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "feols_iv_native_data.csv")


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("vcov", sorted(VCOV))
@pytest.mark.parametrize("fml", sorted(FMLS))
def test_native_iv_matches_fixest(data, fml, vcov, weighted):
    kw = {"weights": "w"} if weighted else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols(FMLS[fml], data=data, vcov=VCOV[vcov], **kw)
    assert fit.model_info["backend"] == "statspai-native"
    ref = R[f"{fml}_{vcov}_{'w' if weighted else 'none'}"]
    assert fit.data_info["nobs"] == ref["nobs"]
    names = list(ref["coef"])
    np.testing.assert_allclose(
        fit.params[names], [ref["coef"][k] for k in names], rtol=RTOL
    )
    np.testing.assert_allclose(
        fit.std_errors[names], [ref["se"][k] for k in names], rtol=RTOL
    )
    if "ivwald1" in ref:
        np.testing.assert_allclose(
            fit.diagnostics["First-stage F (x1)"], ref["ivwald1"], rtol=RTOL
        )


def test_iv_without_fixed_effects_is_delegated(data):
    pytest.importorskip("pyfixest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols("y ~ 1 + x2 | x1 ~ z", data=data)
    assert fit.model_info["backend"] == "pyfixest"
