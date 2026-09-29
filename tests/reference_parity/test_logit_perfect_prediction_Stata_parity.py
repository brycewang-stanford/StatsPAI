"""``sp.logit`` / ``sp.probit`` drop perfectly-predicting indicators (Stata).

A replication's first-stage logit with industry dummies had industries in
which no firm was treated. Stata reports ``"x != 0 predicts failure
perfectly"``, drops those firms and omits the dummy (N 9,870, pseudo R2
0.070); StatsPAI kept them silently (N 10,345, pseudo R2 0.083).

Reference: Stata 18 ``logit y x i.g`` / ``probit y x i.g`` on
``_fixtures/logit_perfect_prediction.csv`` (category 2 all failures,
category 5 all successes), from
``_generate_logit_perfect_prediction_Stata.do``. ``N`` exact; logit
estimates to 1e-12; probit to 1e-7 (Stata's Newton convergence
tolerance -- the two stop at slightly different iterates).
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
REF = json.loads(
    (_FIX / "logit_perfect_prediction_Stata.json").read_text(encoding="utf-8")
)
RTOL = {"logit": 1e-12, "probit": 1e-7}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "logit_perfect_prediction.csv")


@pytest.mark.parametrize("model", ["logit", "probit"])
def test_matches_stata(data, model):
    with pytest.warns(UserWarning, match="predicts failure perfectly"):
        r = getattr(sp, model)("y ~ x + C(g)", data=data)
    ref = REF[model]
    assert r.data_info["nobs"] == ref["N"]
    assert r.model_info["perfect_prediction_omitted"] == ["C(g)[T.2]", "C(g)[T.5]"]
    tol = RTOL[model]
    np.testing.assert_allclose(r.model_info["pseudo_r2"], ref["r2_p"], rtol=tol)
    np.testing.assert_allclose(r.model_info["ll"], ref["ll"], rtol=tol)
    np.testing.assert_allclose(r.params["x"], ref["b_x"], rtol=tol)
    np.testing.assert_allclose(r.std_errors["x"], ref["se_x"], rtol=tol)
    np.testing.assert_allclose(r.params["C(g)[T.3]"], ref["b_g3"], rtol=tol)
    np.testing.assert_allclose(r.std_errors["C(g)[T.3]"], ref["se_g3"], rtol=tol)


def test_keep_restores_full_sample(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.logit("y ~ x + C(g)", data=data, perfect_prediction="keep")
    assert r.data_info["nobs"] == len(data)
    assert r.model_info["n_perfect_prediction_dropped"] == 0


def test_no_indicator_no_change(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to drop, nothing to warn
        r = sp.logit("y ~ x", data=data)
    assert r.data_info["nobs"] == len(data)
