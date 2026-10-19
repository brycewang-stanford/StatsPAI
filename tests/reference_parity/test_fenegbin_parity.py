"""``sp.feglm(family='negbin')`` against R ``fixest::fenegbin`` and
``MASS::glm.nb``.

References (``_fixtures/_generate_fenegbin_R.R`` on
``_fixtures/fenegbin_data.csv``; fixest 0.14.0 with ``fixef.tol=1e-11``,
MASS 7.3; both GPL, used as black boxes):

* coefficients, ``theta``, log likelihood and the number of observations
  after fixest's ``perfect_fit`` removal (five all-zero firms and three
  singletons, 121 rows);
* standard errors: iid (inverse observed information of the coefficients
  and ``theta`` with the effects profiled out), heteroskedasticity-robust,
  clustered one-way (on a variable crossing the effects, on a nested
  effect) and two-way;
* the weighted maximum from ``glm.nb(weights=)`` (fenegbin takes no
  weights).

Tolerances: coefficients and ``theta`` 1e-8 against fixest (its optimiser
stops at about that; against ``glm.nb`` with ``epsilon=1e-14`` the
agreement is 1e-9), standard errors 1e-6 (they inherit the coefficient
error through ``mu``), log likelihood 1e-10.
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

_FIX = Path(__file__).parent / "_fixtures"
D = pd.read_csv(_FIX / "fenegbin_data.csv")
R = json.loads((_FIX / "fenegbin_R.json").read_text(encoding="utf-8"))


def _fit(fml="y ~ x1 + x2 | g + t", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.feglm(fml, D, family="negbin", **kw)


def _check(res, ref, se_rtol=1e-6):
    k = len(res.params)
    np.testing.assert_allclose(res.params.to_numpy(), ref["coef"], rtol=1e-8)
    np.testing.assert_allclose(res.model_info["theta"], ref["theta"], rtol=1e-8)
    np.testing.assert_allclose(res.std_errors.to_numpy(), ref["se"][:k], rtol=se_rtol)
    np.testing.assert_allclose(res.model_info["theta_se"], ref["se"][k], rtol=se_rtol)
    assert res.data_info["nobs"] == ref["nobs"]
    np.testing.assert_allclose(
        res.diagnostics["Log-likelihood"], ref["loglik"], rtol=1e-10
    )


@pytest.mark.parametrize(
    "key, kw",
    [
        ("two_iid", {}),
        ("two_hetero", {"vcov": "hetero"}),
        ("two_cl", {"vcov": {"CRV1": "cl"}}),
        ("two_cl_g", {"vcov": {"CRV1": "g"}}),
        ("two_cl_gt", {"vcov": {"CRV1": "g + t"}}),
    ],
)
def test_two_way_fe_matches_fenegbin(key, kw):
    _check(_fit(**kw), R[key])


def test_one_fe_nested_cluster_and_no_fe():
    _check(_fit("y ~ x1 + x2 | g", vcov={"CRV1": "g"}), R["one_cl_g"])
    _check(_fit("y ~ x1 + x2"), R["none_iid"])


def test_is_the_maximum_glm_nb_finds():
    res = _fit()
    np.testing.assert_allclose(res.params.to_numpy(), R["glmnb"]["coef"], rtol=1e-9)
    np.testing.assert_allclose(res.model_info["theta"], R["glmnb"]["theta"], rtol=1e-8)
    w = _fit(weights="w")
    np.testing.assert_allclose(w.params.to_numpy(), R["glmnb_w"]["coef"], rtol=1e-8)
    np.testing.assert_allclose(w.model_info["theta"], R["glmnb_w"]["theta"], rtol=1e-8)


def test_perfect_fit_rows_are_reported():
    with pytest.warns(UserWarning, match="perfect fit"):
        res = sp.feglm("y ~ x1 + x2 | g + t", D, family="negbin")
    assert res.model_info["n_dropped_separation"] == 121
    # the fitted means are the in-sample predictions
    keep = ~D["g"].isin([10, 11, 12, 13, 14, 77, 78, 79])
    np.testing.assert_allclose(
        res.predict(D[keep]), res.data_info["fitted_values"], rtol=1e-10
    )


def test_refusals():
    with pytest.raises(MethodIncompatibility):
        sp.feglm("y ~ x1 | g", D, family="negbin", backend="pyfixest")
    with pytest.raises(MethodIncompatibility):
        sp.feglm("y ~ x1 | g", D, family="negbin", vcov="CR2", cluster="g")
    bad = D.assign(y=D["y"] - 0.5)
    with pytest.raises(MethodIncompatibility):
        sp.feglm("y ~ x1 | g", bad, family="negbin")
