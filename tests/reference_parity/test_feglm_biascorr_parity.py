"""``sp.feglm(..., bias_correction=True)`` against R ``alpaca::biasCorr``.

Reference: ``_fixtures/_generate_feglm_biascorr_R.R`` on
``_fixtures/feglm_biascorr_data.csv`` (alpaca 0.3.5, GPL, used as a black
box: only its output is read; ``dev.tol=1e-14``). For logit and probit,
unit effects alone and unit plus period effects: the uncorrected and the
corrected slopes, the estimation sample, and the standard errors at the
corrected slopes (alpaca's ``hessian``, ``sandwich`` and ``clustered``
types, no small-sample factors).

Tolerance 1e-7: the uncorrected slopes of the two IRLS fits agree to
3.4e-8 (probit) and the correction and the variance inherit that.
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
D = pd.read_csv(_FIX / "feglm_biascorr_data.csv")
R = json.loads((_FIX / "feglm_biascorr_R.json").read_text(encoding="utf-8"))
CASES = {
    "logit_it": ("logit", "y ~ x1 + x2 | i + t"),
    "logit_i": ("logit", "y ~ x1 + x2 | i"),
    "probit_it": ("probit", "yp ~ x1 + x2 | i + t"),
    "probit_i": ("probit", "yp ~ x1 + x2 | i"),
}
VCOV = {"se_hessian": "iid", "se_sandwich": "hetero", "se_cluster": {"CRV1": "cl"}}


def _fit(key, vcov="iid", **kw):
    fam, fml = CASES[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.feglm(fml, D, family=fam, vcov=vcov, bias_correction=True, **kw)


@pytest.mark.parametrize("key", sorted(CASES))
@pytest.mark.parametrize("se_key", sorted(VCOV))
def test_matches_alpaca_biascorr(key, se_key):
    ref = R[key]
    res = _fit(key, VCOV[se_key])
    np.testing.assert_allclose(res.params.to_numpy(), ref["coef"], rtol=1e-7)
    np.testing.assert_allclose(res.std_errors.to_numpy(), ref[se_key], rtol=1e-7)
    np.testing.assert_allclose(
        list(res.model_info["coef_uncorrected"].values()), ref["coef_uncorr"], rtol=1e-7
    )
    assert res.data_info["nobs"] == ref["nobs"]


def test_shift_is_the_difference_and_default_is_uncorrected():
    res = _fit("logit_it")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.feglm("y ~ x1 + x2 | i + t", D, family="logit")
    np.testing.assert_allclose(
        plain.params.to_numpy(), list(res.model_info["coef_uncorrected"].values())
    )
    np.testing.assert_allclose(
        res.params.to_numpy() - plain.params.to_numpy(),
        list(res.model_info["bias_shift"].values()),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"family": "negbin"},
        {"family": "logit", "backend": "pyfixest"},
        {"family": "logit", "vcov": {"CRV1": "cl + t"}},
        {"family": "logit", "fml": "y ~ x1 + x2"},
    ],
)
def test_refusals(kw):
    kw = dict(kw)
    fml = kw.pop("fml", "y ~ x1 + x2 | i + t")
    with pytest.raises(MethodIncompatibility):
        sp.feglm(fml, D, bias_correction=True, **kw)
