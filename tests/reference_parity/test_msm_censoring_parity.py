"""``sp.msm(censor=)``: stabilized inverse-probability-of-censoring weights
for person-period data, against R ``ipw::ipwtm`` 1.3.0 and a known-truth
DGP with informative dropout.

Frozen reference: ``_fixtures/msm_censoring_R.json`` from
``_generate_msm_censoring_R.R`` on ``_fixtures/msm_censoring_data.csv``
(500 units, 4 periods, 231 censored). The censoring weights are
``ipwtm(exposure = cens, family = "binomial", type = "first")``, the
treatment weights ``ipwtm(type = "all")`` on the uncensored rows, the MSM
``lm`` with ``vcovCL(type = "HC1")``.

Tolerance 1e-6 relative on weights, coefficients and SEs (observed 1e-9 /
2e-8 / 6e-9): R's ``glm`` stops at ``epsilon = 1e-8``, statsmodels' logit
at its own tolerance, and the four logistic fits carry that into the
weights.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.msm._censoring import censoring_weights

_FIX = pathlib.Path(__file__).parent / "_fixtures"
KW = dict(y="y", treat="a", id="id", time="t", time_varying=["l"], baseline=["v"])


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "msm_censoring_data.csv")


@pytest.fixture(scope="module")
def ref():
    with open(_FIX / "msm_censoring_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_censoring_weights_match_ipwtm(data, ref):
    cw, diag = censoring_weights(data, "cens", "a", "id", "t", ["v"], ["l"])
    srt = data.sort_values(["id", "t"])
    unc = srt["cens"].to_numpy() == 0
    np.testing.assert_allclose(
        cw.loc[srt.index].to_numpy()[unc], ref["censoring_weights"], rtol=1e-6
    )
    assert diag["n_censored_units"] == int(data["cens"].sum())


def test_msm_with_censoring_weights_matches_r(data, ref):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.msm(data, censor="cens", **KW)
    np.testing.assert_allclose(r.detail["estimate"], ref["coef"], rtol=1e-6)
    np.testing.assert_allclose(r.detail["se"], ref["se"], rtol=1e-6)
    # without censor= the same rows give the treatment-weights-only fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = sp.msm(data[data["cens"] == 0], **KW)
    np.testing.assert_allclose(
        r0.detail["estimate"], ref["coef_no_censoring_weights"], rtol=1e-6
    )


def _dgp(n, seed, T=4):
    """U unmeasured drives L and Y; L drives treatment and dropout; the
    MSM slope on cumulative treatment is exactly 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        U, V, a_prev, cum = rng.normal(), rng.normal(), 0.0, 0.0
        for t in range(T):
            L = 0.8 * U + 0.4 * a_prev + rng.normal()
            if rng.uniform() < 1 / (1 + np.exp(-(-2.2 + L - 0.3 * a_prev))):
                rows.append((i, t, np.nan, L, V, np.nan, 1))
                break
            a = float(rng.uniform() < 1 / (1 + np.exp(-(-0.2 + 0.7 * L + 0.2 * V))))
            cum += a
            rows.append((i, t, a, L, V, cum + 1.5 * U + rng.normal(), 0))
            a_prev = a
    return pd.DataFrame(rows, columns=["id", "t", "a", "l", "v", "y", "cens"])


def test_censoring_weights_remove_informative_dropout_bias():
    d = _dgp(6000, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.msm(d, censor="cens", **KW)
        cc = sp.msm(d[d["cens"] == 0], **KW)
    assert abs(r.estimate - 1.0) < 3 * r.se
    # the complete-case fit on the same data is biased by the dropout
    assert abs(cc.estimate - 1.0) > 2.5 * cc.se


def test_uninformative_censoring_model_gives_unit_weights(data):
    """With no time-varying covariate in the denominator the two censoring
    models coincide, every ratio is 1, and the fit is the complete-case one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.msm(data, censor="cens", censor_covariates=[], **KW)
        r0 = sp.msm(data[data["cens"] == 0], **KW)
    np.testing.assert_allclose(r.detail["estimate"], r0.detail["estimate"], rtol=1e-12)
    np.testing.assert_allclose(r.detail["se"], r0.detail["se"], rtol=1e-12)


def test_censoring_inputs_fail_loudly(data):
    late = pd.concat([data, data[data["cens"] == 1].assign(t=99, cens=0)])
    with pytest.raises(MethodIncompatibility, match="after their censoring"):
        sp.msm(late, censor="cens", **KW)
    with pytest.raises(MethodIncompatibility, match="needs censor="):
        sp.msm(data, censor_covariates=["l"], **KW)
    with pytest.raises(DataInsufficient, match="no row is censored"):
        sp.msm(data[data["cens"] == 0], censor="cens", **KW)
    holes = data.copy()
    holes.loc[holes["cens"] == 1, "l"] = np.nan
    with pytest.raises(DataInsufficient, match="missing"):
        sp.msm(holes, censor="cens", **KW)
