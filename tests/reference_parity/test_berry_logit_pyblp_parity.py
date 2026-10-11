"""``sp.blp`` without random coefficients vs ``pyblp``: Berry (1994).

Reference: ``_fixtures/berry_logit_pyblp.json`` from
``_generate_berry_logit_pyblp.py`` (pyblp 1.2.0, installed outside the
StatsPAI environment) on ``_fixtures/berry_logit_data.csv``
(``_generate_berry_logit_data.py``: 120 markets, 1 200 products, exact
nested-logit shares with rho = 0.6, endogenous price).

Both models are linear IV-GMM, so there is no optimiser or integration rule
on our side and the comparison is deterministic.

Conventions each number depends on
----------------------------------
* Instruments: the firm's other products and rival products, for the
  product count and each characteristic (BLP 1995, section 5.1), compared
  column by column with ``pyblp.build_blp_instruments`` (1e-11), plus the
  cost shifter. The nested logit adds ``nest_count``; pyblp's tutorial
  instrument is the count including the product itself, which spans the
  same space given the constant.
* ``gmm_steps=1`` is pyblp ``method="1s"`` (``W = (Z'Z/N)^{-1}``, 2SLS);
  ``gmm_steps=2`` is ``"2s"`` with the uncentred robust moment covariance
  (pyblp ``center_moments=False``). Robust sandwich standard errors.
* Objective scale: pyblp reports ``N gbar'W gbar``; ``gmm_objective`` is
  ``N`` times that.
* pyblp concentrates the linear parameters out and searches over rho
  (L-BFGS-B, projected gradient below 1e-10 in the fixture); we solve the
  same moments in closed form. Tolerance 1e-7 relative, observed ~1e-9.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "berry_logit_pyblp.json").read_text(encoding="utf-8"))
RTOL = 1e-7


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_FIX / "berry_logit_data.csv")
    z = sp.blp_instruments(d, ["x1", "x2"], firm_id="firm", nest="nest")
    return pd.concat([d, z], axis=1)


def _iv(nested):
    names = [f"{k}_{c}" for k in ("own", "rival") for c in ("count", "x1", "x2")]
    return names + ["w"] + (["nest_count"] if nested else [])


def _fit(data, nested, steps):
    return sp.blp(
        data,
        shares="share",
        prices="price",
        x_linear=["x1", "x2"],
        x_random=[],
        instruments=_iv(nested),
        nest="nest" if nested else None,
        gmm_steps=steps,
    )


def test_instruments_are_pyblp_build_blp_instruments(data):
    cols = [f"{k}_{c}" for k in ("own", "rival") for c in ("count", "x1", "x2")]
    np.testing.assert_allclose(
        data[cols].to_numpy(), np.array(REF["blp_instruments"]), atol=1e-11
    )


@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("nested", [False, True])
def test_estimates_standard_errors_objective(data, nested, steps):
    ref = REF[f"{'nested' if nested else 'logit'}_{steps}s"]
    res = _fit(data, nested, steps)
    np.testing.assert_allclose(res.linear_params.values, ref["beta"], rtol=RTOL)
    np.testing.assert_allclose(res.se_linear.values, ref["beta_se"], rtol=RTOL)
    np.testing.assert_allclose(
        res.gmm_objective / REF["N"], ref["objective"], rtol=RTOL
    )
    if nested:
        np.testing.assert_allclose(res.rho, ref["rho"], rtol=RTOL)
        np.testing.assert_allclose(res.se_nonlinear["rho"], ref["rho_se"], rtol=RTOL)
        np.testing.assert_allclose(res.mean_utility.values, ref["delta"], atol=1e-7)


@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("nested", [False, True])
def test_elasticities(data, nested, steps):
    ref = REF[f"{'nested' if nested else 'logit'}_{steps}s"]
    res = _fit(data, nested, steps)
    np.testing.assert_allclose(
        res.own_elasticities.values, ref["own_elasticities"], rtol=RTOL
    )
    np.testing.assert_allclose(
        res.elasticity_matrix(0).values, ref["elasticities_market0"], rtol=RTOL
    )


def test_hansen_j_is_the_two_step_objective(data):
    res = _fit(data, True, 2)
    assert res.hansen_j == pytest.approx(REF["nested_2s"]["objective"], rel=RTOL)
    # 8 excluded instruments, 2 endogenous regressors
    assert res.hansen_df == 6
    assert _fit(data, True, 1).hansen_j is None
