"""``sp.blp`` vs ``pyblp`` (Conlon & Gortmaker) on identical integration nodes.

Reference: ``_fixtures/blp_pyblp.json`` from ``_generate_blp_pyblp.py``
(pyblp 1.2.0, installed outside the StatsPAI environment) on
``_fixtures/blp_pyblp_data.csv`` (``_generate_blp_pyblp_data.py``: 80 markets,
609 products, random coefficients on x1 and price, endogenous price).

This is a cross-package comparison against a PYTHON reference: pyblp is the
maintained BLP implementation; there is no R / Stata one fed the same nodes
(Stata SSC ``blp`` draws its own market-specific Halton streams and has no
node input, so it could only give a Monte-Carlo-level comparison).

Conventions each number depends on
----------------------------------
* Integration: pyblp gets StatsPAI's own scrambled-Halton nodes
  (``n_draws=100, seed=7``) as ``agent_data`` with weights 1/R, the same
  nodes in every market. ``test_nodes_are_identical`` pins that, so the
  finite-sample objective is the same function on both sides.
* Two-step GMM, step-1 ``W = (Z'Z/N)^{-1}``, step-2 ``W`` = inverse of the
  UNcentred robust moment covariance (pyblp ``center_moments=False``; its
  default demeans), robust sandwich SEs for ``(beta, sigma)`` jointly.
* Objective scale: pyblp reports ``N gbar'W gbar`` (Hansen's J after two
  steps); ``sp.blp().gmm_objective`` is ``N^2 gbar'W gbar``. Compared as
  ``gmm_objective / N``.
* Tolerance 1e-6 rel on beta, sigma, their SEs, the objective and own-price
  elasticities (observed ~1e-12): both sides stop an optimiser (L-BFGS-B
  with the analytic gradient on both; ``optimizer='nelder-mead'``, the
  outer loop before October 2026, reaches ~1e-8); mean utilities 1e-6
  absolute.
* Diversion ratios and markups (two firms per market, multi-product
  Bertrand): 1e-7 / 1e-6 at the estimates, 1e-8 at fixed parameters.
* Elasticities with a random PRICE coefficient are checked at fixed
  parameters (sigma = (1.0, 0.3) on (x1, price), beta = (1, 1, -2)), through
  StatsPAI's own contraction and elasticity routine: 1e-8.
"""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# the package re-exports the function ``blp`` over the module name
_blp = importlib.import_module("statspai.structural.blp")

_FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "blp_pyblp.json").read_text(encoding="utf-8"))
IV = REF["instruments"]


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "blp_pyblp_data.csv")


@pytest.fixture(scope="module")
def fit(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.blp(
            data,
            shares="share",
            prices="price",
            x_linear=["x1"],
            x_random=["x1"],
            instruments=IV,
            n_draws=REF["n_draws"],
            seed=REF["seed"],
            tol_outer=1e-10,
        )


def test_nodes_are_identical():
    """Both sides integrate on the same nodes.

    Not a bitwise check. The nodes are recomputed here (the fixture stores
    what pyblp was fed), and the inverse-normal transform of the Halton
    draws is one-ULP platform-dependent: on the Linux CI runner 16 of the
    100 draws land 4.4e-16 from the macOS values the fixture was generated
    on, which ``assert_array_equal`` turned into a red gate on an identity
    that holds. 1e-12 is four orders below any difference that could move
    the estimates this file pins at 1e-6, and a genuinely different Halton
    stream would be off by ~1e-1, not 1e-16.
    """
    d1 = _blp._halton_sequence(REF["n_draws"], 1, seed=REF["seed"])
    d2 = _blp._halton_sequence(REF["n_draws"], 2, seed=REF["seed"])
    np.testing.assert_allclose(d1[:, 0], REF["est_x1"]["nodes"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        d2, np.array(REF["elast_fixed"]["nodes"]), rtol=0, atol=1e-12
    )


def test_point_estimates(fit):
    ref = REF["est_x1"]
    assert ref["converged"]
    assert _rel(fit.linear_params.to_numpy(), ref["beta"]) < 1e-6
    assert _rel(fit.nonlinear_params.to_numpy(), [ref["sigma"]]) < 1e-6
    assert np.max(np.abs(fit.mean_utility.to_numpy() - ref["delta"])) < 1e-6
    assert _rel(fit.gmm_objective / ref["N"], ref["objective"]) < 1e-6


def test_standard_errors(fit):
    """Regression: linear SEs were sqrt(N) too small and ignored sigma's
    estimation error; sigma's SE was N times too small."""
    ref = REF["est_x1"]
    assert _rel(fit.se_linear.to_numpy(), ref["beta_se"]) < 1e-6
    assert _rel(fit.se_nonlinear.to_numpy(), [ref["sigma_se"]]) < 1e-6


def test_own_price_elasticities(fit, data):
    ref = REF["est_x1"]
    own = []
    for m in np.unique(data["market_id"]):
        own.extend(np.diag(fit.elasticity_matrix(m).to_numpy()).tolist())
    assert _rel(own, ref["own_elasticities"]) < 1e-6
    assert _rel(fit.elasticity_matrix(0).to_numpy(), ref["elasticities_market0"]) < 1e-6


def test_elasticities_with_random_price_coefficient(data):
    ref = REF["elast_fixed"]
    draws = np.array(ref["nodes"])
    sigma = np.array(ref["sigma"])
    b0, b1, alpha = ref["beta"]
    s = data["share"].to_numpy()
    mk = data["market_id"].to_numpy()
    p = data["price"].to_numpy()
    Xr = np.column_stack([data["x1"].to_numpy(), p])
    mu = _blp._compute_mu(Xr, sigma, draws)
    d0 = np.zeros(len(s))
    for m in np.unique(mk):
        i = mk == m
        d0[i] = np.log(s[i]) - np.log(1.0 - s[i].sum())
    delta, ok = _blp._contraction_mapping(s, d0, mu, mk, 1e-14, 5000)
    assert ok
    assert np.max(np.abs(delta - np.array(ref["delta"]))) < 1e-10
    E = _blp._compute_elasticities(
        delta, mu, p, alpha, sigma[1], draws, mk, price_draw_col=1
    )
    own = np.concatenate([np.diag(E[m]) for m in np.unique(mk)])
    assert _rel(own, ref["own_elasticities"]) < 1e-8
    assert _rel(E[0], ref["elasticities_market0"]) < 1e-8
    # the pre-fix computation (mean price coefficient for every consumer)
    E_old = _blp._compute_elasticities(delta, mu, p, alpha, sigma[1], draws, mk)
    own_old = np.concatenate([np.diag(E_old[m]) for m in np.unique(mk)])
    assert _rel(own_old, ref["own_elasticities"]) > 1e-2


def test_elasticity_identity_plain_logit(data):
    """sigma = 0: own elasticity alpha * p_j * (1 - s_j), cross alpha * p_k * s_k."""
    mk = data["market_id"].to_numpy()
    p = data["price"].to_numpy()
    s = data["share"].to_numpy()
    d0 = np.zeros(len(s))
    for m in np.unique(mk):
        i = mk == m
        d0[i] = np.log(s[i]) - np.log(1.0 - s[i].sum())
    mu = np.zeros((len(s), 5))
    E = _blp._compute_elasticities(d0, mu, p, -2.0, 0.0, np.zeros((5, 1)), mk)
    i0 = np.where(mk == 0)[0]
    expect = -2.0 * (np.diag(p[i0]) - np.outer(np.ones(len(i0)), p[i0] * s[i0]))
    np.testing.assert_allclose(E[0], expect, rtol=1e-10)


# ---------------------------------------------------------------------------
# Diversion ratios, markups, a random coefficient on the constant
# ---------------------------------------------------------------------------


def _firms(data):
    """Same rule as ``_generate_blp_pyblp.firm_ids``."""
    return (data["product_id"].to_numpy() % 2).astype(int)


def _check_diversion(D, ref_block, tol):
    """pyblp keeps the diversion to the outside good on the diagonal."""
    ref_block = np.array(ref_block)
    inside = ref_block.copy()
    np.fill_diagonal(inside, 0.0)
    assert np.max(np.abs(D.drop(columns="outside").to_numpy() - inside)) < tol
    assert np.max(np.abs(D["outside"].to_numpy() - np.diag(ref_block))) < tol


def test_diversion_ratios(fit):
    """Regression: the method divided elasticities, dropping ``s_k / s_j``."""
    ref = REF["est_x1"]
    for m in (0, 3):
        D = fit.diversion_ratios(m, include_outside=True)
        _check_diversion(D, ref[f"diversion_market{m}"], 1e-7)
        np.testing.assert_allclose(D.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_markups(fit, data):
    ref = REF["est_x1"]
    mk = fit.markups(firm_ids=_firms(data))
    assert _rel(mk.to_numpy(), ref["markups"]) < 1e-6
    mc = fit.marginal_costs(firm_ids=_firms(data))
    np.testing.assert_allclose(
        mc.to_numpy(), data["price"].to_numpy() - mk.to_numpy(), rtol=0, atol=1e-12
    )


def test_diversion_and_markups_with_random_price_coefficient(data):
    """Fixed parameters, random coefficient on price: 1e-8."""
    from statspai.structural._blp_post import diversion_matrix, firm_markups

    ref = REF["elast_fixed"]
    draws = np.array(ref["nodes"])
    sigma = np.array(ref["sigma"])
    alpha = ref["beta"][2]
    mk_ids = data["market_id"].to_numpy()
    s = data["share"].to_numpy()
    Xr = np.column_stack([data["x1"].to_numpy(), data["price"].to_numpy()])
    mu = _blp._compute_mu(Xr, sigma, draws)
    jac = _blp._share_jacobians(
        np.array(ref["delta"]), mu, alpha, sigma[1], draws, mk_ids, price_draw_col=1
    )
    firms = _firms(data)
    markups = np.empty(len(s))
    for m in np.unique(mk_ids):
        i = mk_ids == m
        markups[i] = firm_markups(jac[m], s[i], firms[i])
    assert _rel(markups, ref["markups"]) < 1e-8
    for m in (0, 3):
        D, outside, undefined = diversion_matrix(jac[m])
        assert not undefined.any()
        block = np.array(ref[f"diversion_market{m}"])
        assert np.max(np.abs(outside - np.diag(block))) < 1e-8
        np.fill_diagonal(block, 0.0)
        assert np.max(np.abs(D - block)) < 1e-8


def test_random_coefficient_on_the_constant(data):
    """``x_random=['const', 'x1']`` against pyblp ``X2 = 1 + x1``.

    The standard deviation on the constant is weakly identified here (pyblp
    SE 2.5 on an estimate of 0.13), so the objective is flat in it: the two
    optimisers stop 2e-7 apart on that parameter (1e-9 on the others), and
    the standard errors, which are functions of it, 1.4e-6 apart. Hence
    1e-5 on everything that depends on that parameter, 1e-6 on the rest.
    """
    ref = REF["est_const_x1"]
    assert ref["converged"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_c = sp.blp(
            data,
            shares="share",
            prices="price",
            x_linear=["x1"],
            x_random=["const", "x1"],
            instruments=IV,
            n_draws=REF["n_draws"],
            seed=REF["seed"],
            tol_outer=1e-11,
        )
    assert list(fit_c.nonlinear_params.index) == ["sigma_const", "sigma_x1"]
    sig = fit_c.nonlinear_params.to_numpy()
    assert abs(sig[0] - ref["sigma"][0]) < 1e-5
    assert abs(sig[1] - ref["sigma"][1]) < 1e-6 * ref["sigma"][1]
    assert _rel(fit_c.linear_params.to_numpy(), ref["beta"]) < 1e-6
    assert _rel(fit_c.gmm_objective / REF["est_x1"]["N"], ref["objective"]) < 1e-6
    assert _rel(fit_c.se_linear.to_numpy(), ref["beta_se"]) < 1e-5
    assert _rel(fit_c.se_nonlinear.to_numpy(), ref["sigma_se"]) < 1e-5
    D = fit_c.diversion_ratios(0, include_outside=True)
    _check_diversion(D, ref["diversion_market0"], 1e-5)
    assert _rel(fit_c.markups(firm_ids=_firms(data)).to_numpy(), ref["markups"]) < 1e-5
