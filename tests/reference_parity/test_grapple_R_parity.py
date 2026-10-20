"""sp.grapple vs GRAPPLE 0.2.2 (black box) on five simulated scenarios.

Reference: ``_fixtures/grapple_R.json`` written by
``_fixtures/_generate_grapple_R.R`` on ``_fixtures/grapple_sim.csv``
(heterogeneous SEs with and without pleiotropy, gross outliers, no exposure
measurement error, weak instruments). Conventions and the derivation of each
tolerance: ``docs/dev/cleanroom/grapple.md``.

* The sandwich, evaluated at the reference's own ``(beta, tau2)`` -- the
  estimated pair and two fixed ``tau2`` values -- reproduces its SEs: 1e-10
  for L2 (closed-form moments on both sides), 1e-6 for Huber / Tukey (R
  integrates the loss moments with ``integrate()``). For Tukey the
  reference's Var[rho(Z)] is on the biweight's max-1 scale (documented
  divergence); the formula is fed that scale to reproduce it.
* StatsPAI's fit solves both estimating equations to machine precision; the
  reference stops ``optim`` at its default tolerance, so fitted values agree
  only to the tolerances stated in ``test_fit_is_aligned``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.mendelian.frontier.grapple import (
    _grapple_vcov,
    _loss_moments,
    _robust_rho,
)

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "grapple_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "grapple_sim.csv")
SCENARIOS = ["hetero_tau0", "hetero_tau", "outliers", "no_xerror", "weak"]
LOSSES = ["tukey", "huber", "l2"]
TUKEY_C = 4.685


def _data(scenario):
    s = D[D["id"] == scenario]
    return tuple(s[c].to_numpy() for c in ("bx", "by", "sx", "sy"))


def _reference_scale_vcov(loss, beta, tau2, bx, by, sx, sy):
    rho = _robust_rho(loss, None)
    mom = _loss_moments(loss, rho, None)
    rho_var = mom["rho_var"]
    if loss == "tukey":
        rho_var = rho_var * (6 / TUKEY_C**2) ** 2
    return _grapple_vcov(
        beta,
        tau2,
        bx,
        by,
        sx**2,
        sy**2,
        rho,
        mom["score_sq"],
        rho_var,
        mom["slope"],
        mom["slope"],
    )


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_sandwich_reproduces_reference_ses(scenario, loss):
    bx, by, sx, sy = _data(scenario)
    ref = R[scenario][loss]
    tol = 1e-10 if loss == "l2" else 1e-6
    V = _reference_scale_vcov(loss, ref["beta"], ref["tau2"], bx, by, sx, sy)
    np.testing.assert_allclose(np.sqrt(V[0, 0]), ref["se"], rtol=tol)
    np.testing.assert_allclose(np.sqrt(V[1, 1]), ref["tau2_se"], rtol=tol)
    for fx in ref["fixed"]:
        V = _reference_scale_vcov(loss, fx["beta"], fx["tau2"], bx, by, sx, sy)
        np.testing.assert_allclose(np.sqrt(V[0, 0]), fx["se"], rtol=tol)


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fit_is_aligned(scenario, loss):
    """Same mode as the reference; the gaps are its optimiser's stopping
    rule. Observed maxima over the 15 fits: beta 3.8e-3 (weak, Huber), tau2
    9e-4, SE 3.7e-4 (L2 / Huber); StatsPAI's point solves the beta score to
    1e-12 relative to its scale, the reference's does not."""
    bx, by, sx, sy = _data(scenario)
    ref = R[scenario][loss]
    g = sp.grapple(bx, by, sx, sy, loss=loss)
    assert g.converged
    np.testing.assert_allclose(g.estimate, ref["beta"], rtol=5e-3)
    if ref["tau2"] == 0:
        assert g.tau2 == 0
    else:
        np.testing.assert_allclose(g.tau2, ref["tau2"], rtol=2e-3)
    if loss != "tukey":
        np.testing.assert_allclose(g.se, ref["se"], rtol=1e-3)
    rho = _robust_rho(loss, None)
    v = sy**2 + g.estimate**2 * sx**2 + g.tau2
    resid = by - g.estimate * bx
    terms = rho(resid / np.sqrt(v), deriv=1) * (v * bx + resid * sx**2 * g.estimate)
    score = np.sum(terms / v**1.5)
    assert abs(score) <= 1e-9 * np.sum(np.abs(terms / v**1.5))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_tukey_tau2_se_is_the_reference_times_c2_over_6(scenario):
    """The documented Tukey divergence is exactly the loss-scale factor."""
    bx, by, sx, sy = _data(scenario)
    g = sp.grapple(bx, by, sx, sy, loss="tukey")
    if g.tau2 == 0:
        assert np.isnan(g.tau2_se)
        return
    rho = _robust_rho("tukey", None)
    mom = _loss_moments("tukey", rho, None)
    V = _grapple_vcov(
        g.estimate,
        g.tau2,
        bx,
        by,
        sx**2,
        sy**2,
        rho,
        mom["score_sq"],
        mom["rho_var"] * (6 / TUKEY_C**2) ** 2,
        mom["slope"],
        mom["slope"],
    )
    np.testing.assert_allclose(g.tau2_se * 6 / TUKEY_C**2, np.sqrt(V[1, 1]), rtol=1e-12)


@pytest.mark.parametrize("loss", LOSSES)
def test_sandwich_is_invariant_to_rescaling_the_loss(loss):
    """Identity: replacing rho by s * rho (all derivatives and moments
    rescaled consistently) leaves V unchanged. The reference's mixed Tukey
    scale violates it, which is why StatsPAI keeps one scale."""
    bx, by, sx, sy = _data("hetero_tau")
    rho = _robust_rho(loss, None)
    mom = _loss_moments(loss, rho, None)
    beta, tau2 = 0.5, 3e-4
    s = 6 / TUKEY_C**2

    def scaled(r, deriv=0):
        return s * rho(r, deriv=deriv)

    args = (beta, tau2, bx, by, sx**2, sy**2)
    V1 = _grapple_vcov(
        *args,
        rho,
        mom["score_sq"],
        mom["rho_var"],
        mom["slope"],
        mom["slope"],
    )
    V2 = _grapple_vcov(
        *args,
        scaled,
        s**2 * mom["score_sq"],
        s**2 * mom["rho_var"],
        s * mom["slope"],
        s * mom["slope"],
    )
    np.testing.assert_allclose(V2, V1, rtol=1e-12)
