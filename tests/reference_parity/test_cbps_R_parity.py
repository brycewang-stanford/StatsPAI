"""``sp.cbps`` against R ``CBPS`` 0.24 (black box).

Reference: ``_fixtures/cbps_R.json`` written by ``_fixtures/_generate_cbps_R.R``
on ``cbps_sim.csv`` / ``cbps_bound.csv`` (``_generate_cbps_medsens_data.py``)
and ``matching_lalonde.csv``. CBPS is GPL; only its documented arguments and
returned values were used. The conventions these tests pin are derived in
``docs/dev/cleanroom/cbps.md``.

Two kinds of assertion:

* identities evaluated at CBPS's own coefficients (the moment covariance of
  Imai & Ratkovic 2014, eqs. 16-17; the two-step weighting point; the
  weights), which do not depend on any optimiser and hold to 1e-9 or better;
* comparisons of fitted coefficients / effects, limited by CBPS's ``optim``
  and scalar-search stopping rules (tolerances stated where used); for the
  over-identified ATT, where CBPS stops at a non-stationary point, dominance
  on the criterion both fits target.
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

# the module (``statspai.matching.cbps`` the attribute is the function)
cb = importlib.import_module("statspai.matching.cbps")

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "cbps_R.json").read_text(encoding="utf-8"))

SETS = {
    "sim": ("cbps_sim.csv", "y", "d", ["x1", "x2", "x3"]),
    "bound": ("cbps_bound.csv", "y", "d", ["x1", "x2"]),
    "lalonde": (
        "matching_lalonde.csv",
        "re78",
        "treat",
        ["age", "educ", "married", "nodegree", "re74", "re75", "black", "hispan"],
    ),
}
CASES = [(s, e) for s in SETS for e in ("ate", "att")]


def _data(name):
    file, y, t, cov = SETS[name]
    df = pd.read_csv(_FIX / file)
    X = np.column_stack([np.ones(len(df)), df[cov].to_numpy(float)])
    return df, y, t, cov, X, df[t].to_numpy(float)


def _criterion(beta, X, T, att, weight_point):
    g, _ = cb._stacked(np.asarray(beta, float), X, T, att)
    S = cb._moment_covariance(np.asarray(weight_point, float), X, T, att)
    return float(g @ np.linalg.solve(S, g))


def _fit(name, estimand, variant):
    df, y, t, cov, _, _ = _data(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.cbps(
            df,
            y=y,
            treat=t,
            covariates=cov,
            estimand=estimand.upper(),
            variant=variant,
            n_bootstrap=0,
        )


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


@pytest.mark.parametrize("name,est", CASES)
def test_cue_J_is_the_eq16_17_criterion(name, est):
    """CBPS(twostep = FALSE) reports J = g' Sigma(beta)^-1 g at its own
    coefficients and mle.J the same at the glm MLE: the moments and the
    model-implied covariance are those of eqs. 7, 10/11 and 16/17."""
    _, _, _, _, X, T = _data(name)
    att = est == "att"
    ref = R[name][f"{est}_over"]
    b = np.array(ref["cue_coef"])
    assert _criterion(b, X, T, att, b) == pytest.approx(ref["cue_J"], rel=1e-9)
    m = np.array(R[name]["mle"])
    assert _criterion(m, X, T, att, m) == pytest.approx(ref["cue_mle_J"], rel=1e-9)


@pytest.mark.parametrize("name,est", CASES)
def test_two_step_weighting_point_reproduces_mle_J(name, est):
    """Two-step CBPS fixes Sigma at a * beta_MLE, a = argmin over [0.8, 1.1]
    of the CUE criterion along that ray. CBPS's mle.J (its criterion at the
    MLE) is reproduced with our a. CBPS's scalar search stops ~1e-4 short in
    a, which moves mle.J at first order: hence 1e-3 (a 50% error in a would
    move it by several percent)."""
    _, _, _, _, X, T = _data(name)
    att = est == "att"
    ref = R[name][f"{est}_over"]
    a = _fit(name, est, "over").model_info["weighting_point_scale"]
    assert 0.8 <= a <= 1.1
    m = np.array(R[name]["mle"])
    assert _criterion(m, X, T, att, a * m) == pytest.approx(ref["mle_J"], rel=1e-3)
    # its reported J is the same criterion at its own coefficients
    assert _criterion(ref["coef"], X, T, att, a * m) == pytest.approx(
        ref["J"], rel=1e-3
    )


def test_weighting_point_scale_sits_on_the_bound_when_the_ray_minimum_is_below():
    """cbps_bound.csv is built so that the CUE criterion along the MLE ray is
    minimised below 0.8 (ATE); CBPS's mle.J is then the criterion with Sigma
    at 0.8 beta_MLE, and the unconstrained minimiser would not reproduce it."""
    _, _, _, _, X, T = _data("bound")
    m = np.array(R["bound"]["mle"])
    a = _fit("bound", "ate", "over").model_info["weighting_point_scale"]
    assert a == pytest.approx(0.8, abs=1e-6)  # bounded scalar search, xatol 1e-10
    ref = R["bound"]["ate_over"]["mle_J"]
    assert _criterion(m, X, T, False, 0.8 * m) == pytest.approx(ref, rel=1e-3)
    assert _criterion(m, X, T, False, 0.7 * m) != pytest.approx(ref, rel=1e-2)


@pytest.mark.parametrize("name", list(SETS))
def test_ate_over_matches(name):
    """Same criterion, same weighting point: the coefficients agree up to
    the optimiser slack of CBPS (it stops with a relative gradient of about
    1e-6 in its own basis)."""
    res = _fit(name, "ate", "over")
    ref = R[name]["ate_over"]
    np.testing.assert_allclose(
        res.model_info["beta"], ref["coef"], rtol=2e-3, atol=1e-6
    )
    _, _, _, _, X, T = _data(name)
    a = res.model_info["weighting_point_scale"]
    m = np.array(R[name]["mle"])
    ours = _criterion(res.model_info["beta"], X, T, False, a * m)
    theirs = _criterion(ref["coef"], X, T, False, a * m)
    # we are at the minimum of the criterion CBPS also minimises
    assert ours <= theirs * (1 + 1e-12)
    assert ours == pytest.approx(ref["J"], rel=2e-3)


@pytest.mark.parametrize("name,est", CASES)
def test_exact_matches(name, est):
    """Just-identified: the unique root of the balance equations. We solve
    them to rounding; CBPS's optim leaves a balance residual of 2e-5 to
    3e-4 (in units of the standardised covariates), which moves the
    coefficients by up to 3.2e-3 relative on lalonde (ATT), so 5e-3."""
    res = _fit(name, est, "exact")
    ref = R[name][f"{est}_exact"]
    np.testing.assert_allclose(
        res.model_info["beta"], ref["coef"], rtol=5e-3, atol=1e-6
    )
    _, _, _, _, X, T = _data(name)
    std = cb._Standardiser(X)
    to_std = np.linalg.lstsq(X, std.Z, rcond=None)[0].T  # Z'w = to_std X'w

    def worst_balance(beta):
        g, _ = cb._balance_moments(np.asarray(beta, float), X, T, est == "att")
        return float(np.max(np.abs(to_std @ g)))

    ours, theirs = worst_balance(res.model_info["beta"]), worst_balance(ref["coef"])
    assert ours < 1e-12 and ours < theirs
    smd = res.model_info["std_mean_diff_after"]
    assert max(abs(v) for v in smd.values()) < 1e-8


@pytest.mark.parametrize("name", list(SETS))
def test_att_over_attains_a_lower_criterion(name):
    """Over-identified ATT: CBPS stops where the two-step criterion is not
    stationary; ours is at least as low on the same criterion (same Sigma)."""
    res = _fit(name, "att", "over")
    _, _, _, _, X, T = _data(name)
    a = res.model_info["weighting_point_scale"]
    m = np.array(R[name]["mle"])
    ours = _criterion(res.model_info["beta"], X, T, True, a * m)
    theirs = _criterion(R[name]["att_over"]["coef"], X, T, True, a * m)
    assert ours <= theirs * (1 + 1e-12)
    # gmm_objective uses StatsPAI's own logit MLE for the weighting point;
    # this test uses glm's (converged to ~1e-8 in the deviance), hence 1e-6.
    assert res.model_info["gmm_objective"] == pytest.approx(ours, rel=1e-6)


@pytest.mark.parametrize("name,est", CASES)
@pytest.mark.parametrize("variant", ["over", "exact"])
def test_weights_are_the_hajek_weights(name, est, variant):
    """CBPS's fitted values are expit(X coef); its standardize = FALSE weights
    are T/pi + (1-T)/(1-pi) (ATE) or (n/n1)(T-pi)/(1-pi) (ATT) in absolute
    value, and standardize = TRUE normalises them within each arm. sp.cbps's
    arm weights at CBPS's scores are exactly the standardized weights."""
    _, _, _, _, X, T = _data(name)
    ref = R[name][f"{est}_{variant}"]
    pi = np.array(ref["fitted"])
    np.testing.assert_allclose(
        pi, 1 / (1 + np.exp(-X @ np.array(ref["coef"]))), rtol=1e-10
    )
    w1, w0 = cb._arm_weights(T, pi, est == "att")
    ours = np.where(T == 1, w1 / w1.sum(), w0 / w0.sum())
    np.testing.assert_allclose(ours, ref["weights_std"], rtol=1e-9, atol=1e-15)
