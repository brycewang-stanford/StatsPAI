"""Joint lead-and-horizon covariance of ``sp.did_imputation`` vs Stata.

With ``pretrends(k)``, Stata ``did_imputation`` (Borusyak, SSC; November 22,
2023) estimates the leads in an auxiliary ``reghdfe`` regression on the
untreated observations and the horizons by imputation, then builds ``e(V)``
by ``matrix accum`` of both blocks' per-cluster scores, so the lead-horizon
cross-covariance is part of it. Through 1.39 StatsPAI set that block to
zero and flagged the matrix block-diagonal, which made ``sp.honest_did`` and
``sp.uniform_bands`` fall back to their no-covariance approximations on BJS
fits.

The reference (``_fixtures/_generate_did_imputation_pretrend_vcov_Stata.do``,
Stata 18) is ``horizons(0/3) pretrends(3) autosample`` on Track A module
05's mpdta bytes. Tolerances, each from the mechanism that limits it:

* lead block: ``rtol=1e-12`` -- closed-form OLS on both sides (observed
  9e-14). This row also pins ``reghdfe``'s singleton rule: the 125 counties
  first treated in 2004 have one untreated year, ``reghdfe`` drops them
  before its dof adjustment, and leaving them in made these variances 0.1%
  too small;
* horizon block: ``rtol=1e-8`` -- ``did_imputation``'s imputation weights,
  tightened to ``tol(1e-12)`` (observed 7e-10);
* cross block: ``rtol=1e-7`` -- the products of the two blocks' scores carry
  ``reghdfe``'s 1e-8 fixed-effect solver tolerance in Stata's residuals
  (observed 2.2e-8).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_HERE = Path(__file__).resolve().parent
_FIXTURE = _HERE / "_fixtures" / "did_imputation_pretrend_vcov_Stata.json"
_MPDTA = _HERE.parent / "r_parity" / "data" / "05_sunab.csv"
_TIME = {"tau0": 0, "tau1": 1, "tau2": 2, "tau3": 3, "pre1": -1, "pre2": -2, "pre3": -3}


@pytest.fixture(scope="module")
def ref():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fit():
    mp = pd.read_csv(_MPDTA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_imputation(
            mp,
            y="lemp",
            group="countyreal",
            time="year",
            first_treat="first_treat",
            horizon=[0, 1, 2, 3],
            pretrends=3,
        )


@pytest.fixture(scope="module")
def aligned(fit, ref):
    es = sp.event_study_vcov(fit)
    pos = [list(es.times).index(_TIME[n]) for n in ref["names"]]
    return es, es.vcov[np.ix_(pos, pos)], np.asarray(ref["V"], dtype=float)


def test_covariance_is_joint(aligned):
    es, _, _ = aligned
    assert es.joint
    assert sorted(es.times.tolist()) == [-3, -2, -1, 0, 1, 2, 3]


def test_lead_block_matches_stata(aligned):
    _, V, R = aligned
    np.testing.assert_allclose(V[4:, 4:], R[4:, 4:], rtol=1e-12, atol=0)


def test_horizon_block_matches_stata(aligned):
    _, V, R = aligned
    np.testing.assert_allclose(V[:4, :4], R[:4, :4], rtol=1e-8, atol=0)


def test_lead_horizon_cross_covariance_matches_stata(aligned):
    _, V, R = aligned
    np.testing.assert_allclose(V[:4, 4:], R[:4, 4:], rtol=1e-7, atol=0)
    np.testing.assert_allclose(V[4:, :4], R[4:, :4], rtol=1e-7, atol=0)


def test_reported_standard_errors_are_the_diagonal(fit, aligned):
    es, _, _ = aligned
    table = fit.model_info["event_study"].set_index("relative_time")
    se = np.sqrt(np.diag(es.vcov))
    np.testing.assert_allclose(
        se, table.loc[es.times.tolist(), "se"].to_numpy(), rtol=1e-12
    )


def test_singletons_do_not_move_the_lead_coefficients(ref, fit):
    """Dropping singletons changes only the dof count, never the estimates
    (a singleton fits its own fixed effect exactly): the leads equal Stata's
    e(b) to reghdfe's solver tolerance."""
    es = fit.model_info["event_study"].set_index("relative_time")
    b_ref = dict(zip(ref["names"], ref["b"]))
    for k in (1, 2, 3):
        assert es.loc[-k, "att"] == pytest.approx(b_ref[f"pre{k}"], rel=1e-6)


def test_honest_did_uses_the_joint_covariance(fit):
    """With the cross block in place the FLCI is computed, not the warned
    worst-case-bias fallback."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sp.honest_did(fit, e=0, m_grid=[0.0, 0.01])
    assert out.attrs["interval"] == "flci"
    assert out.attrs["source"] == "did_imputation"
