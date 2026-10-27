"""sp.choo_siow against cupid_matching 1.3 (Galichon and Salanie's package).

Fixture: ``_fixtures/choo_siow_cupid.json``, written by
``_generate_choo_siow_cupid.py`` in a scratch environment (cupid is MIT
licensed but pins numpy < 2, so it is not a dependency). The table is
cupid's own seeded simulation from a Choo-Siow population with a
quadratic surplus (X = 5, Y = 4, K = 6, 200,000 households).

Rows
----
* ``poisson_glm_newton`` -- cupid's ``choo_siow_poisson_glm`` with
  scikit-learn's solver switched to Newton-Cholesky (``tol=1e-14``):
  beta and its sandwich covariance. Strict parity, 1e-9 relative (observed
  9e-12 for beta, 1.2e-11 for the covariance). cupid's covariance uses the
  multinomial ``Var(mu_hat)``; ours uses the Poisson one. The equality of
  the beta block is the identity proved in ``poisson_glm_fit``'s docstring,
  and this row checks it numerically.
* ``poisson_glm`` -- cupid as shipped (scikit-learn L-BFGS). Its beta solves
  the moment-matching conditions only to 5e-6 of the number of couples,
  ours to 1e-15, so the gap (3e-5 absolute, 1.6e-4 relative) is the
  reference optimiser's stopping rule; the row bounds it at 1e-4 absolute
  and checks that ours is the better root.
* ``ipfp`` -- the equilibrium for the true surplus at new margins
  (``ipfp_homoskedastic_solver(tol=1e-14)``), 1e-10 relative (observed 5e-13).
* ``nonparametric_phi`` -- the formula on the same table, 1e-12 absolute
  (it is the same arithmetic).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.structural._choo_siow_core import ipfp_equilibrium

FIX = Path(__file__).parent / "_fixtures" / "choo_siow_cupid.json"


@pytest.fixture(scope="module")
def ref():
    return json.loads(FIX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fit(ref):
    i = ref["inputs"]
    return sp.choo_siow(
        mu=np.array(i["muxy"], float),
        singles_men=i["mux0"],
        singles_women=i["mu0y"],
        basis=np.array(i["basis"]),
    )


def test_beta_and_covariance_match_cupid_newton(ref, fit):
    r = ref["poisson_glm_newton"]
    np.testing.assert_allclose(fit.params.to_numpy(), r["beta"], rtol=1e-9)
    np.testing.assert_allclose(fit.vcov_params.to_numpy(), r["varcov_beta"], rtol=1e-9)
    np.testing.assert_allclose(fit.std_errors.to_numpy(), r["stderrs_beta"], rtol=1e-9)


def test_cupid_lbfgs_row_differs_only_by_its_stopping_rule(ref, fit):
    i = ref["inputs"]
    r = ref["poisson_glm"]
    assert np.max(np.abs(fit.params.to_numpy() - r["beta"])) < 1e-4
    np.testing.assert_allclose(fit.std_errors.to_numpy(), r["stderrs_beta"], rtol=1e-4)
    # first-order conditions at each solution: ours is the root
    mu = np.array(i["muxy"], float)
    B = np.array(i["basis"])
    n = mu.sum(1) + np.array(i["mux0"])
    m = mu.sum(0) + np.array(i["mu0y"])

    def foc(beta):
        eq = ipfp_equilibrium(B @ beta, n, m, tol=1e-15)
        return np.max(np.abs(np.einsum("xyk,xy->k", B, mu - eq["mu"]))) / mu.sum()

    assert foc(fit.params.to_numpy()) < 1e-12
    assert foc(np.array(r["beta"])) > 1e-7


def test_ipfp_matches_cupid(ref):
    i = ref["inputs"]
    eq = ipfp_equilibrium(
        np.array(i["phi_true"]), np.array(i["cf_men"]), np.array(i["cf_women"])
    )
    np.testing.assert_allclose(eq["mu"], ref["ipfp"]["muxy"], rtol=1e-10)
    np.testing.assert_allclose(eq["singles_men"], ref["ipfp"]["mux0"], rtol=1e-10)
    np.testing.assert_allclose(eq["singles_women"], ref["ipfp"]["mu0y"], rtol=1e-10)


def test_counterfactual_method_matches_cupid_ipfp(ref, fit):
    i = ref["inputs"]
    eq = fit.counterfactual(
        new_margins=(i["cf_men"], i["cf_women"]), phi=np.array(i["phi_true"])
    )
    np.testing.assert_allclose(eq.mu.to_numpy(), ref["ipfp"]["muxy"], rtol=1e-10)


def test_nonparametric_phi_matches(ref):
    i = ref["inputs"]
    res = sp.choo_siow(
        mu=np.array(i["muxy"], float), singles_men=i["mux0"], singles_women=i["mu0y"]
    )
    np.testing.assert_allclose(res.phi.to_numpy(), ref["nonparametric_phi"], atol=1e-12)


def test_known_truth_recovered(ref, fit):
    """cupid simulated the table from beta_true; |z| < 3.5 (observed max 2.1)."""
    z = (fit.params.to_numpy() - ref["inputs"]["beta_true"]) / fit.std_errors.to_numpy()
    assert np.all(np.abs(z) < 3.5)
