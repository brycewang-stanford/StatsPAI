"""``sp.garch`` against Stata 18 ``arch``: higher orders, AR mean, t errors.

Reference numbers: ``_fixtures/garch_extensions_Stata.csv``, written by
``_fixtures/_generate_garch_extensions_Stata.do`` from the simulated series
``_fixtures/garch_extensions.csv`` (``_generate_garch_extensions_data.py``).

Stata's ``arch`` stops at its own convergence tolerance (tightened in the
do-file) and differentiates numerically, so coefficients agree to about
1e-5 and standard errors to about 1e-4, not to rounding. The
log-likelihood is compared two ways: at each program's own optimum, and
ours evaluated at Stata's estimates, which separates "same objective" from
"same optimum".
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.timeseries.garch import garch_loglik

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "garch_extensions.csv")


@pytest.fixture(scope="module")
def stata():
    ref = pd.read_csv(FIX / "garch_extensions_Stata.csv")
    return {m: g.set_index("name")["value"] for m, g in ref.groupby("model")}


def _stata_vectors(ref, k):
    b = np.array([ref[f"b{j}"] for j in range(1, k + 1)])
    se = np.array([ref[f"se{j}"] for j in range(1, k + 1)])
    return b, se


def test_garch21_matches_stata(data, stata):
    # Stata order: mu, arch, garch1, garch2, omega; ours: mu, omega, alpha,
    # beta1, beta2.
    order = [0, 4, 1, 2, 3]
    for model, vce in (("g21", "opg"), ("g21oim", "oim")):
        b, se = _stata_vectors(stata[model], 5)
        fit = sp.garch("r", data=data, p=2, q=1, vce=vce)
        # coefficients: Stata's convergence tolerance
        np.testing.assert_allclose(fit.params.to_numpy(), b[order], rtol=5e-5)
        # standard errors: numerical derivatives on both sides
        np.testing.assert_allclose(fit.std_errors.to_numpy(), se[order], rtol=5e-4)
        # log-likelihood: the objectives agree to 2e-9 relative
        assert fit.log_likelihood == pytest.approx(stata[model]["ll"], rel=1e-8)


def test_same_objective_at_statas_estimates(data, stata):
    b, _ = _stata_vectors(stata["g21"], 5)
    ours = garch_loglik(data["r"].to_numpy(), b[[0, 4, 1, 2, 3]], p=2, q=1)
    assert ours == pytest.approx(stata["g21"]["ll"], rel=1e-8)
    # and Stata's point is not above our optimum on our objective
    fit = sp.garch("r", data=data, p=2, q=1)
    assert fit.log_likelihood >= ours - 1e-9


def test_ar1_garch11_matches_stata(data, stata):
    # Stata: mu, ar, arch, garch, omega; ours: mu, ar, omega, alpha, beta
    order = [0, 1, 4, 2, 3]
    b, se = _stata_vectors(stata["ar1"], 5)
    fit = sp.garch("r", data=data, ar=1, vce="opg")
    assert list(fit.params.index) == ["mu", "ar[1]", "omega", "alpha[1]", "beta[1]"]
    np.testing.assert_allclose(fit.params.to_numpy(), b[order], rtol=5e-5)
    np.testing.assert_allclose(fit.std_errors.to_numpy(), se[order], rtol=5e-4)
    assert fit.log_likelihood == pytest.approx(stata["ar1"]["ll"], rel=1e-8)


def test_ar1_garch21_student_t_matches_stata(data, stata):
    # Stata: mu, ar, arch, garch1, garch2, omega, ln(df - 2)
    ref = stata["ar1t"]
    b, se = _stata_vectors(ref, 7)
    fit = sp.garch("r", data=data, p=2, q=1, ar=1, dist="t", vce="opg")
    order = [0, 1, 5, 2, 3, 4]
    np.testing.assert_allclose(fit.params.to_numpy()[:6], b[order], rtol=5e-5)
    np.testing.assert_allclose(fit.std_errors.to_numpy()[:6], se[order], rtol=5e-4)
    assert fit.nu == pytest.approx(ref["df"], rel=5e-5)
    # delta method: se(nu) = (nu - 2) * se(ln(nu - 2))
    assert fit.std_errors["nu"] == pytest.approx((ref["df"] - 2) * se[6], rel=5e-4)
    assert fit.log_likelihood == pytest.approx(ref["ll"], rel=1e-8)
