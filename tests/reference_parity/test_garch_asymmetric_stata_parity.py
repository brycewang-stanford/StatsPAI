"""``sp.garch(model="gjr" | "egarch")`` against Stata 18 ``arch``.

Reference: ``_fixtures/garch_asymmetric_Stata.csv`` from
``_fixtures/_generate_garch_asymmetric_Stata.do`` on the simulated
``_fixtures/garch_asymmetric.csv`` (a threshold GARCH with a leverage
effect and t innovations).

Parameterisations. Stata's ``tarch`` multiplies the squared *positive*
shock; ``sp`` writes the threshold term on the squared *negative* shock, as
Glosten, Jagannathan and Runkle do, so ``gamma = -tarch`` and ``alpha =
arch + tarch``. The standard error of ``gamma`` is that of ``tarch``; that
of ``alpha`` is a different linear combination and is not compared. For
EGARCH ``theta = earch`` (signed shock) and ``gamma = earch_a`` (its
magnitude).

Tolerances are those of two optimisers stopping independently on a
likelihood that is flat in some directions: coefficients 2e-4 absolute
(two hundredths of a standard error or less), OPG standard errors 2e-3
relative, log-likelihood 1e-8 relative. The pre-sample conventions were
identified by evaluating our likelihood at Stata's estimates
(``test_same_objective_at_statas_estimates``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.timeseries.garch import _general_filter

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "garch_asymmetric.csv")


@pytest.fixture(scope="module")
def stata():
    ref = pd.read_csv(FIX / "garch_asymmetric_Stata.csv")
    return {m: g.set_index("name")["value"] for m, g in ref.groupby("model")}


def _vec(ref, prefix, k):
    return np.array([ref[f"{prefix}{j}"] for j in range(1, k + 1)])


# model -> (sp arguments, number of Stata coefficients, has AR, has t)
GJR = {
    "gjr": (dict(), 5, False, False),
    "gjrt": (dict(dist="t"), 6, False, True),
    "gjrar": (dict(ar=1), 6, True, False),
}


@pytest.mark.parametrize("name", list(GJR))
def test_threshold_garch(data, stata, name):
    kwargs, k, has_ar, has_t = GJR[name]
    ref = stata[name]
    b, se = _vec(ref, "b", k), _vec(ref, "se", k)
    o = 1 + int(has_ar)  # Stata: mu, [ar], arch, tarch, garch, omega, [lndfm2]
    fit = sp.garch("r", data=data, model="gjr", vce="opg", **kwargs)
    assert fit.log_likelihood == pytest.approx(ref["ll"], rel=1e-8)
    assert fit.mu == pytest.approx(b[0], abs=2e-4)
    assert fit.gamma[0] == pytest.approx(-b[o + 1], abs=2e-4)
    assert fit.alpha[0] == pytest.approx(b[o] + b[o + 1], abs=2e-4)
    assert fit.beta[0] == pytest.approx(b[o + 2], abs=2e-4)
    assert fit.omega == pytest.approx(b[o + 3], abs=2e-4)
    assert fit.std_errors["gamma[1]"] == pytest.approx(se[o + 1], rel=2e-3)
    assert fit.std_errors["beta[1]"] == pytest.approx(se[o + 2], rel=2e-3)
    assert fit.std_errors["omega"] == pytest.approx(se[o + 3], rel=2e-3)
    if has_ar:
        assert fit.ar[0] == pytest.approx(b[1], abs=2e-4)
    if has_t:
        assert fit.nu == pytest.approx(2 + np.exp(b[-1]), rel=2e-3)


@pytest.mark.parametrize("name,dist", [("eg", "normal"), ("egt", "t")])
def test_egarch(data, stata, name, dist):
    ref = stata[name]
    k = 6 if dist == "t" else 5
    b, se = _vec(ref, "b", k), _vec(ref, "se", k)
    # Stata: mu, earch, earch_a, egarch, omega, [lndfm2]
    fit = sp.garch("r", data=data, model="egarch", dist=dist, vce="opg")
    assert list(fit.params.index[:5]) == [
        "mu",
        "omega",
        "theta[1]",
        "gamma[1]",
        "beta[1]",
    ]
    assert fit.log_likelihood == pytest.approx(ref["ll"], rel=1e-8)
    got = [fit.mu, fit.theta[0], fit.gamma[0], fit.beta[0], fit.omega]
    np.testing.assert_allclose(got, b[:5], atol=2e-4)
    ours = fit.std_errors[["mu", "theta[1]", "gamma[1]", "beta[1]", "omega"]]
    np.testing.assert_allclose(ours.to_numpy(), se[:5], rtol=2e-3)
    if dist == "t":
        assert fit.nu == pytest.approx(2 + np.exp(b[5]), rel=2e-3)


def test_egarch_with_two_shock_lags(data, stata):
    ref = stata["eg2"]
    b = _vec(ref, "b", 7)  # mu, earch1, earch2, earch_a1, earch_a2, egarch, omega
    fit = sp.garch("r", data=data, model="egarch", q=2, vce="opg")
    assert fit.log_likelihood == pytest.approx(ref["ll"], rel=1e-8)
    np.testing.assert_allclose(fit.theta, b[1:3], atol=2e-4)
    np.testing.assert_allclose(fit.gamma, b[3:5], atol=2e-4)
    assert fit.beta[0] == pytest.approx(b[5], abs=2e-4)


def test_same_objective_at_statas_estimates(data, stata):
    y = data["r"].to_numpy()
    # threshold model: ours is (mu, omega, alpha, gamma, beta)
    b = _vec(stata["gjr"], "b", 5)
    theta = np.array([b[0], b[4], b[1] + b[2], -b[2], b[3]])
    ll = _general_filter(theta, y, 1, 1, True, 0, "normal", "stata", "gjr")[3].sum()
    assert ll == pytest.approx(stata["gjr"]["ll"], rel=1e-9)
    # EGARCH: ours is (mu, omega, theta, gamma, beta); the centring of |z|
    # is sqrt(2 / pi) under t innovations too, as in Stata
    for name, dist in (("eg", "normal"), ("egt", "t")):
        b = _vec(stata[name], "b", 6 if dist == "t" else 5)
        theta = [b[0], b[4], b[1], b[2], b[3]]
        if dist == "t":
            theta.append(2 + np.exp(b[5]))
        ll = _general_filter(
            np.array(theta), y, 1, 1, True, 0, dist, "stata", "egarch"
        )[3].sum()
        assert ll == pytest.approx(stata[name]["ll"], rel=1e-9)
