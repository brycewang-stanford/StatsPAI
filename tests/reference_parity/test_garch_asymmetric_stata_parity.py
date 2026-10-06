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


# ---------------------------------------------------------------------------
# Variance in the mean (Stata archm)
# ---------------------------------------------------------------------------

# model -> (sp arguments, Stata positions of [mu, archm, then the rest])
IN_MEAN = {
    # Stata: mu, sigma2, arch, garch, omega
    "mg": (dict(), ["mu", "archm", "alpha[1]", "beta[1]", "omega"]),
    # Stata: mu, sigma2, ar, arch, garch, omega
    "mar": (dict(ar=1), ["mu", "archm", "ar[1]", "alpha[1]", "beta[1]", "omega"]),
    # Stata: mu, sigma2, earch, earch_a, egarch, omega
    "meg": (
        dict(model="egarch"),
        ["mu", "archm", "theta[1]", "gamma[1]", "beta[1]", "omega"],
    ),
}


@pytest.mark.parametrize("name", list(IN_MEAN))
def test_variance_in_mean(data, stata, name):
    kwargs, order = IN_MEAN[name]
    ref = stata[name]
    b, se = _vec(ref, "b", len(order)), _vec(ref, "se", len(order))
    fit = sp.garch("r", data=data, in_mean=True, vce="opg", **kwargs)
    assert fit.log_likelihood == pytest.approx(ref["ll"], rel=1e-8)
    # the mean and the in-mean coefficient are nearly collinear, so the
    # likelihood is flat along them: 1e-4 is still 0.003 standard errors
    np.testing.assert_allclose(fit.params[order].to_numpy(), b, atol=1e-4)
    np.testing.assert_allclose(fit.std_errors[order].to_numpy(), se, rtol=2e-3)
    assert fit.archm == pytest.approx(b[1], abs=1e-4)


def test_variance_in_mean_with_threshold_and_with_t(data, stata):
    # Stata: mu, sigma2, arch, tarch, garch, omega
    b = _vec(stata["mgjr"], "b", 6)
    fit = sp.garch("r", data=data, model="gjr", in_mean=True, vce="opg")
    assert fit.log_likelihood == pytest.approx(stata["mgjr"]["ll"], rel=1e-8)
    assert fit.archm == pytest.approx(b[1], abs=1e-4)
    assert fit.gamma[0] == pytest.approx(-b[3], abs=2e-4)
    assert fit.alpha[0] == pytest.approx(b[2] + b[3], abs=2e-4)
    # Stata: mu, sigma2, arch, garch, omega, ln(df - 2)
    b = _vec(stata["mt"], "b", 6)
    fit = sp.garch("r", data=data, dist="t", in_mean=True, vce="opg")
    assert fit.log_likelihood == pytest.approx(stata["mt"]["ll"], rel=1e-8)
    assert fit.archm == pytest.approx(b[1], abs=1e-4)
    assert fit.nu == pytest.approx(2 + np.exp(b[5]), rel=2e-3)


def test_presample_value_is_held_fixed_as_in_stata(data, stata):
    """Stata's pre-sample variance is the mean squared innovation at the
    estimates but is a constant while the likelihood is climbed. Treating
    it as a function of the parameters gives a slightly higher value of
    the same expression at a different point (0.008 standard errors away
    here); the test documents that this is the reason and not noise."""
    y = data["r"].to_numpy()
    b = _vec(stata["mg"], "b", 5)
    at_stata = np.array([b[0], b[1], b[4], b[2], b[3]])  # mu, psi, omega, a, b

    def loglik(theta, m_fixed=None):
        out = _general_filter(
            theta, y, 1, 1, True, 0, "normal", "stata", "garch", 0, True, m_fixed
        )
        return float(out[3].sum()), float(np.mean(out[1] ** 2))

    ll, m = loglik(at_stata)
    assert ll == pytest.approx(stata["mg"]["ll"], rel=1e-9)
    fit = sp.garch("r", data=data, in_mean=True)
    ours = fit.params[["mu", "archm", "omega", "alpha[1]", "beta[1]"]].to_numpy()
    # with the pre-sample value frozen, Stata's point is the maximum ...
    assert loglik(ours, m)[0] == pytest.approx(loglik(at_stata, m)[0], abs=1e-6)
    # ... while letting it move with the parameters there is a higher point
    from scipy.optimize import minimize

    free = minimize(
        lambda th: -loglik(th)[0], ours, method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 5000},
    )  # fmt: skip
    assert -free.fun > ll + 5e-5
    assert abs(free.x[0] - ours[0]) > 1e-4


def test_threshold_at_fewer_lags_than_arch(data, stata):
    # Stata: mu, arch1, arch2, tarch1, garch, omega
    ref = stata["thr"]
    b = _vec(ref, "b", 6)
    y = data["r"].to_numpy()
    # same objective: ours is (mu, omega, alpha1, alpha2, gamma1, beta)
    theta = np.array([b[0], b[5], b[1] + b[3], b[2], -b[3], b[4]])
    ll = _general_filter(theta, y, 1, 2, True, 0, "normal", "stata", "gjr", 1)[3]
    assert ll.sum() == pytest.approx(ref["ll"], rel=1e-9)
    # Stata's estimate has arch1 + tarch1 < 0: the variance would fall
    # after a positive shock. sp.garch keeps alpha >= 0, stops on that
    # boundary, says so, and fits slightly less well.
    assert b[1] + b[3] < 0
    with pytest.warns(RuntimeWarning, match="boundary"):
        fit = sp.garch("r", data=data, model="gjr", q=2, threshold=1)
    assert fit.alpha[0] == pytest.approx(0.0, abs=1e-8)
    assert ref["ll"] - 0.5 < fit.log_likelihood < ref["ll"]
    assert fit.gamma.shape == (1,)
