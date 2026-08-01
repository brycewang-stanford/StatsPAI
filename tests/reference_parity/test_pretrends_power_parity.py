"""Reference parity: ``sp.pretrends_power`` vs Roth's ``pretrends`` package.

``pretrends_power`` cited Roth (2022) but computed the power of the *joint*
Wald pre-test. Roth's own ``pretrends`` R package computes something else:
the power of the pre-test analysts actually run, which is to look at the
event-study plot and object if *any* pre-period coefficient is individually
significant. That is a multivariate-normal rectangle probability, not a
non-central chi-squared tail, and on this fixture the two differ by a factor
of more than two.

Since 1.21.0 the default is the package's convention; ``test="joint"`` keeps
the Wald quantity, which is also always reported as ``power_joint``.

Reference generation (R 4.5.2, pretrends 0.1.0, mvtnorm 1.3-3)::

    pretrends(betahat = beta, sigma = sigma, deltatrue = slope * (tVec + 1),
              tVec = tVec, referencePeriod = -1)$df_power
    slope_for_power(sigma = sigma, targetPower = p, tVec = tVec,
                    referencePeriod = -1)

``pretrends`` is not on CRAN; install with
``remotes::install_github("jonathandroth/pretrends")``.

The inputs are literals rather than a fixture file: the estimator is a
function of ``(betahat, sigma)`` alone, and both sides build the same
covariance from the same standard errors and an AR(1) correlation.

.. note::
   Tolerances here are looser than elsewhere in this directory and that is
   deliberate. ``pretrends`` integrates through ``mvtnorm::pmvnorm``, whose
   Genz-Bretz algorithm is randomised: twenty repeated R calls on this
   fixture spread over ~5e-4 (sd 1.3e-4). Pinning tighter would be pinning
   R's Monte-Carlo noise. The likelihood ratio is the exception -- closed
   form on both sides, and pinned at 1e-10.

References
----------
Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing
for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.
[@roth2022pretest]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult

T_VEC = np.array([-4.0, -3.0, -2.0, 0.0, 1.0, 2.0])
SE = np.array([0.050, 0.045, 0.040, 0.100, 0.110, 0.120])
BETA = np.array([0.012, -0.008, 0.021, 0.180, 0.240, 0.310])
RHO = 0.5

# pretrends::pretrends and pretrends::slope_for_power, referencePeriod = -1.
R_POWER = {0.02: 0.331636386010325, 0.05: 0.901493717618343}
R_BAYES = {0.02: 0.767764750944334, 0.05: 0.113163052944783}
R_LR = {0.02: 0.369180669397493, 0.05: 0.00448269494048729}
R_SLOPE = {0.5: 0.027901593818477, 0.8: 0.0424249541097126}

# Monte-Carlo budget: R's own spread on this fixture is ~5e-4.
MC_TOL = 1e-3


def _sigma() -> np.ndarray:
    idx = np.arange(len(SE))
    corr = RHO ** np.abs(idx[:, None] - idx[None, :])
    return corr * np.outer(SE, SE)


@pytest.fixture(scope="module")
def result() -> CausalResult:
    sigma = _sigma()
    pre = T_VEC < -1
    es = pd.DataFrame({"relative_time": T_VEC, "att": BETA, "se": SE})
    return CausalResult(
        method="ReferencePretrendsInput",
        estimand="ATT(0)",
        estimate=float(BETA[3]),
        se=float(SE[3]),
        pvalue=0.0,
        ci=(float(BETA[3] - 1.96 * SE[3]), float(BETA[3] + 1.96 * SE[3])),
        alpha=0.05,
        n_obs=1000,
        model_info={"event_study": es, "vcv_pre": sigma[np.ix_(pre, pre)]},
    )


def _power(result: CausalResult, slope: float, **kwargs):
    pre = T_VEC < -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.pretrends_power(result, delta=slope * (T_VEC[pre] + 1.0), **kwargs)


@pytest.mark.parametrize("slope", sorted(R_POWER))
def test_power_matches_pretrends_package(result, slope):
    got = _power(result, slope)["power"]
    assert got == pytest.approx(
        R_POWER[slope], abs=MC_TOL
    ), f"slope={slope}: StatsPAI {got:.8f} vs pretrends {R_POWER[slope]:.8f}"


@pytest.mark.parametrize("slope", sorted(R_BAYES))
def test_bayes_factor_matches_pretrends_package(result, slope):
    assert _power(result, slope)["bayes_factor"] == pytest.approx(
        R_BAYES[slope], abs=MC_TOL
    )


@pytest.mark.parametrize("slope", sorted(R_LR))
def test_likelihood_ratio_matches_pretrends_package(result, slope):
    """Closed form on both sides -- no Monte-Carlo budget needed."""
    assert _power(result, slope)["likelihood_ratio"] == pytest.approx(
        R_LR[slope], rel=1e-10
    )


@pytest.mark.parametrize("target", sorted(R_SLOPE))
def test_slope_for_power_matches_pretrends_package(result, target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.pretrends_slope_for_power(result, target_power=target)
    assert out["slope"] == pytest.approx(R_SLOPE[target], rel=2e-3)
    assert out["achieved_power"] == pytest.approx(target, abs=1e-4)


def test_joint_and_individual_are_different_tests(result):
    """The 1.21.0 default change is not cosmetic: on this fixture the two
    pre-tests differ by more than a factor of two (0.332 vs 0.157).

    The eyeball test comes out ahead here, but that is not a general
    ordering -- it also has the larger size (see the size tests below), so
    the two are not comparable at face value.
    """
    out = _power(result, 0.02)
    assert out["test"] == "individual"
    assert out["power"] > out["power_joint"] * 1.5
    assert out["power_joint"] == pytest.approx(0.157183, abs=1e-4)
    joint = _power(result, 0.02, test="joint")
    assert joint["power"] == pytest.approx(out["power_joint"], rel=1e-12)


def test_power_under_null_is_alpha_for_the_joint_test(result):
    assert _power(result, 0.02, test="joint")["power_under_null"] == pytest.approx(0.05)


def test_power_under_null_exceeds_alpha_for_the_individual_test(result):
    """Three correlated coefficients, each tested at 5% -- the family-wise
    rejection rate under the null is above 5% by construction."""
    out = _power(result, 0.02)
    assert out["power_under_null"] > 0.05
    assert out["power_under_null"] < 0.15


def test_power_is_monotone_in_the_slope(result):
    powers = [_power(result, s)["power"] for s in (0.0, 0.01, 0.02, 0.05, 0.10)]
    assert all(b >= a - 1e-6 for a, b in zip(powers, powers[1:]))


def test_unknown_test_name_fails_loudly(result):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="individual"):
        _power(result, 0.02, test="wald")


def test_target_power_below_alpha_fails_loudly(result):
    from statspai.exceptions import MethodIncompatibility

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MethodIncompatibility, match="target_power"):
            sp.pretrends_slope_for_power(result, target_power=0.01, alpha=0.05)
