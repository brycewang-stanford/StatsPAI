"""Default dispersion of the gamma and inverse-Gaussian GLMs: Stata and R.

Both references estimate the scale of these families by the Pearson
chi-squared over the residual degrees of freedom. They differ in the
information matrix behind the default standard errors: Stata ``glm`` uses
the observed information, R ``summary.glm`` the expected one. ``sp.glm``
follows Stata by default and reproduces R with ``information='expected'``.

Reference numbers, on ``_fixtures/glm_gamma_dispersion.csv`` (n = 40):

Stata 18 ::

    glm y x1 x2, family(gamma) link(log) ///
        ltolerance(1e-14) nrtolerance(1e-14) tolerance(1e-14) iterate(200)
    glm y x1 x2, family(igaussian) link(log) (same tolerances)

R 4.5 ::

    summary(glm(y ~ x1 + x2, family = Gamma(link = "log"), data = d,
                control = glm.control(epsilon = 1e-14, maxit = 200)))
    (and family = inverse.gaussian(link = "log"))

Stata's default tolerances leave its observed-information standard errors
1.5e-5 away from these; the tolerances above are needed for the comparison.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_DATA = pathlib.Path(__file__).parent / "_fixtures" / "glm_gamma_dispersion.csv"

# (family, dispersion, Stata OIM se, R expected-information se), in the
# order intercept, x1, x2.
REFERENCE = {
    "gamma": (
        0.362358835,
        (0.09675037936, 0.10723990903, 0.12848672523),
        (0.09627939968, 0.11169445751, 0.10696259311),
    ),
    "inverse_gaussian": (
        0.360308278,
        (0.13821853406, 0.11876711969, 0.14399847611),
        (0.12234442710, 0.11484077606, 0.10269743934),
    ),
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_DATA, encoding="utf-8")


@pytest.mark.parametrize("family", sorted(REFERENCE))
def test_default_dispersion_and_se_are_stata_glm(data, family):
    phi, se_oim, _ = REFERENCE[family]
    fit = sp.glm("y ~ x1 + x2", data, family=family, link="log")
    # The references print 9 to 11 digits; the inverse-Gaussian fit with a
    # log link converges to about 1e-8 on each side.
    assert fit.diagnostics["Dispersion"] == pytest.approx(phi, rel=1e-7)
    np.testing.assert_allclose(fit.std_errors, se_oim, rtol=1e-6)


@pytest.mark.parametrize("family", sorted(REFERENCE))
def test_expected_information_is_r_summary_glm(data, family):
    phi, _, se_eim = REFERENCE[family]
    fit = sp.glm("y ~ x1 + x2", data, family=family, link="log", information="expected")
    assert fit.diagnostics["Dispersion"] == pytest.approx(phi, rel=1e-7)
    np.testing.assert_allclose(fit.std_errors, se_eim, rtol=1e-6)


def test_scale_dev_gives_the_former_default(data):
    # Through 1.39.3 the gamma family scaled its covariance by deviance / df.
    pearson = sp.glm("y ~ x1 + x2", data, family="gamma", link="log")
    deviance = sp.glm("y ~ x1 + x2", data, family="gamma", link="log", scale="dev")
    ratio = np.sqrt(
        0.4470343985738015 / 0.36235883481431136
    )  # Stata e(dispers) / e(phi)
    np.testing.assert_allclose(
        np.asarray(deviance.std_errors) / np.asarray(pearson.std_errors),
        ratio,
        rtol=1e-7,
    )
    np.testing.assert_allclose(deviance.params, pearson.params, rtol=0, atol=0)
