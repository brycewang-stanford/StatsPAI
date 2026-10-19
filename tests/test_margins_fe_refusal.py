"""sp.margins after a non-linear fit with absorbed fixed effects.

The average marginal effect after ``sp.fepois`` depends on the estimated
effects. sp.margins used to report a delta-method standard error holding
them fixed: 0.121 (cluster) on ``fixtures/margins_fe/fep.csv``, where Stata
reports 0.106 after both ``poisson y x i.g, vce(cluster g)`` and
``ppmlhdfe y x, a(g) d vce(cluster g)`` + ``margins, dydx(x)``
(``stata_margins_fe.do``). It now refuses those standard errors, as R
marginaleffects does on a fixest model, and returns the estimates with
``vcov=False``. Fitting the effects as ``C(g)`` terms gives Stata's number.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

DATA = pd.read_csv(Path(__file__).parent / "fixtures" / "margins_fe" / "fep.csv")
# Stata 18: margins, dydx(x) after poisson y x i.g (oim)
STATA_DUMMY_OIM = (0.377157455319247, 0.0869408430351664)


@pytest.fixture(scope="module")
def fe_fit():
    return sp.fepois("y ~ x | g", data=DATA)


def test_standard_errors_are_refused(fe_fit):
    with pytest.raises(MethodIncompatibility, match="vcov=False"):
        sp.margins(fe_fit, data=DATA)


def test_estimates_without_standard_errors(fe_fit):
    m = sp.margins(fe_fit, data=DATA, vcov=False)
    assert list(m.columns) == ["variable", "dy/dx"]
    # Poisson with group effects: the fitted means add up to the outcome
    # within every group, so the AME of x is b_x times the mean of y
    np.testing.assert_allclose(
        m["dy/dx"], [fe_fit.params["x"] * DATA["y"].mean()], rtol=1e-8
    )
    np.testing.assert_allclose(m["dy/dx"], [STATA_DUMMY_OIM[0]], rtol=1e-6)


def test_dummy_variable_fit_matches_stata():
    r = sp.poisson("y ~ x + C(g)", data=DATA)
    m = sp.margins(r, data=DATA, variables=["x"])
    np.testing.assert_allclose(m["dy/dx"], [STATA_DUMMY_OIM[0]], rtol=1e-6)
    np.testing.assert_allclose(m["se"], [STATA_DUMMY_OIM[1]], rtol=1e-6)


def test_predictive_margins_refused_but_linear_contrast_allowed():
    ols = sp.feols("y ~ x | g", data=DATA)
    with pytest.raises(MethodIncompatibility, match="absorbed fixed effects"):
        sp.margins_at(ols, data=DATA, at={"x": [0, 1]})
    c = sp.margins_at(ols, data=DATA, at={"x": [0, 1]}, contrast="first")
    # a linear model: the contrast is the coefficient, with its own SE
    np.testing.assert_allclose(c["contrast"], [ols.params["x"]], rtol=1e-10)
    np.testing.assert_allclose(c["se"], [ols.std_errors["x"]], rtol=1e-10)


def test_linear_fe_margins_keep_standard_errors():
    ols = sp.feols("y ~ x | g", data=DATA)
    m = sp.margins(ols, data=DATA)
    np.testing.assert_allclose(m["se"], [ols.std_errors["x"]], rtol=1e-10)
