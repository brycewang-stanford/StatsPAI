"""``sp.series`` against least squares in R (``lm``, ``splines::bs``, ``sandwich``).

A series regression is least squares on a basis, and the fitted function
and its standard error do not depend on how the basis is written. The
reference therefore fits ``lm(y ~ z + poly(u, k, raw = TRUE))`` and
``lm(y ~ z + bs(u, knots = ..., degree = 3))`` in R 4.5.2 on
``_fixtures/series_data.csv`` and takes the variances from ``sandwich``
3.1.1 (``_generate_series_R.R`` -> ``series_R.json``). ``sp.series`` fits
powers and truncated-power pieces of the standardised variable through a
QR decomposition, a different set of columns for the same space.

Compared:

* the leave-one-out cross-validation criterion for polynomial degrees 1 to
  8 and for 0 to 6 quantile knots, and the order it selects;
* at degree 5 and at 3 knots, on 11 grid points: the fitted function and
  its standard error under the classical, HC1, HC2, HC3 and clustered
  variances; the coefficient of the linear regressor and its standard
  errors; the knot positions;
* for the polynomial, the first derivative and its HC1 standard error.

Tolerance: 1e-8 relative (observed <= 4e-9); the derivative 1e-7, since
``sp.series`` differentiates the basis numerically (observed 3e-9).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "series_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "series_data.csv")
GRID = np.asarray(REF["grid"])
VCES = {
    "ols": {"vce": "ols"},
    "hc1": {"vce": "robust"},
    "hc2": {"vce": "hc2"},
    "hc3": {"vce": "hc3"},
    "cluster": {"cluster": "g"},
}
CASES = {"polynomial": 5, "spline": 3}


def test_fixture_is_r_output_and_the_function_is_not_flat():
    assert REF["versions"]["sandwich"] == "3.1.1"
    fit = np.asarray(REF["polynomial"]["fit"])
    assert fit.max() - fit.min() > 1.5  # sin(2x) over (0, 3)
    # the variance estimators give different standard errors
    assert not np.allclose(
        REF["polynomial"]["se_ols"], REF["polynomial"]["se_hc1"], rtol=0.02
    )


def test_cross_validation_criterion_and_choice_for_polynomials():
    fit = sp.series("y ~ z", DATA, "x")
    table = fit.model_info["cv"]
    assert list(table["order"]) == list(range(1, 9))
    np.testing.assert_allclose(table["cv"], REF["poly_cv"], rtol=1e-8)
    assert fit.model_info["order"] == REF["poly_cv_order"] == 5


def test_cross_validation_criterion_and_choice_for_splines():
    fit = sp.series("y ~ z", DATA, "x", basis="spline", orders=range(0, 7))
    np.testing.assert_allclose(fit.model_info["cv"]["cv"], REF["spline_cv"], rtol=1e-8)
    assert fit.model_info["order"] == REF["spline_cv_order"] == 3


@pytest.mark.parametrize("basis", sorted(CASES))
@pytest.mark.parametrize("vce", sorted(VCES))
def test_function_and_standard_errors_match_r(basis, vce):
    ref = REF[basis]
    fit = sp.series(
        "y ~ z", DATA, "x", basis=basis, order=CASES[basis], grid=GRID, **VCES[vce]
    )
    curve = fit.model_info["function"]
    np.testing.assert_allclose(curve["x"], GRID, rtol=1e-12)
    np.testing.assert_allclose(curve["fit"], ref["fit"], rtol=1e-8)
    np.testing.assert_allclose(curve["se"], ref[f"se_{vce}"], rtol=1e-8)
    assert fit.params["z"] == pytest.approx(ref["z_coef"], rel=1e-8)
    assert fit.std_errors["z"] == pytest.approx(ref[f"z_se_{vce}"], rel=1e-8)


def test_polynomial_derivative_matches_the_analytic_one():
    ref = REF["polynomial"]
    fit = sp.series("y ~ z", DATA, "x", order=5, grid=GRID)
    curve = fit.model_info["function"]
    np.testing.assert_allclose(curve["derivative"], ref["derivative"], rtol=1e-7)
    np.testing.assert_allclose(
        curve["derivative_se"], ref["derivative_se_hc1"], rtol=1e-7
    )


def test_spline_knots_are_the_quantiles_r_uses():
    fit = sp.series("y ~ z", DATA, "x", basis="spline", order=3)
    np.testing.assert_allclose(
        fit.model_info["knots"], REF["spline"]["knots_x"], rtol=1e-12
    )
