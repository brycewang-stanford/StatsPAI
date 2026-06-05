"""Guard for the vectorized margins dy/dx computation.

``_compute_dydx`` was rewritten to evaluate the linear predictor for the whole
frame at once (looping only over model terms) instead of a per-row
``data.iloc[i]`` Python loop. These tests pin that it is bit-identical to the
per-row ``_predict_row`` reference (intercept / interaction / single terms) and
that the closed-form dy/dx equals the analytic derivative for a linear model.
"""

import numpy as np
import pandas as pd

from statspai.postestimation.margins import _compute_dydx, _predict_row


def _reference_dydx(params, data, var, eps):
    """The original per-row central-difference computation."""
    n = len(data)
    out = np.zeros(n)
    for i in range(n):
        row = data.iloc[i]
        yp = _predict_row(params, row, var, row[var] + eps)
        ym = _predict_row(params, row, var, row[var] - eps)
        out[i] = (yp - ym) / (2 * eps)
    return out


def test_vectorized_dydx_is_bit_identical_to_per_row():
    rng = np.random.default_rng(0)
    n = 1500
    data = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "z": rng.normal(0, 1, n),
        "w": rng.integers(0, 3, n).astype(float),
    })
    params = {"Intercept": 0.5, "x": 1.2, "z": -0.7, "x:z": 0.3,
              "w": 0.4, "x:w": -0.2}
    eps = 1e-4
    for var in ("x", "z", "w"):
        new = _compute_dydx(params, data, var, eps)
        ref = _reference_dydx(params, data, var, eps)
        assert np.array_equal(new, ref), var


def test_dydx_recovers_analytic_derivative_linear():
    # For y = b0 + bx*x + bz*z + bxz*x*z, dy/dx = bx + bxz*z (per row).
    rng = np.random.default_rng(1)
    n = 800
    data = pd.DataFrame({"x": rng.normal(0, 1, n), "z": rng.normal(0, 1, n)})
    params = {"Intercept": 0.5, "x": 1.2, "z": -0.7, "x:z": 0.3}
    dydx = _compute_dydx(params, data, "x", eps=1e-4)
    analytic = 1.2 + 0.3 * data["z"].values
    assert np.allclose(dydx, analytic, rtol=1e-6, atol=1e-6)


def test_missing_term_factor_contributes_zero():
    # An interaction referencing a column not in the data contributes nothing,
    # exactly as _predict_row's `else: val = 0; break` branch.
    data = pd.DataFrame({"x": np.linspace(-1, 1, 50)})
    params = {"x": 1.0, "x:missing": 5.0}
    new = _compute_dydx(params, data, "x", eps=1e-4)
    ref = _reference_dydx(params, data, "x", eps=1e-4)
    assert np.array_equal(new, ref)
    assert np.allclose(new, 1.0)  # only the x term contributes
