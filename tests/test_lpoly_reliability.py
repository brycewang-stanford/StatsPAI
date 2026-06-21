import numpy as np
import pandas as pd
import pytest

from statspai.nonparametric.kdensity import kdensity
from statspai.nonparametric.lpoly import _kernel_fn, _local_poly_fit, lpoly


def test_lpoly_recovers_noiseless_linear_function_at_grid_points():
    x = np.linspace(0.0, 1.0, 21)
    y = 2.0 + 3.0 * x
    grid = np.array([0.2, 0.5, 0.8])
    df = pd.DataFrame({"y": y, "x": x})

    result = lpoly(
        df,
        y="y",
        x="x",
        degree=1,
        bandwidth=0.35,
        kernel="epanechnikov",
        grid=grid,
    )

    np.testing.assert_allclose(result.fitted, 2.0 + 3.0 * grid, atol=1e-12)
    np.testing.assert_allclose(result.se, 0.0, atol=1e-12)


def test_local_poly_fit_uses_kernel_sandwich_se():
    x = np.array([-0.8, -0.4, -0.1, 0.2, 0.6, 0.9])
    y = 1.0 + 0.7 * x + np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.2])
    x0 = 0.0
    h = 1.0
    degree = 1

    fitted, se = _local_poly_fit(x0, x, y, h, degree, "epanechnikov")

    u = (x - x0) / h
    weights = _kernel_fn(u, "epanechnikov") / h
    mask = weights > 0
    x_local = x[mask] - x0
    X = np.column_stack([x_local**j for j in range(degree + 1)])
    w = weights[mask]
    y_local = y[mask]

    XtWX = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(XtWX, X.T @ (w * y_local))
    resid = y_local - X @ beta
    bread = np.linalg.inv(XtWX)
    meat = X.T @ ((w**2 * resid**2)[:, None] * X)
    correction = mask.sum() / (mask.sum() - (degree + 1))
    expected_vcov = correction * bread @ meat @ bread

    old_inverse_variance_formula = np.sqrt(
        (np.sum(w * resid**2) / (mask.sum() - (degree + 1))) * bread[0, 0]
    )

    assert fitted == pytest.approx(beta[0])
    assert se == pytest.approx(np.sqrt(expected_vcov[0, 0]))
    assert se != pytest.approx(old_inverse_variance_formula)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"bandwidth": -1.0}, "bandwidth"),
        ({"degree": -1}, "degree"),
        ({"n_grid": 0}, "n_grid"),
        ({"grid": np.array([[0.0, 1.0]])}, "grid"),
        ({"grid": np.array([0.0, np.nan])}, "grid"),
    ],
)
def test_lpoly_rejects_invalid_smoothing_inputs(kwargs, match):
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 0.5, 1.0]})

    with pytest.raises(ValueError, match=match):
        lpoly(df, y="y", x="x", **kwargs)


def test_kdensity_single_observation_gaussian_matches_closed_form():
    df = pd.DataFrame({"x": [0.0]})

    result = kdensity(
        df,
        x="x",
        bandwidth=1.0,
        kernel="gaussian",
        grid=np.array([0.0]),
    )

    np.testing.assert_allclose(result.density, [1.0 / np.sqrt(2.0 * np.pi)])


def test_kdensity_weighted_gaussian_matches_manual_kernel_sum():
    df = pd.DataFrame({"x": [-1.0, 1.0], "w": [1.0, 3.0]})
    grid = np.array([0.0, 1.0])

    result = kdensity(
        df,
        x="x",
        weights="w",
        bandwidth=0.5,
        kernel="gaussian",
        grid=grid,
    )

    weights = np.array([0.25, 0.75])
    manual = []
    for g in grid:
        u = (g - df["x"].to_numpy()) / 0.5
        manual.append(
            np.sum(weights * np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)) / 0.5
        )
    np.testing.assert_allclose(result.density, manual)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"bandwidth": -1.0}, "bandwidth"),
        ({"n_grid": 0}, "n_grid"),
        ({"grid": np.array([[0.0, 1.0]])}, "grid"),
        ({"grid": np.array([0.0, np.nan])}, "grid"),
        ({"bw_method": "unknown"}, "bw_method"),
        ({"weights": "bad"}, "weights"),
    ],
)
def test_kdensity_rejects_invalid_smoothing_inputs(kwargs, match):
    df = pd.DataFrame({"x": [0.0, 0.5, 1.0], "bad": [1.0, -1.0, 1.0]})

    with pytest.raises(ValueError, match=match):
        kdensity(df, x="x", **kwargs)
