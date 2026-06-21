"""Coverage tests for statspai.rd.rd2d (2D / boundary RD).

Exercises both the 'distance' and 'location' approaches, custom boundary
callables, multiple evaluation points, kernels, manual bandwidth, the
rd2d_bw selector, and error paths. Real synthetic boundary RD data.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statspai as sp
from statspai.core.results import CausalResult
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _make_2d(n=2500, tau=2.0, seed=42):
    rng = np.random.default_rng(seed)
    X1 = rng.uniform(-1, 1, n)
    X2 = rng.uniform(-1, 1, n)
    D = (X1 >= 0).astype(float)
    Y = 0.3 * X1 + 0.2 * X2 + tau * D + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": Y, "x1": X1, "x2": X2, "d": D})


def test_rd2d_distance_properties():
    df = _make_2d()
    res = sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="d", approach="distance")
    assert isinstance(res, CausalResult)
    assert res.se > 0
    assert abs(res.estimate - 2.0) < 1.5


def test_rd2d_location_single_and_multi_eval():
    df = _make_2d()
    res1 = sp.rd2d(
        df, y="y", x1="x1", x2="x2", treatment="d", approach="location", n_eval=1
    )
    assert res1.se > 0
    res3 = sp.rd2d(
        df, y="y", x1="x1", x2="x2", treatment="d", approach="location", n_eval=3
    )
    assert res3.detail is not None
    assert len(res3.detail) >= 1


def test_rd2d_custom_boundary_callable():
    rng = np.random.default_rng(1)
    n = 2500
    X1 = rng.uniform(-1, 1, n)
    X2 = rng.uniform(-1, 1, n)
    # boundary: x2 = 0.2 * x1; treated above the line
    D = (X2 >= 0.2 * X1).astype(float)
    Y = 0.2 * X1 + 0.2 * X2 + 1.5 * D + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": Y, "x1": X1, "x2": X2, "d": D})
    res = sp.rd2d(
        df,
        y="y",
        x1="x1",
        x2="x2",
        treatment="d",
        approach="distance",
        boundary=lambda x1: 0.2 * x1,
    )
    assert res.se > 0


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_rd2d_kernels(kernel):
    df = _make_2d()
    res = sp.rd2d(
        df,
        y="y",
        x1="x1",
        x2="x2",
        treatment="d",
        approach="distance",
        kernel=kernel,
        h=0.5,
    )
    assert res.se > 0


def test_rd2d_manual_bandwidth():
    df = _make_2d()
    res = sp.rd2d(
        df, y="y", x1="x1", x2="x2", treatment="d", approach="distance", h=0.4
    )
    assert res.se > 0


def test_rd2d_bw_selector():
    df = _make_2d()
    h = sp.rd2d_bw(df, y="y", x1="x1", x2="x2", treatment="d", approach="distance")
    assert isinstance(h, float)
    assert h > 0
    np.testing.assert_allclose(h, 0.21015049861820304, atol=1e-12)


def test_rd2d_invalid_approach():
    df = _make_2d()
    with pytest.raises(MethodIncompatibility, match="approach"):
        sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="d", approach="bad")


def test_rd2d_input_validation_taxonomy():
    df = _make_2d()

    with pytest.raises(MethodIncompatibility, match="Column"):
        sp.rd2d(df, y="missing", x1="x1", x2="x2", treatment="d")

    with pytest.raises(DataInsufficient, match="Too few valid"):
        sp.rd2d(df.head(10), y="y", x1="x1", x2="x2", treatment="d")

    all_treated = df.assign(d=1.0)
    with pytest.raises(DataInsufficient, match="Too few treated"):
        sp.rd2d(all_treated, y="y", x1="x1", x2="x2", treatment="d")

    with pytest.raises(MethodIncompatibility, match="eval_points"):
        sp.rd2d(
            df,
            y="y",
            x1="x1",
            x2="x2",
            treatment="d",
            approach="location",
            eval_points=np.ones((2, 3)),
        )

    with pytest.raises(MethodIncompatibility, match="Column"):
        sp.rd2d_bw(df, y="missing", x1="x1", x2="x2", treatment="d")


@pytest.mark.parametrize("ptype", ["scatter", "heatmap"])
def test_rd2d_plot_types_vertical_boundary(ptype):
    df = _make_2d()
    res = sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="d", approach="distance")
    fig, ax = sp.rd2d_plot(
        df, y="y", x1="x1", x2="x2", treatment="d", result=res, plot_type=ptype
    )
    assert fig is not None
    plt.close("all")


def test_rd2d_plot_custom_boundary_and_boundary_effects():
    rng = np.random.default_rng(1)
    n = 2000
    X1 = rng.uniform(-1, 1, n)
    X2 = rng.uniform(-1, 1, n)
    D = (X2 >= 0.2 * X1).astype(float)
    Y = 0.2 * X1 + 0.2 * X2 + 1.5 * D + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": Y, "x1": X1, "x2": X2, "d": D})
    bdry = lambda v: 0.2 * v
    res = sp.rd2d(
        df,
        y="y",
        x1="x1",
        x2="x2",
        treatment="d",
        approach="location",
        boundary=bdry,
        n_eval=3,
    )
    fig, ax = sp.rd2d_plot(
        df,
        y="y",
        x1="x1",
        x2="x2",
        treatment="d",
        boundary=bdry,
        result=res,
        plot_type="scatter",
    )
    assert fig is not None
    # boundary_effects panel needs multiple eval points in the result
    fig2, ax2 = sp.rd2d_plot(
        df,
        y="y",
        x1="x1",
        x2="x2",
        treatment="d",
        boundary=bdry,
        result=res,
        plot_type="boundary_effects",
    )
    assert fig2 is not None
    plt.close("all")


def test_rd2d_plot_invalid_type():
    df = _make_2d()
    with pytest.raises(MethodIncompatibility, match="plot_type"):
        sp.rd2d_plot(df, y="y", x1="x1", x2="x2", treatment="d", plot_type="bad")


def test_rd2d_plot_boundary_effects_requires_location_detail():
    df = _make_2d()
    res = sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="d")
    with pytest.raises(MethodIncompatibility, match="boundary eval points"):
        sp.rd2d_plot(
            df,
            y="y",
            x1="x1",
            x2="x2",
            treatment="d",
            result=res,
            plot_type="boundary_effects",
        )
