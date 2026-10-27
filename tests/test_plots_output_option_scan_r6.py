"""Option scan round 6 (2026-10-27): ``sp.binscatter`` (classic) and
``sp.mean_comparison``.

``sp.binscatter`` reference: Stata 18, ``binscatter`` (Stepner, SSC) with
``nquantiles(10) savedata()``, on the data built below (written to CSV with
pandas' round-trip float format and read with ``set type double``). The
saved points carry 8 significant digits.

Defects pinned here:

- ``absorb=`` put the points on the means of the within-transformed
  variables, which are zero: the plot sat at (0, 0) instead of at the
  sample means of ``x`` and ``y`` (Stata: x 7.94 ... 11.46).
- ``weights=`` did not enter the absorption, the means the residuals are
  put back on, the quantile bins or the bin intervals.
- ``absorb=`` with ``controls=`` residualized on controls that had not been
  within-transformed.
- ``sp.mean_comparison(weights=)`` accepted the argument and ignored it.
"""

from __future__ import annotations

from contextlib import nullcontext

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

matplotlib.use("Agg")

# Stata 18 binscatter y x, nquantiles(10) savedata(...)
_STATA_ABSORB = (  # absorb(f)
    [7.9405522, 8.6423186, 9.0710711, 9.3282284, 9.5935329,
     9.8409162, 10.093054, 10.404969, 10.766398, 11.460606],
    [35.384348, 36.875658, 37.555036, 38.306316, 38.64869,
     39.289385, 39.815955, 40.371921, 41.074573, 42.462583],
)  # fmt: skip
_STATA_CONTROLS_AW = (  # [aw=w], controls(f)
    [7.964156, 8.663259, 9.0782873, 9.3384804, 9.5966869,
     9.8804406, 10.13338, 10.433323, 10.773141, 11.501364],
    [35.518154, 36.978881, 37.662289, 38.151953, 38.803132,
     39.385168, 39.832999, 40.449928, 41.150334, 42.492646],
)  # fmt: skip
_STATA_ABSORB_AW = (  # [aw=w], absorb(f)
    [7.9819424, 8.667391, 9.0872504, 9.3473393, 9.6055559,
     9.8757475, 10.130894, 10.436947, 10.766744, 11.469394],
    [35.536988, 37.08756, 37.602145, 38.22547, 38.714545,
     39.385239, 39.856054, 40.487269, 41.029137, 42.509574],
)  # fmt: skip


@pytest.fixture(scope="module")
def bs_data() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 1000
    f = rng.integers(0, 20, n)
    x = 5 + rng.normal(size=n) + 0.5 * f
    y = 10 + 2 * x + f + rng.normal(size=n)
    w = rng.uniform(0.2, 3, n)
    return pd.DataFrame(dict(y=y, x=x, f=f, w=w))


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(absorb=["f"]), _STATA_ABSORB),
        (dict(controls=["f"], weights="w"), _STATA_CONTROLS_AW),
        (dict(absorb=["f"], weights="w"), _STATA_ABSORB_AW),
    ],
    ids=["absorb", "controls-aw", "absorb-aw"],
)
def test_binscatter_matches_stata(bs_data, kwargs, expected):
    import matplotlib.pyplot as plt

    with pytest.warns(Warning) if "controls" in kwargs else nullcontext():
        fig, _, bins = sp.binscatter(bs_data, "y", "x", n_bins=10, **kwargs)
    plt.close(fig)
    # Stata saves 8 significant digits.
    np.testing.assert_allclose(bins["x_mean"], expected[0], rtol=1e-7)
    np.testing.assert_allclose(bins["y_mean"], expected[1], rtol=1e-7)


def test_binscatter_absorb_with_controls_is_one_regression(bs_data):
    # FWL: absorbing f and residualizing on z equals residualizing on z and
    # the f dummies at once.
    import matplotlib.pyplot as plt

    d = bs_data.assign(z=np.sin(bs_data["x"]) + 0.1 * bs_data["f"])
    dummies = pd.get_dummies(d["f"], prefix="f", drop_first=True, dtype=float)
    d2 = pd.concat([d, dummies], axis=1)
    with pytest.warns(Warning):
        fig1, _, a = sp.binscatter(d, "y", "x", controls=["z"], absorb=["f"], n_bins=8)
        fig2, _, b = sp.binscatter(
            d2, "y", "x", controls=["z", *dummies.columns], n_bins=8
        )
    plt.close(fig1)
    plt.close(fig2)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-9)


def test_binscatter_weighted_interval_is_the_weighted_mean_se(bs_data):
    import matplotlib.pyplot as plt

    fig, _, bins = sp.binscatter(bs_data, "y", "x", n_bins=1, weights="w", ci=True)
    plt.close(fig)
    m = sp.regress("y ~ 1", data=bs_data, weights="w")  # Stata regress [aw=]
    se = float(m.std_errors.iloc[0])
    assert bins["y_mean"].iloc[0] == pytest.approx(float(m.params.iloc[0]), rel=1e-12)
    half = bins["ci_upper"].iloc[0] - bins["y_mean"].iloc[0]
    assert half == pytest.approx(stats.norm.ppf(0.975) * se, rel=1e-10)


def test_binscatter_unweighted_plain_points_are_the_raw_bin_means(bs_data):
    import matplotlib.pyplot as plt

    fig, _, bins = sp.binscatter(bs_data, "y", "x", n_bins=10)
    plt.close(fig)
    order = np.argsort(bs_data["x"].to_numpy(), kind="stable")
    ys = bs_data["y"].to_numpy()[order].reshape(10, 100).mean(axis=1)
    np.testing.assert_allclose(bins["y_mean"], ys, rtol=1e-12)


def test_mean_comparison_refuses_weights_it_would_ignore():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        dict(x=rng.normal(size=50), t=np.repeat([0, 1], 25), w=rng.uniform(1, 2, 50))
    )
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.mean_comparison(df, ["x"], "t", weights="w")


def test_mean_comparison_refuses_a_group_with_more_than_two_values():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(dict(x=rng.normal(size=60), t=np.repeat([0, 1, 2], 20)))
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.mean_comparison(df, ["x"], "t")
