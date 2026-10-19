"""``sp.dose_response(method='dr')``: the doubly robust curve of Kennedy,
Ma, McHugh and Small (2017).

Known truth is compared with the *smoothed* truth ``sum_i w_i(t)
theta(T_i)`` (the local linear smoother's own target), so smoothing bias
does not count against the estimator; the curve must sit within its own
pointwise bands of it whichever nuisance model is right. R
``npcausal::ctseff`` is a shape screen (its nuisances, density and
smoother differ), not parity.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import statspai as sp
from statspai.dose_response.dr import _local_linear_weights
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


def _dgp(n=1000, seed=3):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = 1 + 0.8 * x1 + 0.3 * x2 + rng.normal(size=n)
    y = 0.8 * t + 0.3 * t**2 + x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, t=t, x1=x1, x2=x2))


def _truth(t):
    return 0.8 * t + 0.3 * t**2


def _quad():
    return make_pipeline(PolynomialFeatures(2), LinearRegression())


@pytest.mark.parametrize(
    "outcome_model,treatment_model",
    [(LinearRegression(), LinearRegression()), (_quad(), DummyRegressor())],
    ids=["density-right", "outcome-right"],
)
def test_curve_covers_the_smoothed_truth_when_one_model_is_right(
    outcome_model, treatment_model
):
    df = _dgp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.dose_response(
            df,
            "y",
            "t",
            ["x1", "x2"],
            method="dr",
            outcome_model=outcome_model,
            treatment_model=treatment_model,
            n_dose_points=9,
        )
    d = r.detail
    h = r.model_info["bandwidth"]
    T = df["t"].to_numpy()
    smoothed = np.array(
        [_local_linear_weights(T, t0, h)[0] @ _truth(T) for t0 in d["dose"]]
    )
    z = (d["response"].to_numpy() - smoothed) / d["se"].to_numpy()
    assert np.max(np.abs(z)) < 3.5
    assert np.mean(np.abs(z)) < 1.6


def test_infinite_bandwidth_is_the_least_squares_line_on_pseudo_outcomes():
    df = _dgp(n=300)
    r = sp.dose_response(
        df,
        "y",
        "t",
        ["x1", "x2"],
        method="dr",
        outcome_model=LinearRegression(),
        treatment_model=LinearRegression(),
        bandwidth=1e6,
        n_dose_points=5,
    )
    xi = r.model_info["pseudo_outcomes"]
    b1, b0 = np.polyfit(df["t"], xi, 1)
    np.testing.assert_allclose(
        r.detail["response"], b0 + b1 * r.detail["dose"], rtol=1e-8
    )


def test_shape_screen_against_npcausal_ctseff():
    d = pd.read_csv(_FIX / "ctseff_data.csv")
    R = json.loads((_FIX / "ctseff_R.json").read_text(encoding="utf-8"))
    res = pd.DataFrame(R["res"])
    bw = pd.DataFrame(R["bw"])
    grid = res["a.vals"].to_numpy()
    inner = (grid > np.quantile(d["a"], 0.05)) & (grid < np.quantile(d["a"], 0.95))
    r = sp.dose_response(
        d,
        "y",
        "a",
        ["x1", "x2"],
        method="dr",
        outcome_model=LinearRegression(),
        treatment_model=LinearRegression(),
        n_folds=1,
        bandwidth=float(bw["bw"][bw["risk"].idxmin()]),
        dose_range=(grid[0], grid[-1]),
        n_dose_points=len(grid),
    )
    ours = r.detail["response"].to_numpy()
    # observed 0.999; largest interior gap 0.21 (two of ctseff's SEs)
    assert np.corrcoef(ours[inner], res["est"].to_numpy()[inner])[0, 1] > 0.99


def test_unknown_method_is_refused():
    with pytest.raises(MethodIncompatibility):
        sp.dose_response(_dgp(n=100), "y", "t", ["x1"], method="kernel")
