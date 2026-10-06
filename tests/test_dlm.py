"""API, edge cases and failure modes of ``sp.dlm``.

Numerical correctness is in ``tests/reference_parity/test_dlm_parity.py``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    T = 200
    x = rng.normal(size=T)
    slope = 1.0 + np.cumsum(0.12 * rng.normal(size=T))
    return pd.DataFrame(
        {
            "t": np.arange(T),
            "x": x,
            "slope_true": slope,
            "y": 0.5 + slope * x + 0.3 * rng.normal(size=T),
        }
    )


def test_tracks_a_drifting_slope(df):
    fit = sp.dlm("y ~ x", df)
    assert list(fit.variances.index) == ["obs", "state:Intercept", "state:x"]
    path = fit.smoothed["x"].to_numpy()
    truth = df["slope_true"].to_numpy()
    assert np.corrcoef(path, truth)[0, 1] > 0.9
    inside = (fit.smoothed["x_lower"] < truth) & (truth < fit.smoothed["x_upper"])
    assert inside.mean() > 0.85
    # a fixed-coefficient regression cannot follow it
    ols = sp.regress("y ~ x", df)
    assert np.mean((path - truth) ** 2) < 0.3 * np.mean(
        (float(ols.params["x"]) - truth) ** 2
    )
    # the intercept is constant in the data: its state variance is small
    assert (
        fit.variances.loc["state:Intercept", "estimate"]
        < 0.2 * fit.variances.loc["state:x", "estimate"]
    )
    assert fit.variances.loc["obs", "estimate"] == pytest.approx(0.09, rel=0.5)
    assert "Dynamic linear model" in fit.summary()
    json.dumps(fit.to_dict())
    assert fit.cite().startswith("petris2009dynamic")


def test_filtered_uses_the_past_only(df):
    full = sp.dlm("y ~ x", df, obs_var=0.1, state_var=0.01)
    head = sp.dlm("y ~ x", df.head(120), obs_var=0.1, state_var=0.01)
    assert np.allclose(
        full.filtered["x"].to_numpy()[:120], head.filtered["x"].to_numpy()
    )
    # the smoother uses the future, so it differs before the end
    assert not np.allclose(
        full.smoothed["x"].to_numpy()[:119], head.smoothed["x"].to_numpy()[:119]
    )
    assert full.smoothed["x_sd"].iloc[60] < full.filtered["x_sd"].iloc[60]


def test_time_sorts_the_rows(df):
    shuffled = df.sample(frac=1.0, random_state=1)
    a = sp.dlm("y ~ x", df, obs_var=0.1, state_var=0.01)
    b = sp.dlm("y ~ x", shuffled, time="t", obs_var=0.1, state_var=0.01)
    assert np.allclose(a.smoothed["x"].to_numpy(), b.smoothed["x"].to_numpy())
    assert list(b.smoothed.index) == list(range(len(df)))


def test_forecast(df):
    fit = sp.dlm("y ~ x", df)
    new = pd.DataFrame({"x": [0.0, 1.0, 1.0]})
    fc = fit.forecast(3, new)
    assert list(fc.columns) == ["mean", "lower", "upper"]
    assert fc["mean"].iloc[1] == pytest.approx(
        fit.filtered["Intercept"].iloc[-1] + fit.filtered["x"].iloc[-1]
    )
    # uncertainty grows with the horizon at the same regressor value
    width = (fc["upper"] - fc["lower"]).to_numpy()
    assert width[2] > width[1]
    level = sp.dlm("y ~ 1", df)
    assert level.forecast(4).shape == (4, 3)
    with pytest.raises(sp.MethodIncompatibility, match="Pass data="):
        fit.forecast(2)
    with pytest.raises(sp.MethodIncompatibility, match="rows"):
        fit.forecast(5, new)


def test_gibbs_agrees_with_maximum_likelihood(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = sp.dlm(
            "y ~ x",
            df,
            method="gibbs",
            constant=["Intercept"],
            draws=3000,
            burnin=500,
            seed=1,
        )
    m = sp.dlm("y ~ x", df, constant=["Intercept"])
    assert g.draws.shape == (3000, 3)
    assert {"sd", "lower", "upper", "ess"} <= set(g.variances.columns)
    lo, hi = g.variances.loc["state:x", ["lower", "upper"]]
    assert lo < m.variances.loc["state:x", "estimate"] < hi
    assert np.corrcoef(g.smoothed["x"], m.smoothed["x"])[0, 1] > 0.99
    again = sp.dlm(
        "y ~ x",
        df,
        method="gibbs",
        constant=["Intercept"],
        draws=3000,
        burnin=500,
        seed=1,
    )
    assert g.draws.equals(again.draws)
    assert sp.geweke_diag(g.draws[["obs", "state:x"]]).table.shape == (2, 2)


def test_refusals(df):
    with pytest.raises(sp.MethodIncompatibility, match="method must be"):
        sp.dlm("y ~ x", df, method="em")
    with pytest.raises(sp.MethodIncompatibility, match="not in the model"):
        sp.dlm("y ~ x", df, constant=["z"])
    with pytest.raises(sp.MethodIncompatibility, match="time column"):
        sp.dlm("y ~ x", df, time="date")
    with pytest.raises(sp.MethodIncompatibility, match="state_var"):
        sp.dlm("y ~ x", df, state_var=[0.1, 0.1, 0.1])
    with pytest.raises(sp.MethodIncompatibility, match="obs_var must be positive"):
        sp.dlm("y ~ x", df, obs_var=0.0)
    with pytest.raises(sp.DataInsufficient):
        sp.dlm("y ~ x", df.head(4))
    holes = df.copy()
    holes.loc[10, "y"] = np.nan
    with pytest.raises(sp.MethodIncompatibility, match="not\\s+adjacent"):
        sp.dlm("y ~ x", holes)


def test_registered():
    assert "dlm" in sp.list_functions()
    assert sp.describe_function("dlm")["category"] == "timeseries"
    assert "Kalman" in sp.bibtex(keys=["kalman1960new"])
