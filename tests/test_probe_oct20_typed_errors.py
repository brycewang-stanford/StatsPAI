"""Typed errors where the 2026-10-20 probe of the new functions hit raw
numpy / pandas exceptions (or, for the spatial NaN case, a silent NaN).

Each case states the input a user or an agent plausibly sends and the
error class that now names the problem.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"c": np.repeat(np.arange(10), 20), "t": np.tile(np.arange(20), 10)}
    )
    df["x"] = rng.normal(size=len(df))
    df["x2"] = 2 * df["x"]
    df["z"] = df["x"] + rng.normal(size=len(df))
    df["y"] = df["x"] + rng.normal(size=len(df))
    return df


@pytest.fixture(scope="module")
def ring():
    n = 40
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"y": rng.normal(size=n), "x": rng.normal(size=n), "g": np.tile(["a", "b"], 20)}
    )
    return W, df


def test_xtpcse_collinear_regressors(panel):
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.xtpcse(panel, y="y", x=["x", "x2"], id="c", time="t")


def test_xtpcse_ar1_without_consecutive_periods(panel):
    # time coded in steps of 2: rho had no pairs and came out 0/0
    with pytest.raises(DataInsufficient, match="consecutive"):
        sp.xtpcse(
            panel.assign(t=panel["t"] * 2),
            y="y",
            x=["x"],
            id="c",
            time="t",
            correlation="ar1",
        )


def test_panel_granger_constant_series_within_a_unit(panel):
    with pytest.raises(DataInsufficient, match="singular"):
        sp.granger_causality(
            data=panel.assign(xc=panel["c"] * 1.0),
            caused="y",
            causing="xc",
            lags=1,
            id="c",
            time="t",
        )


def test_xtivreg_one_cluster(panel):
    with pytest.raises(DataInsufficient, match="at least two"):
        sp.xtivreg(
            panel.assign(one=1),
            y="y",
            endog="x",
            instruments=["z"],
            id="c",
            cluster="one",
        )


def test_wild_cluster_boot_unknown_bootcluster(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.regress("y ~ x", data=panel, cluster="c")
    with pytest.raises(ColumnNotFound):
        sp.wild_cluster_boot(fit, panel, "c", "x", bootcluster="nope", n_boot=99)


@pytest.mark.parametrize("fn", ["sar", "sdem", "gns"])
def test_spatial_formula_terms_must_be_columns(ring, fn):
    W, df = ring
    with pytest.raises(MethodIncompatibility, match="plain column names"):
        getattr(sp, fn)(W, df, "y ~ x + C(g)")


def test_spatial_unknown_column_suggests(ring):
    W, df = ring
    with pytest.raises(ColumnNotFound):
        sp.sar(W, df, "y ~ xx")


def test_spatial_missing_values_are_refused(ring):
    W, df = ring
    bad = df.copy()
    bad.loc[3, "x"] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing values"):
        sp.sem(W, bad, "y ~ x")


@pytest.fixture(scope="module")
def labelled():
    rng = np.random.default_rng(0)
    n = 600
    a = rng.integers(0, 2, n)
    m = 0.5 * a + rng.normal(size=n)
    m2 = 0.3 * a + m + rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": np.where(a == 1, "T", "C"),
            "m": m,
            "m2": m2,
            "y": a + m + m2 + rng.normal(size=n),
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 2, n),
            "s": (rng.uniform(size=n) < 0.8).astype(int),
            "t": rng.exponential(5, n),
            "d": (rng.uniform(size=n) < 0.7).astype(int),
        }
    )


LABEL_CALLS = {
    "mediate_weighted": lambda d: sp.mediate(
        d, y="y", treat="a", mediator="m", weights="w", n_boot=20
    ),
    "mediate": lambda d: sp.mediate(d, y="y", treat="a", mediator="m", n_boot=20),
    "four_way": lambda d: sp.four_way_decomposition(d, y="y", treat="a", mediator="m"),
    "lee_bounds": lambda d: sp.lee_bounds(d, "y", "a", "s"),
    "mediate_paths": lambda d: sp.mediate_paths(
        d, y="y", treat="a", mediators=["m", "m2"], n_boot=20
    ),
    "stteffects": lambda d: sp.stteffects(d, "t", "d", "a", covariates=["x"]),
}


@pytest.mark.parametrize("fn", sorted(LABEL_CALLS))
def test_label_treatment_is_a_named_error(labelled, fn):
    # a raw "could not convert string to float: 'T'" before
    with pytest.raises(MethodIncompatibility, match="holds labels"):
        LABEL_CALLS[fn](labelled)


@pytest.fixture(scope="module")
def holed_weights():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    d = (x + rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame(
        {
            "x": x,
            "d": d,
            "y": d + x + rng.normal(size=n),
            "w": rng.uniform(0.5, 2, n),
            "t": rng.exponential(5, n),
            "ev": rng.integers(0, 3, n),
        }
    )
    df.loc[:3, "w"] = np.nan
    return df


WEIGHT_CALLS = {
    "ebalance": lambda d: sp.ebalance(
        d, y="y", treat="d", covariates=["x"], weights="w"
    ),
    "match": lambda d: sp.match(
        d, y="y", treat="d", covariates=["x"], sampling_weights="w"
    ),
    "finegray": lambda d: sp.finegray(
        d, duration="t", event="ev", x=["x"], cause=1, weights="w"
    ),
}


@pytest.mark.parametrize("fn", sorted(WEIGHT_CALLS))
def test_rows_with_a_missing_weight_are_dropped_aloud(holed_weights, fn):
    # 2026-10-20 probe: the 4 rows were dropped without a word
    with pytest.warns(UserWarning, match="dropped 4 row"):
        r = WEIGHT_CALLS[fn](holed_weights)
    assert r.n_obs == len(holed_weights) - 4
