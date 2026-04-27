"""Tests for ``sp.fast.within`` and ``sp.fast.{i, fe_interact, sw, csw}``.

Phase 3 acceptance gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _panel(n_units=80, n_periods=15, seed=0):
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    n = i.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = rng.normal(0, 0.5, size=n_units)[i]
    g = rng.normal(0, 0.3, size=n_periods)[t]
    y = 1.0 + 0.3 * x1 - 0.2 * x2 + a + g + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "i": i, "t": t})


# ---------------------------------------------------------------------------
# WithinTransformer — basic correctness
# ---------------------------------------------------------------------------

def test_within_transform_matches_demean():
    df = _panel(seed=1)
    wt = sp.fast.within(df, fe=["i", "t"], drop_singletons=False)
    y_in = df["y"].to_numpy()
    y_via_within, _ = wt.transform(y_in)
    y_direct, _ = sp.fast.demean(y_in, df[["i", "t"]], drop_singletons=False)
    assert np.allclose(y_via_within, y_direct, atol=1e-12)


def test_within_transform_columns_returns_dataframe():
    df = _panel(seed=2)
    wt = sp.fast.within(df, fe=["i", "t"], drop_singletons=False)
    out = wt.transform_columns(df, ["x1", "x2"])
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["x1", "x2"]
    assert out.index.equals(df.index)
    # Values match per-column transform
    x1_dem, _ = wt.transform(df["x1"].to_numpy())
    assert np.allclose(out["x1"].to_numpy(), x1_dem, atol=1e-12)


def test_within_caches_singleton_drop():
    """Singletons detected once at construction; subsequent transforms reuse."""
    rng = np.random.default_rng(3)
    n = 200
    i = rng.integers(0, 30, size=n).astype(np.int64)
    i[-1] = 999  # force a singleton
    df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n), "fe": i})

    wt = sp.fast.within(df, fe=["fe"], drop_singletons=True)
    assert wt.n_dropped >= 1
    assert wt.keep_mask[-1] == False  # noqa: E712

    y_dem, _ = wt.transform(df["y"].to_numpy())
    assert y_dem.shape[0] == wt.n_kept


def test_within_already_masked_path():
    df = _panel(seed=4)
    wt = sp.fast.within(df, fe=["i", "t"], drop_singletons=False)
    y_full = df["y"].to_numpy()
    y_kept = y_full[wt.keep_mask]
    out_a, _ = wt.transform(y_full, already_masked=False)
    out_b, _ = wt.transform(y_kept, already_masked=True)
    assert np.allclose(out_a, out_b, atol=1e-12)


def test_within_accepts_multiple_fe_input_shapes():
    df = _panel(seed=5)
    wt1 = sp.fast.within(df, fe=["i", "t"], drop_singletons=False)
    wt2 = sp.fast.within(fe=df[["i", "t"]], drop_singletons=False)
    wt3 = sp.fast.within(fe=df[["i", "t"]].to_numpy(), drop_singletons=False)
    wt4 = sp.fast.within(fe=[df["i"].to_numpy(), df["t"].to_numpy()],
                         drop_singletons=False)

    y = df["y"].to_numpy()
    a, _ = wt1.transform(y)
    for wt in (wt2, wt3, wt4):
        b, _ = wt.transform(y)
        assert np.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# DSL: i()
# ---------------------------------------------------------------------------

def test_i_default_drops_first_level():
    s = pd.Series([2010, 2011, 2012, 2010, 2011], name="year")
    out = sp.fast.i(s)
    # Default drop_first=True ⇒ 2010 dropped, two columns remain
    assert list(out.columns) == ["year::2011", "year::2012"]
    assert out.shape == (5, 2)


def test_i_with_explicit_ref():
    s = pd.Series([2010, 2011, 2012, 2010, 2011], name="year")
    out = sp.fast.i(s, ref=2011)
    assert "year::2011" not in out.columns
    assert "year::2010" in out.columns
    assert "year::2012" in out.columns


def test_i_with_unknown_ref_raises():
    s = pd.Series([2010, 2011, 2012], name="year")
    with pytest.raises(ValueError, match="ref"):
        sp.fast.i(s, ref=2099)


def test_i_event_study_in_fepois():
    """End-to-end: i(year, ref=...) used inside an event-study fepois."""
    rng = np.random.default_rng(6)
    n = 1000
    firm = rng.integers(0, 50, size=n)
    year = rng.choice([2010, 2011, 2012, 2013, 2014], size=n)
    treated = (firm % 5 == 0).astype(float)
    eta = 0.5 + 0.3 * (year >= 2012) * treated  # positive treatment post-2012
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.int64)
    df = pd.DataFrame({"y": y, "firm": firm, "year": year, "treated": treated})

    dummies = sp.fast.i(df["year"], ref=2010)
    dummy_cols = list(dummies.columns)
    df_aug = pd.concat([df, dummies], axis=1)

    rhs = " + ".join([f"`{c}`" for c in dummy_cols])  # backticks for ::
    # Backticks aren't supported by the minimal DSL; rename for safety.
    sane = {c: c.replace("::", "_") for c in dummy_cols}
    df_aug = df_aug.rename(columns=sane)
    sane_cols = list(sane.values())
    rhs = " + ".join(sane_cols)

    fit = sp.fast.fepois(f"y ~ {rhs} | firm", df_aug)
    assert fit.converged


# ---------------------------------------------------------------------------
# DSL: fe_interact (i^j)
# ---------------------------------------------------------------------------

def test_fe_interact_two_columns():
    a = np.array([0, 0, 1, 1, 2])
    b = np.array([0, 1, 0, 1, 0])
    codes = sp.fast.fe_interact(a, b)
    # Five unique (a, b) pairs ⇒ 5 unique codes
    assert len(np.unique(codes)) == 5


def test_fe_interact_three_columns():
    a = np.array([0, 0, 0, 0, 1])
    b = np.array([0, 0, 1, 1, 0])
    c = np.array([0, 1, 0, 1, 0])
    codes = sp.fast.fe_interact(a, b, c)
    assert len(np.unique(codes)) == 5
    # Codes should be in [0, 5)
    assert codes.min() >= 0
    assert codes.max() < 5


def test_fe_interact_passes_to_fepois():
    """End-to-end: i^j interacted FE used inside fepois."""
    rng = np.random.default_rng(7)
    n = 500
    firm = rng.integers(0, 20, size=n)
    year = rng.integers(0, 5, size=n)
    eta = 0.3 * rng.normal(size=n)
    y = rng.poisson(np.exp(eta)).astype(np.int64)
    df = pd.DataFrame({
        "y": y, "x": rng.normal(size=n),
        "firm": firm, "year": year,
        "firm_x_year": sp.fast.fe_interact(firm, year),
    })
    fit = sp.fast.fepois("y ~ x | firm_x_year", df)
    assert fit.converged


# ---------------------------------------------------------------------------
# DSL: sw / csw
# ---------------------------------------------------------------------------

def test_sw_emits_separate_specs():
    out = sp.fast.sw(["x1"], ["x2"], ["x1", "x2"])
    assert out == [["x1"], ["x2"], ["x1", "x2"]]


def test_csw_cumulative():
    out = sp.fast.csw(["x1"], ["x2"], ["x3"])
    assert out == [["x1"], ["x1", "x2"], ["x1", "x2", "x3"]]


def test_sw_drives_multiple_regressions():
    """End-to-end: use sw() to fit several specs, collect results."""
    df = _panel(seed=9)

    results = []
    for cols in sp.fast.csw(["x1"], ["x2"]):
        formula = "y ~ " + " + ".join(cols) + " | i + t"
        # Use existing OLS path (sp.feols) since sw is regression-agnostic
        fit = sp.feols(formula, data=df)
        results.append(fit)

    assert len(results) == 2
    # Second spec has both regressors so params is one larger
    p0 = results[0].params
    p1 = results[1].params
    assert len(p1) == len(p0) + 1
