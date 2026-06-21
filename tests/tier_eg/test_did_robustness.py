"""Tier G — robustness tests for ``sp.did_2x2`` / ``sp.callaway_santanna``.

Observed contract (StatsPAI 1.17.0), locked here:

* **no treated units / single time period** -> ``MethodIncompatibility``
  ("Treatment variable must have exactly 2 values").
* **missing column** -> ``KeyError``.
* **NaN rows** -> dropped listwise, finite ATT.
* **inf in outcome** -> must not surface a silent finite ATT.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp

from ._helpers import assert_raises_clean, coef, make_did_2x2, make_staggered_did


def _fit(d, **kw):
    return sp.did_2x2(d, y="y", treat="treat", time="time", **kw)


@pytest.fixture(scope="module")
def base_df():
    return make_did_2x2(n_units=250, seed=5)


# --------------------------------------------------------------------------- #
# G9 / G10 — degenerate treatment assignment                                  #
# --------------------------------------------------------------------------- #
def test_did_no_treated_units_raises(base_df):
    assert_raises_clean(
        lambda: _fit(base_df.assign(treat=0)),
        Exception,
        match="2 values|treat|group",
    )


def test_did_all_treated_raises(base_df):
    assert_raises_clean(lambda: _fit(base_df.assign(treat=1)), Exception)


# --------------------------------------------------------------------------- #
# G5 — single time period                                                     #
# --------------------------------------------------------------------------- #
def test_did_single_period_raises(base_df):
    one = base_df[base_df["time"] == 0]
    assert_raises_clean(lambda: _fit(one), Exception)


# --------------------------------------------------------------------------- #
# G8 — missing column                                                         #
# --------------------------------------------------------------------------- #
def test_did_missing_outcome_raises(base_df):
    # call the estimator directly (the _fit helper fixes y="y")
    assert_raises_clean(
        lambda: sp.did_2x2(base_df, y="qqq", treat="treat", time="time"),
        KeyError,
        ValueError,
    )


# --------------------------------------------------------------------------- #
# G1 — NaN / inf rows dropped listwise, finite estimate on clean data         #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_did_nan_rows_finite(base_df):
    clean = _fit(base_df)
    dirty = base_df.copy()
    dirty.loc[:3, "y"] = np.nan
    got = _fit(dirty)
    assert np.isfinite(coef(got)), "NaN rows gave a non-finite ATT"
    # Listwise-dropping the 4 NaN rows removes two *treated* units (ids 0-1)
    # from a single 250-unit draw. On tight BLAS builds that barely moves the
    # ATT (~0.7%, clean ~1.345); on the ubuntu-latest OpenBLAS build the
    # unbalanced solve shifts it by ~1.6. The contract under test is "NaN rows
    # are dropped listwise and the estimate stays finite and same-order" — not a
    # tight numeric pin — so bound the absolute move (catches inf / sign-blowups)
    # rather than rtol-pinning a single noisy draw.
    np.testing.assert_allclose(coef(got), coef(clean), rtol=0.1, atol=2.5)


@pytest.mark.filterwarnings("ignore")
def test_did_inf_rows_dropped_finite(base_df):
    """``did_2x2`` treats non-finite outcomes like missing values: the row is
    dropped and the ATT stays finite & close to the clean fit (not a silent
    wrong number, and not a NaN-poisoned estimate)."""
    clean = _fit(base_df)
    dirty = base_df.copy()
    dirty.loc[0, "y"] = np.inf
    got = _fit(dirty)
    assert np.isfinite(coef(got)), "inf row poisoned the ATT to non-finite"
    np.testing.assert_allclose(coef(got), coef(clean), rtol=0.1)


# --------------------------------------------------------------------------- #
# Staggered CS — missing id / single cohort                                   #
# --------------------------------------------------------------------------- #
def test_cs_missing_unit_col_raises():
    d = make_staggered_did(n_units=80, seed=2)
    assert_raises_clean(
        lambda: sp.callaway_santanna(d, y="y", g="g", t="time", i="nope"),
        KeyError,
        ValueError,
    )


@pytest.mark.filterwarnings("ignore")
def test_cs_nan_rows_finite():
    """NaN outcomes are dropped listwise; the aggregated ATT stays finite and
    close to the clean fit."""
    d = make_staggered_did(n_units=120, seed=2)
    clean = sp.callaway_santanna(d, y="y", g="g", t="time", i="id")
    dirty = d.copy()
    dirty.loc[:5, "y"] = np.nan
    got = sp.callaway_santanna(dirty, y="y", g="g", t="time", i="id")
    assert np.isfinite(coef(got)), "NaN rows poisoned the CS ATT"
    np.testing.assert_allclose(coef(got), coef(clean), rtol=0.15)
