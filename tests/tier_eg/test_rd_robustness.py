"""Tier G — robustness tests for ``sp.rdrobust`` (sharp RD).

Observed contract (StatsPAI 1.17.0), locked here:

* **all observations on one side of the cutoff / cutoff outside the support**
  -> ``ValueError`` ("Not enough observations on each side of the cutoff").
* **missing column** -> ``ValueError`` naming it.
* **observation weights** -> ``NotImplementedError`` (honest §7 limitation,
  not a silent no-op) — so the Tier E weight-identity is N/A for RD.
* **NaN / inf in y or x** -> dropped listwise, τ̂ stays finite & close.
* **constant outcome** -> τ̂ ≈ 0 (no jump), finite.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp

from ._helpers import assert_raises_clean, coef, make_rd


@pytest.fixture(scope="module")
def base_df():
    return make_rd(n=1500, tau=3.0, cutoff=0.0, seed=7)


# --------------------------------------------------------------------------- #
# G — one-sided support / cutoff outside range                                #
# --------------------------------------------------------------------------- #
def test_rd_all_one_side_raises(base_df):
    left_only = base_df[base_df["x"] < 0]
    assert_raises_clean(
        lambda: sp.rdrobust(left_only, y="y", x="x", c=0.0),
        ValueError,
        match="side|cutoff|observ",
    )


def test_rd_cutoff_outside_support_raises(base_df):
    assert_raises_clean(
        lambda: sp.rdrobust(base_df, y="y", x="x", c=999.0),
        ValueError,
        match="side|cutoff|observ",
    )


# --------------------------------------------------------------------------- #
# G8 — missing column                                                         #
# --------------------------------------------------------------------------- #
def test_rd_missing_column_raises(base_df):
    assert_raises_clean(
        lambda: sp.rdrobust(base_df, y="qqq", x="x"),
        ValueError,
        KeyError,
        match="qqq|not found",
    )


# --------------------------------------------------------------------------- #
# weight-identity is N/A: weights must raise, not be silently ignored          #
# --------------------------------------------------------------------------- #
def test_rd_weights_not_silently_ignored(base_df):
    # Implemented in 1.32: non-constant weights must move the estimate.
    rng = np.random.default_rng(1)
    plain = sp.rdrobust(base_df, y="y", x="x")
    w = rng.uniform(0.2, 3.0, len(base_df))
    weighted = sp.rdrobust(base_df.assign(w=w), y="y", x="x", weights="w")
    assert weighted.estimate != plain.estimate
    assert weighted.model_info["weighted"] is True


# --------------------------------------------------------------------------- #
# G1 — NaN / inf dropped listwise, finite & close                             #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_rd_nan_rows_finite(base_df):
    clean = sp.rdrobust(base_df, y="y", x="x")
    dirty = base_df.copy()
    dirty.loc[:9, "y"] = np.nan
    got = sp.rdrobust(dirty, y="y", x="x")
    assert np.isfinite(coef(got)), "NaN rows poisoned tau"
    np.testing.assert_allclose(coef(got), coef(clean), rtol=0.1)


@pytest.mark.filterwarnings("ignore")
def test_rd_inf_running_var_finite(base_df):
    clean = sp.rdrobust(base_df, y="y", x="x")
    dirty = base_df.copy()
    dirty.loc[0, "x"] = np.inf
    got = sp.rdrobust(dirty, y="y", x="x")
    assert np.isfinite(coef(got)), "inf running value poisoned tau"
    np.testing.assert_allclose(coef(got), coef(clean), rtol=0.1)


# --------------------------------------------------------------------------- #
# G3 — constant outcome -> no jump                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_rd_constant_outcome_zero_jump(base_df):
    r = sp.rdrobust(base_df.assign(y=5.0), y="y", x="x")
    assert (
        np.isfinite(coef(r)) and abs(coef(r)) < 1e-6
    ), f"constant outcome should give a zero jump, got {coef(r)}"


# --------------------------------------------------------------------------- #
# G2 — tiny n: no crash; finite estimate or a clean raise                     #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_rd_tiny_n_no_crash(base_df):
    tiny = base_df.head(8)
    try:
        r = sp.rdrobust(tiny, y="y", x="x")
    except Exception as e:
        assert str(e).strip()
        return
    assert np.isfinite(coef(r)), "tiny-n fit returned a non-finite tau silently"
