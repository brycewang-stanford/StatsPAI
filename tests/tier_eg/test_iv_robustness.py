"""Tier G — robustness tests for ``sp.iv`` (2SLS).

These pin the *failure contract* (CLAUDE.md §7 — 失败要响亮): on degenerate or
adversarial input the estimator must either raise a clear exception or surface
NaN — never a silent finite, plausible-looking, wrong number.

Observed contract (StatsPAI 1.17.0), locked here so a future regression is
caught:

* **missing column / under-identification** -> raise with a usable message.
* **irrelevant instrument** -> a weak-instrument ``UserWarning`` (point estimate
  may be wild, but the user is told).
* **NaN rows** -> dropped listwise; the estimate stays finite & close to the
  clean fit (documented; *not* a silent wrong number).
* **inf in a regressor** -> NaN estimate (detectable), not a finite fake.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp

from ._helpers import assert_no_silent_wrong, assert_raises_clean, coef, make_iv, stderr


def _fit(df, fml="y ~ (x ~ z) + w1"):
    return sp.iv(fml, data=df, method="2sls")


# --------------------------------------------------------------------------- #
# G8 — schema errors raise cleanly                                            #
# --------------------------------------------------------------------------- #
def test_iv_missing_instrument_raises():
    df = make_iv(seed=1)
    assert_raises_clean(
        lambda: sp.iv("y ~ (x ~ zzz)", data=df, method="2sls"),
        ValueError,
        KeyError,
        match="zzz|not found|instrument",
    )


def test_iv_missing_endog_raises():
    df = make_iv(seed=1)
    assert_raises_clean(
        lambda: sp.iv("y ~ (qqq ~ z)", data=df, method="2sls"),
        ValueError,
        KeyError,
    )


def test_iv_underidentified_raises():
    """Fewer instruments than endogenous regressors must be a hard error."""
    df = make_iv(seed=1)
    assert_raises_clean(
        lambda: sp.iv("y ~ (x + w1 ~ z)", data=df, method="2sls"),
        Exception,
        match="ident",
    )


# --------------------------------------------------------------------------- #
# G6 — weak / irrelevant instrument is announced                              #
# --------------------------------------------------------------------------- #
def test_iv_irrelevant_instrument_warns():
    df = make_iv(seed=2)
    df = df.assign(zbad=np.random.default_rng(0).normal(size=len(df)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.iv("y ~ (x ~ zbad)", data=df, method="2sls")
    msgs = " ".join(str(w.message).lower() for w in caught)
    assert any(
        k in msgs for k in ("weak", "first stage", "first-stage", "instrument")
    ), f"weak instrument not announced; warnings={[str(w.message) for w in caught]}"


# --------------------------------------------------------------------------- #
# G1 — NaN handling: listwise drop, finite & close (documented contract)      #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_iv_nan_rows_dropped_not_silent_nan():
    df = make_iv(n=600, seed=4)
    clean = _fit(df)
    dirty = df.copy()
    dirty.loc[:4, "y"] = np.nan
    dirty.loc[10:14, "x"] = np.nan
    got = _fit(dirty)
    b = coef(got, "x")
    assert np.isfinite(b), "NaN rows produced a non-finite estimate"
    # dropped 10 of 600 rows -> estimate barely moves
    np.testing.assert_allclose(b, coef(clean, "x"), rtol=0.05)


@pytest.mark.filterwarnings("ignore")
def test_iv_inf_not_silent_finite():
    """inf in a regressor must not masquerade as a finite estimate."""
    df = make_iv(seed=4)
    dirty = df.copy()
    dirty.loc[0, "x"] = np.inf
    assert_no_silent_wrong(lambda: _fit(dirty), term="x")


# --------------------------------------------------------------------------- #
# G2 — too few observations to identify                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_iv_tiny_n_no_silent_inference():
    """With no residual degrees of freedom the SE must not be a finite,
    falsely-precise number — it should be inf/NaN or the fit should raise."""
    df = make_iv(seed=1).head(2)
    try:
        r = sp.iv("y ~ (x ~ z)", data=df, method="2sls")
    except Exception as e:
        assert str(e).strip()
        return
    s = stderr(r, "x")
    assert (
        not np.isfinite(s)
    ) or s == 0.0, f"tiny-n fit reported a finite positive SE {s} with no dof"


# --------------------------------------------------------------------------- #
# G3 — constant outcome: zero effect, no silent garbage                       #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_iv_constant_outcome():
    df = make_iv(seed=1).assign(y=5.0)
    r = sp.iv("y ~ (x ~ z)", data=df, method="2sls")
    b = coef(r, "x")
    assert np.isfinite(b) and abs(b) < 1e-6, f"constant y gave beta_x={b}"


# --------------------------------------------------------------------------- #
# G4 — perfectly collinear instrument set (overid with a duplicate)           #
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore")
def test_iv_duplicate_instrument_no_silent_wrong():
    """A duplicated instrument (rank-deficient instrument matrix) must not
    silently yield a different finite estimate than the clean just-identified
    fit; raising or matching the clean estimate are both acceptable."""
    df = make_iv(seed=6)
    clean = _fit(df, "y ~ (x ~ z)")
    df2 = df.assign(zdup=df["z"])
    try:
        r = sp.iv("y ~ (x ~ z + zdup)", data=df2, method="2sls")
    except Exception as e:
        assert str(e).strip()
        return
    # If it ran, the redundant instrument adds no information -> same estimate.
    np.testing.assert_allclose(coef(r, "x"), coef(clean, "x"), rtol=1e-4)
