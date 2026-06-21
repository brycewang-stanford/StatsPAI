"""Tier E — invariance / metamorphic tests for ``sp.iv`` (2SLS).

These assert the *defining* algebraic properties of just-identified linear IV
directly from fitted results — no external reference (no ``linearmodels``)
needed. The generic invariances (row permutation, row duplication) are driven
by ``hypothesis``; the IV-specific metamorphic relations (instrument scaling,
outcome scale/shift equivariance, the FWL partialling-out identity) are
seeded-randomized in the house style of
``tests/test_dml_orthogonality_invariants.py``.

For the just-identified single-endogenous model with one excluded instrument,
the 2SLS estimator is the simple IV estimator and satisfies:

* invariance to the *scale* of the instrument (the projection ``x̂`` is a
  scalar multiple, which cancels in ``(ẑ'y)/(ẑ'x)``),
* exact equivariance to outcome rescaling/shift and regressor rescaling,
* the Frisch–Waugh–Lovell identity against the included exogenous controls.

References
----------
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel
  Data* (2nd ed.), ch. 5 (IV/2SLS). [@wooldridge2010econometric]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

from ._helpers import (
    HAS_HYPOTHESIS,
    assert_invariant,
    assert_scaled,
    coef,
    make_iv,
    stderr,
)

pytestmark = pytest.mark.filterwarnings("ignore")

FML = "y ~ (x ~ z) + w1"


def _fit(df, fml=FML):
    return sp.iv(fml, data=df, method="2sls")


@pytest.fixture(scope="module")
def base_df():
    return make_iv(n=500, beta=2.0, seed=11, n_exog=1)


# --------------------------------------------------------------------------- #
# E1 — row permutation invariance (hypothesis-driven)                         #
# --------------------------------------------------------------------------- #
if HAS_HYPOTHESIS:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(parent=settings.get_profile("tier_eg"))
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_iv_row_permutation_invariant(seed):
        """Shuffling observation order leaves β̂_x and SE bit-identical."""
        df = make_iv(n=300, seed=7)
        base = _fit(df)
        perm = df.sample(frac=1.0, random_state=seed % (2**31)).reset_index(drop=True)
        got = _fit(perm)
        assert_invariant(coef(base, "x"), coef(got, "x"), what="beta_x (perm)")
        assert_invariant(stderr(base, "x"), stderr(got, "x"), what="se_x (perm)")


def test_iv_row_permutation_explicit():
    """Deterministic permutation invariance (runs even without hypothesis)."""
    df = make_iv(n=400, seed=3)
    base = _fit(df)
    perm = df.iloc[::-1].reset_index(drop=True)
    got = _fit(perm)
    assert_invariant(coef(base, "x"), coef(got, "x"), what="beta_x")
    assert_invariant(stderr(base, "x"), stderr(got, "x"), what="se_x")
    assert_invariant(coef(base, "w1"), coef(got, "w1"), what="gamma_w1")


# --------------------------------------------------------------------------- #
# E2 / E3 — outcome location & scale equivariance                             #
# --------------------------------------------------------------------------- #
def test_iv_outcome_shift_leaves_slopes(base_df):
    """y -> y + c shifts only the intercept; β̂_x and γ̂_w1 unchanged."""
    base = _fit(base_df)
    shifted = base_df.assign(y=base_df["y"] + 7.5)
    got = _fit(shifted)
    assert_invariant(coef(base, "x"), coef(got, "x"), what="beta_x (shift)")
    assert_invariant(coef(base, "w1"), coef(got, "w1"), what="gamma_w1 (shift)")
    assert_invariant(
        coef(got, "Intercept") - coef(base, "Intercept"),
        7.5,
        atol=1e-6,
        what="intercept shift",
    )


def test_iv_outcome_scale_equivariant(base_df):
    """y -> a*y scales β̂ and SE by a; t-stat invariant."""
    a = 3.0
    base = _fit(base_df)
    scaled = base_df.assign(y=a * base_df["y"])
    got = _fit(scaled)
    assert_scaled(coef(base, "x"), coef(got, "x"), a, what="beta_x")
    assert_scaled(stderr(base, "x"), stderr(got, "x"), a, what="se_x")
    assert_invariant(
        coef(got, "x") / stderr(got, "x"),
        coef(base, "x") / stderr(base, "x"),
        rtol=1e-6,
        what="t-stat",
    )


# --------------------------------------------------------------------------- #
# E4 — exogenous-regressor scale equivariance                                 #
# --------------------------------------------------------------------------- #
def test_iv_exog_scale_equivariant(base_df):
    """w1 -> w1/s scales γ̂_w1 by s and leaves β̂_x untouched."""
    s = 10.0
    base = _fit(base_df)
    scaled = base_df.assign(w1=base_df["w1"] / s)
    got = _fit(scaled)
    assert_scaled(coef(base, "w1"), coef(got, "w1"), s, what="gamma_w1")
    assert_invariant(coef(base, "x"), coef(got, "x"), what="beta_x (exog scale)")


# --------------------------------------------------------------------------- #
# E (IV-specific) — instrument-scale invariance                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s", [0.1, 100.0, 1e3])
def test_iv_instrument_scale_invariant(base_df, s):
    """Just-identified 2SLS is invariant to the scale of the instrument:
    β̂_x and SE_x are unchanged when z -> s*z."""
    base = _fit(base_df)
    scaled = base_df.assign(z=s * base_df["z"])
    got = _fit(scaled)
    assert_invariant(coef(base, "x"), coef(got, "x"), what=f"beta_x (z*{s})")
    assert_invariant(stderr(base, "x"), stderr(got, "x"), what=f"se_x (z*{s})")


# --------------------------------------------------------------------------- #
# E5 — row-duplication: point estimate exact, SE shrinks ~1/sqrt(2)           #
# --------------------------------------------------------------------------- #
def test_iv_duplication_point_invariant(base_df):
    base = _fit(base_df)
    dup = pd.concat([base_df, base_df], ignore_index=True)
    got = _fit(dup)
    assert_invariant(coef(base, "x"), coef(got, "x"), what="beta_x (dup)")
    # SE shrinks toward 1/sqrt(2) of the original (dof correction keeps it
    # from being exact); it must at least strictly decrease.
    ratio = stderr(got, "x") / stderr(base, "x")
    assert ratio < 1.0, f"SE did not shrink under duplication (ratio={ratio})"
    np.testing.assert_allclose(ratio, 1 / np.sqrt(2), rtol=0.05)


# --------------------------------------------------------------------------- #
# E8 — covariate column reordering invariance                                 #
# --------------------------------------------------------------------------- #
def test_iv_exog_reorder_invariant():
    """Permuting the exogenous controls leaves every estimate unchanged."""
    df = make_iv(n=400, seed=5, n_exog=2)
    a = sp.iv("y ~ (x ~ z) + w1 + w2", data=df, method="2sls")
    b = sp.iv("y ~ (x ~ z) + w2 + w1", data=df, method="2sls")
    for term in ["x", "w1", "w2", "Intercept"]:
        assert_invariant(coef(a, term), coef(b, term), what=f"coef[{term}]")
        assert_invariant(stderr(a, term), stderr(b, term), what=f"se[{term}]")


# --------------------------------------------------------------------------- #
# E9 — Frisch–Waugh–Lovell partialling-out identity                           #
# --------------------------------------------------------------------------- #
def test_iv_fwl_identity(base_df):
    """β̂_x from the full IV fit equals the IV slope after partialling the
    included exogenous controls (intercept + w1) out of y, x and z."""
    base = _fit(base_df)
    W = np.column_stack([np.ones(len(base_df)), base_df["w1"].to_numpy()])

    def resid(v):
        b, *_ = np.linalg.lstsq(W, v, rcond=None)
        return v - W @ b

    yr = resid(base_df["y"].to_numpy())
    xr = resid(base_df["x"].to_numpy())
    zr = resid(base_df["z"].to_numpy())
    fwl_beta = (zr @ yr) / (zr @ xr)
    assert_invariant(coef(base, "x"), fwl_beta, rtol=1e-7, what="FWL beta_x")
