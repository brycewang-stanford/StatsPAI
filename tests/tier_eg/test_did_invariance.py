"""Tier E — invariance / metamorphic tests for ``sp.did_2x2`` and
``sp.callaway_santanna`` (staggered DiD).

The DiD estimand is a difference of group×time means, so it *differences out*
additive level effects (E2) and flips sign under a treated⇄control relabel (E7)
— its defining metamorphic properties. We pin both, plus the generic
permutation / scale / duplication / weight-identity invariances, on the clean
2×2 estimator, and the headline invariances on the staggered Callaway–Sant'Anna
ATT(g,t) aggregation.

References
----------
- Callaway, B. & Sant'Anna, P. H. C. (2021). Difference-in-differences with
  multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
  [@callaway2021difference]
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
    make_did_2x2,
    make_staggered_did,
    stderr,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _fit(d, **kw):
    return sp.did_2x2(d, y="y", treat="treat", time="time", **kw)


@pytest.fixture(scope="module")
def base_df():
    return make_did_2x2(n_units=300, att=2.0, seed=5)


# --------------------------------------------------------------------------- #
# E1 — row permutation (hypothesis-driven)                                    #
# --------------------------------------------------------------------------- #
if HAS_HYPOTHESIS:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(parent=settings.get_profile("tier_eg"), max_examples=12)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_did_row_permutation_invariant(seed):
        d = make_did_2x2(n_units=200, seed=6)
        base = _fit(d)
        perm = d.sample(frac=1.0, random_state=seed % (2**31)).reset_index(drop=True)
        got = _fit(perm)
        assert_invariant(coef(base), coef(got), what="ATT (perm)")
        assert_invariant(stderr(base), stderr(got), what="se (perm)")


def test_did_permutation_explicit(base_df):
    base = _fit(base_df)
    got = _fit(base_df.iloc[::-1].reset_index(drop=True))
    assert_invariant(coef(base), coef(got), what="ATT")
    assert_invariant(stderr(base), stderr(got), what="se")


# --------------------------------------------------------------------------- #
# E2 — additive level shift differenced out                                   #
# --------------------------------------------------------------------------- #
def test_did_outcome_shift_invariant(base_df):
    base = _fit(base_df)
    got = _fit(base_df.assign(y=base_df["y"] + 9.0))
    assert_invariant(coef(base), coef(got), what="ATT (y shift)")
    assert_invariant(stderr(base), stderr(got), what="se (y shift)")


# --------------------------------------------------------------------------- #
# E3 — outcome scale equivariance                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a", [3.0, -2.0])
def test_did_outcome_scale_equivariant(base_df, a):
    base = _fit(base_df)
    got = _fit(base_df.assign(y=a * base_df["y"]))
    assert_scaled(coef(base), coef(got), a, what="ATT (y scale)")
    assert_scaled(stderr(base), stderr(got), abs(a), what="se (y scale)")


# --------------------------------------------------------------------------- #
# E7 — treated⇄control relabel flips the ATT sign, |ATT| & SE preserved       #
# --------------------------------------------------------------------------- #
def test_did_treatment_relabel_sign_flip(base_df):
    base = _fit(base_df)
    swapped = base_df.copy()
    swapped["group"] = 1 - swapped["group"]
    swapped["treat"] = swapped["group"] * swapped["time"]
    got = _fit(swapped)
    assert_invariant(coef(got), -coef(base), what="ATT (relabel)")
    assert_invariant(stderr(got), stderr(base), what="se (relabel)")


# --------------------------------------------------------------------------- #
# E5 — duplication: point estimate exact, SE shrinks                          #
# --------------------------------------------------------------------------- #
def test_did_duplication_point_invariant(base_df):
    base = _fit(base_df)
    dup = pd.concat(
        [base_df, base_df.assign(id=base_df["id"] + 100000)], ignore_index=True
    )
    got = _fit(dup)
    assert_invariant(coef(base), coef(got), rtol=1e-7, what="ATT (dup)")
    assert stderr(got) < stderr(base), "SE did not shrink under duplication"


# --------------------------------------------------------------------------- #
# E6 — uniform weights identical to unweighted                                #
# --------------------------------------------------------------------------- #
def test_did_uniform_weights_identity(base_df):
    base = _fit(base_df)
    weighted = _fit(base_df.assign(w=1.0), weights="w")
    assert_invariant(coef(base), coef(weighted), what="ATT (w=1)")
    assert_invariant(stderr(base), stderr(weighted), what="se (w=1)")


# --------------------------------------------------------------------------- #
# E8 — covariate column reordering                                            #
# --------------------------------------------------------------------------- #
def test_did_covariate_reorder_invariant(base_df):
    rng = np.random.default_rng(0)
    d = base_df.copy()
    d["c1"] = rng.normal(size=len(d))
    d["c2"] = rng.normal(size=len(d))
    a = _fit(d, covariates=["c1", "c2"])
    b = _fit(d, covariates=["c2", "c1"])
    assert_invariant(coef(a), coef(b), rtol=1e-7, what="ATT (cov reorder)")


# --------------------------------------------------------------------------- #
# Staggered Callaway–Sant'Anna: permutation / scale / shift                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def staggered_df():
    return make_staggered_did(n_units=150, att=2.0, seed=3)


def _cs(d, **kw):
    return sp.callaway_santanna(d, y="y", g="g", t="time", i="id", **kw)


def test_cs_row_permutation_invariant(staggered_df):
    base = _cs(staggered_df)
    perm = staggered_df.sample(frac=1.0, random_state=4).reset_index(drop=True)
    got = _cs(perm)
    assert_invariant(coef(base), coef(got), rtol=1e-7, what="CS ATT (perm)")


def test_cs_outcome_shift_invariant(staggered_df):
    base = _cs(staggered_df)
    got = _cs(staggered_df.assign(y=staggered_df["y"] + 5.0))
    assert_invariant(coef(base), coef(got), rtol=1e-6, what="CS ATT (y shift)")


@pytest.mark.parametrize("a", [2.0, 0.5])
def test_cs_outcome_scale_equivariant(staggered_df, a):
    base = _cs(staggered_df)
    got = _cs(staggered_df.assign(y=a * staggered_df["y"]))
    assert_scaled(coef(base), coef(got), a, rtol=1e-6, what="CS ATT (y scale)")
