"""Tier E — invariance / metamorphic tests for ``sp.rdrobust`` (sharp RD).

The RD estimand is the size of the jump in E[y | x] at the cutoff, τ =
lim_{x↓c} − lim_{x↑c}. This gives it a distinctive set of exact invariances:

* **Cutoff translation** (E12) — shifting the running variable and the cutoff
  together leaves τ̂ and SE untouched.
* **Running-variable scaling** — rescaling x (and c) rescales the bandwidth but
  not the jump, so τ̂ is unchanged.
* **Reflection** — x → −x swaps the one-sided limits, flipping the sign of τ̂
  while preserving |τ̂| and SE.
* generic outcome-scale equivariance and row-permutation invariance.

References
----------
- Calonico, S., Cattaneo, M. D. & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*,
  82(6), 2295-2326. [@calonico2014robust]
"""

from __future__ import annotations

import pytest

import statspai as sp

from ._helpers import (
    HAS_HYPOTHESIS,
    assert_invariant,
    assert_scaled,
    coef,
    make_rd,
    stderr,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def base_df():
    return make_rd(n=1500, tau=3.0, cutoff=0.0, seed=7)


# --------------------------------------------------------------------------- #
# E1 — row permutation                                                        #
# --------------------------------------------------------------------------- #
if HAS_HYPOTHESIS:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(parent=settings.get_profile("tier_eg"), max_examples=10)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_rd_row_permutation_invariant(seed):
        d = make_rd(n=900, seed=8)
        base = sp.rdrobust(d, y="y", x="x")
        perm = d.sample(frac=1.0, random_state=seed % (2**31)).reset_index(drop=True)
        got = sp.rdrobust(perm, y="y", x="x")
        assert_invariant(coef(base), coef(got), what="tau (perm)")
        assert_invariant(stderr(base), stderr(got), what="se (perm)")


def test_rd_permutation_explicit(base_df):
    base = sp.rdrobust(base_df, y="y", x="x")
    got = sp.rdrobust(base_df.iloc[::-1].reset_index(drop=True), y="y", x="x")
    assert_invariant(coef(base), coef(got), what="tau")
    assert_invariant(stderr(base), stderr(got), what="se")


# --------------------------------------------------------------------------- #
# E12 — cutoff translation invariance                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta", [5.0, -3.5, 100.0])
def test_rd_cutoff_translation_invariant(base_df, delta):
    base = sp.rdrobust(base_df, y="y", x="x", c=0.0)
    got = sp.rdrobust(base_df.assign(x=base_df["x"] + delta), y="y", x="x", c=delta)
    assert_invariant(coef(base), coef(got), what=f"tau (x+{delta})")
    assert_invariant(stderr(base), stderr(got), what=f"se (x+{delta})")


# --------------------------------------------------------------------------- #
# E (RD-specific) — running-variable scaling invariance                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s", [2.0, 0.25])
def test_rd_running_scale_invariant(base_df, s):
    base = sp.rdrobust(base_df, y="y", x="x", c=0.0)
    got = sp.rdrobust(base_df.assign(x=s * base_df["x"]), y="y", x="x", c=0.0)
    assert_invariant(coef(base), coef(got), rtol=1e-4, what=f"tau (x*{s})")


# --------------------------------------------------------------------------- #
# E3 — outcome-scale equivariance                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a", [4.0, -2.0])
def test_rd_outcome_scale_equivariant(base_df, a):
    base = sp.rdrobust(base_df, y="y", x="x")
    got = sp.rdrobust(base_df.assign(y=a * base_df["y"]), y="y", x="x")
    assert_scaled(coef(base), coef(got), a, rtol=1e-5, what="tau (y scale)")
    assert_scaled(stderr(base), stderr(got), abs(a), rtol=1e-5, what="se (y scale)")


# --------------------------------------------------------------------------- #
# E (RD-specific) — reflection flips the sign of the jump                      #
# --------------------------------------------------------------------------- #
def test_rd_reflection_sign_flip(base_df):
    base = sp.rdrobust(base_df, y="y", x="x", c=0.0)
    got = sp.rdrobust(base_df.assign(x=-base_df["x"]), y="y", x="x", c=0.0)
    assert_invariant(coef(got), -coef(base), rtol=1e-5, what="tau (reflection)")
    assert_invariant(stderr(got), stderr(base), rtol=1e-5, what="se (reflection)")
