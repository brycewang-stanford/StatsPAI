"""Tier E — invariance / metamorphic tests for ``sp.synth`` (classic SCM).

The synthetic-control estimator chooses convex donor weights (summing to one)
to match the treated unit's pre-period path, then reports the post-period gap.
The convexity + sum-to-one constraint give it clean invariances:

* **Outcome location shift** (E2) — a common additive shift cancels in the
  treated-minus-synthetic gap, since the donor weights sum to one.
* **Outcome scale** (E3) — rescaling y rescales the gap (and ATT) by the same
  factor; the optimal weights are unchanged.
* **Donor relabelling** (E13) — the donor unit *labels* carry no information;
  relabelling controls leaves the weights and the gap unchanged.
* generic row-permutation invariance.

Inference is disabled (``placebo=False``) so the tests target the point
estimator deterministically.

References
----------
- Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control methods
  for comparative case studies. *Journal of the American Statistical
  Association*, 105(490), 493-505. [@abadie2010synthetic]
"""

from __future__ import annotations

import pytest

import statspai as sp

from ._helpers import HAS_HYPOTHESIS, assert_invariant, assert_scaled, coef, make_synth

pytestmark = pytest.mark.filterwarnings("ignore")

_BASE = dict(
    outcome="y",
    unit="unit",
    time="time",
    method="classic",
    placebo=False,
)


def _fit(d, **kw):
    return sp.synth(d, treated_unit=0, treatment_time=15, **_BASE, **kw)


@pytest.fixture(scope="module")
def base_df():
    return make_synth(n_donors=18, n_periods=20, treat_period=15, effect=-4.0, seed=1)


# --------------------------------------------------------------------------- #
# E1 — row permutation                                                        #
# --------------------------------------------------------------------------- #
if HAS_HYPOTHESIS:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(parent=settings.get_profile("tier_eg"), max_examples=8)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_synth_row_permutation_invariant(seed):
        d = make_synth(n_donors=12, n_periods=16, treat_period=12, seed=2)
        base = sp.synth(d, treated_unit=0, treatment_time=12, **_BASE)
        perm = d.sample(frac=1.0, random_state=seed % (2**31)).reset_index(drop=True)
        got = sp.synth(perm, treated_unit=0, treatment_time=12, **_BASE)
        assert_invariant(coef(base), coef(got), rtol=1e-5, what="ATT (perm)")


def test_synth_permutation_explicit(base_df):
    base = _fit(base_df)
    got = _fit(base_df.iloc[::-1].reset_index(drop=True))
    assert_invariant(coef(base), coef(got), rtol=1e-5, what="ATT")


# --------------------------------------------------------------------------- #
# E2 — outcome location shift cancels in the gap                              #
# --------------------------------------------------------------------------- #
def test_synth_outcome_shift_invariant(base_df):
    base = _fit(base_df)
    got = _fit(base_df.assign(y=base_df["y"] + 10.0))
    assert_invariant(coef(base), coef(got), rtol=1e-4, what="ATT (y shift)")


# --------------------------------------------------------------------------- #
# E3 — outcome-scale equivariance                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a", [3.0, 0.5])
def test_synth_outcome_scale_equivariant(base_df, a):
    base = _fit(base_df)
    got = _fit(base_df.assign(y=a * base_df["y"]))
    assert_scaled(coef(base), coef(got), a, rtol=1e-4, what="ATT (y scale)")


# --------------------------------------------------------------------------- #
# E13 — donor relabelling invariance                                          #
# --------------------------------------------------------------------------- #
def test_synth_donor_relabel_invariant(base_df):
    relabelled = base_df.copy()
    mask = relabelled["unit"] != 0
    relabelled.loc[mask, "unit"] = relabelled.loc[mask, "unit"] + 1000
    base = _fit(base_df)
    got = _fit(relabelled)
    assert_invariant(coef(base), coef(got), rtol=1e-5, what="ATT (donor relabel)")
