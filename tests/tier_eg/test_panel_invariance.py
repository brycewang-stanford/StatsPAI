"""Tier E — invariance / metamorphic tests for ``sp.panel`` (fixed effects).

The within (fixed-effects) estimator is defined by the Frisch–Waugh–Lovell
within transformation, so it satisfies a rich set of exact invariances. We
assert them for the one-way (entity) FE estimator and check the headline ones
also hold for two-way FE.

References
----------
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel
  Data* (2nd ed.), ch. 10 (within estimator). [@wooldridge2010econometric]
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
    make_panel,
    stderr,
)

pytest.importorskip("linearmodels")
pytestmark = pytest.mark.filterwarnings("ignore")


def _df(seed=2, n_units=50, n_periods=6):
    d = make_panel(n_units=n_units, n_periods=n_periods, seed=seed)
    d["x2"] = np.random.default_rng(seed + 99).normal(size=len(d))
    return d


def _fit(data, *, formula="y ~ x + x2", method="fe", **kw):
    return sp.panel(
        data=data,
        formula=formula,
        entity="unit",
        time="period",
        method=method,
        **kw,
    )


@pytest.fixture(scope="module")
def base_df():
    return _df()


# --------------------------------------------------------------------------- #
# E1 — row permutation (hypothesis-driven, exact for FE)                       #
# --------------------------------------------------------------------------- #
if HAS_HYPOTHESIS:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(parent=settings.get_profile("tier_eg"), max_examples=12)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_panel_row_permutation_invariant(seed):
        d = _df(seed=4)
        base = _fit(d)
        perm = d.sample(frac=1.0, random_state=seed % (2**31)).reset_index(drop=True)
        got = _fit(perm)
        assert_invariant(coef(base, "x"), coef(got, "x"), what="beta_x (perm)")
        assert_invariant(stderr(base, "x"), stderr(got, "x"), what="se_x (perm)")


@pytest.mark.parametrize("method", ["fe", "twoway"])
def test_panel_permutation_explicit(method):
    d = _df(seed=8)
    base = _fit(d, method=method)
    perm = d.iloc[::-1].reset_index(drop=True)
    got = _fit(perm, method=method)
    assert_invariant(coef(base, "x"), coef(got, "x"), what=f"beta_x ({method})")
    assert_invariant(stderr(base, "x"), stderr(got, "x"), what=f"se_x ({method})")


# --------------------------------------------------------------------------- #
# E2 — outcome shift absorbed by the within transform                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["fe", "twoway"])
def test_panel_outcome_shift_invariant(base_df, method):
    base = _fit(base_df, method=method)
    got = _fit(base_df.assign(y=base_df["y"] + 13.0), method=method)
    assert_invariant(coef(base, "x"), coef(got, "x"), what="beta_x (y shift)")
    assert_invariant(coef(base, "x2"), coef(got, "x2"), what="beta_x2 (y shift)")


# --------------------------------------------------------------------------- #
# E3 / E4 — outcome & regressor scale equivariance                            #
# --------------------------------------------------------------------------- #
def test_panel_outcome_scale_equivariant(base_df):
    a = 3.0
    base = _fit(base_df)
    got = _fit(base_df.assign(y=a * base_df["y"]))
    assert_scaled(coef(base, "x"), coef(got, "x"), a, what="beta_x")
    assert_scaled(stderr(base, "x"), stderr(got, "x"), a, what="se_x")
    assert_invariant(
        coef(got, "x") / stderr(got, "x"),
        coef(base, "x") / stderr(base, "x"),
        rtol=1e-6,
        what="t-stat",
    )


def test_panel_regressor_scale_equivariant(base_df):
    s = 4.0
    base = _fit(base_df)
    got = _fit(base_df.assign(x=base_df["x"] / s))
    assert_scaled(coef(base, "x"), coef(got, "x"), s, what="beta_x")
    # the other regressor's coefficient is untouched
    assert_invariant(coef(base, "x2"), coef(got, "x2"), what="beta_x2")


# --------------------------------------------------------------------------- #
# E5 — duplication: point estimate exact, SE shrinks                          #
# --------------------------------------------------------------------------- #
def test_panel_duplication_point_invariant(base_df):
    base = _fit(base_df)
    dup = pd.concat(
        [base_df, base_df.assign(unit=base_df["unit"] + 1000)], ignore_index=True
    )
    got = _fit(dup)
    assert_invariant(coef(base, "x"), coef(got, "x"), rtol=1e-7, what="beta_x (dup)")
    assert stderr(got, "x") < stderr(base, "x"), "SE did not shrink under duplication"


# --------------------------------------------------------------------------- #
# E6 — uniform weights identical to unweighted                                #
# --------------------------------------------------------------------------- #
def test_panel_uniform_weights_identity(base_df):
    base = _fit(base_df)
    weighted = _fit(base_df.assign(w=1.0), weights="w")
    assert_invariant(coef(base, "x"), coef(weighted, "x"), what="beta_x (w=1)")
    assert_invariant(coef(base, "x2"), coef(weighted, "x2"), what="beta_x2 (w=1)")


# --------------------------------------------------------------------------- #
# E8 — covariate column reordering                                            #
# --------------------------------------------------------------------------- #
def test_panel_covariate_reorder_invariant(base_df):
    a = _fit(base_df, formula="y ~ x + x2")
    b = _fit(base_df, formula="y ~ x2 + x")
    for term in ["x", "x2"]:
        assert_invariant(coef(a, term), coef(b, term), what=f"coef[{term}]")
        assert_invariant(stderr(a, term), stderr(b, term), what=f"se[{term}]")


# --------------------------------------------------------------------------- #
# E9 — within (FWL) identity: FE slope == OLS on entity-demeaned data         #
# --------------------------------------------------------------------------- #
def test_panel_within_identity(base_df):
    base = _fit(base_df, formula="y ~ x")
    g = base_df.groupby("unit")
    yw = base_df["y"] - g["y"].transform("mean")
    xw = base_df["x"] - g["x"].transform("mean")
    within_beta = float((xw @ yw) / (xw @ xw))
    assert_invariant(coef(base, "x"), within_beta, rtol=1e-7, what="within beta_x")
