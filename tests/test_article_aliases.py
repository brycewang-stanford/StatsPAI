"""Tests for the article-facing aliases (sp.rdd / sp.frontdoor / ...).

These tests pin the alias → implementation mapping exposed by
``src/statspai/_article_aliases.py``.  They guard against regressions in
the public README / blog API and ensure the blog snippets actually run.

We deliberately keep the assertions *structural* (right return type,
reasonable magnitude) rather than numerical — exact numerical
correctness is covered by the per-method parity tests in
``tests/reference_parity/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


# ---------------------------------------------------------------------------
# Existence pin: every advertised alias resolves at the top level.
# ---------------------------------------------------------------------------

ADVERTISED_ALIASES = [
    "rdd",
    "frontdoor",
    "xlearner",
    "conformal_ite",
    "psm",
    "partial_identification",
    "anderson_rubin_ci",
    "conditional_lr_ci",
    "tF_adjustment",
]


@pytest.mark.parametrize("name", ADVERTISED_ALIASES)
def test_alias_is_exported(name):
    assert hasattr(sp, name), f"sp.{name} missing from top-level namespace"
    assert callable(getattr(sp, name)), f"sp.{name} is not callable"


def test_alias_in_dunder_all():
    for name in ADVERTISED_ALIASES:
        assert name in sp.__all__, f"{name} missing from sp.__all__"


# ---------------------------------------------------------------------------
# sp.rdd — sharp RD integration smoke test
# ---------------------------------------------------------------------------

def test_rdd_sharp():
    df = sp.dgp_rd(n=500, effect=0.4, cutoff=0.0, seed=7)
    result = sp.rdd(df, y="y", running="x", cutoff=0.0)
    assert isinstance(result, CausalResult)
    assert np.isfinite(result.estimate)
    # the DGP has effect ≈ 0.4; allow a wide band because bandwidth
    # selection + bias correction introduces variance in small samples
    assert abs(result.estimate - 0.4) < 0.5


def test_rdd_matches_rdrobust():
    """sp.rdd should be identical to sp.rdrobust with x/c naming."""
    df = sp.dgp_rd(n=400, effect=0.3, cutoff=0.0, seed=11)
    a = sp.rdd(df, y="y", running="x", cutoff=0.0)
    b = sp.rdrobust(df, y="y", x="x", c=0.0)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-10)


# ---------------------------------------------------------------------------
# sp.xlearner — X-Learner CATE integration smoke test
# ---------------------------------------------------------------------------

def _make_xlearner_df(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.5).astype(int)
    y = 1.0 + 0.5 * X1 + 0.3 * X2 + 0.8 * d + rng.normal(size=n) * 0.5
    return pd.DataFrame({"y": y, "d": d, "x1": X1, "x2": X2})


def test_xlearner_delegates_to_metalearner():
    df = _make_xlearner_df(seed=0)
    result = sp.xlearner(df, y="y", d="d", X=["x1", "x2"])
    assert isinstance(result, CausalResult)
    assert np.isfinite(result.estimate)
    assert "cate" in result.model_info
    assert len(result.model_info["cate"]) == len(df)


def test_xlearner_matches_metalearner_x():
    """Byte-identity with sp.metalearner(..., learner='x') — pins delegation."""
    df = _make_xlearner_df(seed=42)
    a = sp.xlearner(df, y="y", d="d", X=["x1", "x2"])
    b = sp.metalearner(df, y="y", treat="d", covariates=["x1", "x2"], learner="x")
    assert a.estimate == pytest.approx(b.estimate, rel=1e-10)


def test_xlearner_rejects_learner_kwarg():
    df = _make_xlearner_df(seed=1)
    with pytest.raises(TypeError, match="fixed to learner='x'"):
        sp.xlearner(df, y="y", d="d", X=["x1", "x2"], learner="t")


# ---------------------------------------------------------------------------
# sp.psm — propensity-score matching
# ---------------------------------------------------------------------------

def test_psm_nearest_neighbor():
    df = sp.dgp_observational(n=400, seed=3)
    X = [c for c in df.columns if c.startswith("x")]
    result = sp.psm(df, y="y", d="treatment", X=X, method="nn")
    assert isinstance(result, CausalResult)
    assert np.isfinite(result.estimate)


def test_psm_method_alias_equivalence():
    """'nn' and 'nearest' must yield identical estimates."""
    df = sp.dgp_observational(n=300, seed=5)
    X = [c for c in df.columns if c.startswith("x")]
    a = sp.psm(df, y="y", d="treatment", X=X, method="nn")
    b = sp.psm(df, y="y", d="treatment", X=X, method="nearest")
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)


# ---------------------------------------------------------------------------
# sp.conformal_ite — conformal CATE intervals
# ---------------------------------------------------------------------------

def test_conformal_ite_returns_intervals():
    df = sp.dgp_observational(n=400, seed=13)
    X = [c for c in df.columns if c.startswith("x")]
    result = sp.conformal_ite(df, y="y", d="treatment", X=X)
    assert isinstance(result, CausalResult)
    info = result.model_info
    assert "cate_lower" in info and "cate_upper" in info
    lower = np.asarray(info["cate_lower"])
    upper = np.asarray(info["cate_upper"])
    assert lower.shape == upper.shape == (400,)
    assert np.all(upper >= lower)


# ---------------------------------------------------------------------------
# sp.partial_identification — dispatch to bounds
# ---------------------------------------------------------------------------

def test_partial_identification_manski():
    df = sp.dgp_observational(n=300, seed=17)
    # Clip y to [0, 1] so Manski bounds are well-defined.
    df = df.assign(y=df["y"].rank(pct=True))
    result = sp.partial_identification(
        df, y="y", d="treatment", method="manski",
        y_lower=0.0, y_upper=1.0,
    )
    # manski_bounds returns a CausalResult whose model_info carries the
    # interval; check the actual bounds structure, not just __repr__.
    assert isinstance(result, CausalResult)
    info = result.model_info
    assert "lower_bound" in info and "upper_bound" in info
    assert info["lower_bound"] <= info["upper_bound"]


def test_partial_identification_manski_rejects_covariates():
    """Manski bounds are covariate-free; passing X should fail fast."""
    df = sp.dgp_observational(n=100, seed=1)
    with pytest.raises(ValueError, match="does not use"):
        sp.partial_identification(
            df, y="y", d="treatment",
            X=["x1"], method="manski",
        )


def test_partial_identification_horowitz_manski():
    df = sp.dgp_observational(n=300, seed=19)
    df = df.assign(y=df["y"].rank(pct=True))  # bounded outcome
    result = sp.partial_identification(
        df, y="y", d="treatment",
        X=["x1", "x2"],
        method="horowitz_manski",
        y_lower=0.0, y_upper=1.0,
    )
    # horowitz_manski returns a BoundsResult (see bounds/partial_id.py).
    assert hasattr(result, "lower")
    assert hasattr(result, "upper")
    assert result.lower <= result.upper


def test_partial_identification_horowitz_manski_requires_X():
    df = sp.dgp_observational(n=100, seed=1)
    with pytest.raises(ValueError, match="requires a non-empty"):
        sp.partial_identification(
            df, y="y", d="treatment",
            method="horowitz_manski",
        )


def test_partial_identification_lee_requires_selection():
    df = sp.dgp_observational(n=100, seed=1)
    with pytest.raises(ValueError, match="requires"):
        sp.partial_identification(
            df, y="y", d="treatment", method="lee",
        )


def test_partial_identification_iv_requires_instrument():
    df = sp.dgp_iv(n=100, seed=1)
    with pytest.raises(ValueError, match="requires"):
        sp.partial_identification(
            df, y="y", d="treatment", method="iv",
        )


def test_partial_identification_iv_smoke():
    """The IV-bounds branch should actually run end-to-end when given an
    instrument.  We constrain y to [0, 1] because the DGP is unbounded."""
    df = sp.dgp_iv(n=300, seed=5)
    df = df.assign(y=df["y"].rank(pct=True))
    result = sp.partial_identification(
        df, y="y", d="treatment",
        instrument="instrument",
        method="iv",
        # iv_bounds's assumption='monotone_iv' default requires a bounded
        # outcome, which the rank-pct transform gives us.
    )
    assert hasattr(result, "lower")
    assert hasattr(result, "upper")
    assert result.lower <= result.upper


def test_partial_identification_rejects_unknown_method():
    df = sp.dgp_observational(n=100, seed=1)
    with pytest.raises(ValueError, match="Unknown partial_identification method"):
        sp.partial_identification(df, y="y", d="treatment", method="does_not_exist")


# ---------------------------------------------------------------------------
# sp.anderson_rubin_ci / sp.conditional_lr_ci — weak-IV robust CIs
# ---------------------------------------------------------------------------

def test_anderson_rubin_ci_smoke():
    from statspai.iv.weak_iv_ci import WeakIVConfidenceSet
    df = sp.dgp_iv(n=500, seed=21)
    out = sp.anderson_rubin_ci(
        y="y",
        endog="treatment",
        instruments="instrument",
        data=df,
    )
    assert isinstance(out, WeakIVConfidenceSet)
    assert out.method.lower().startswith("anderson") or "AR" in out.method


def test_conditional_lr_ci_smoke():
    from statspai.iv.weak_iv_ci import WeakIVConfidenceSet
    df = sp.dgp_iv(n=500, seed=23)
    out = sp.conditional_lr_ci(
        y="y",
        endog="treatment",
        instruments="instrument",
        data=df,
    )
    assert isinstance(out, WeakIVConfidenceSet)
    # CLR is Moreira (2003)'s Conditional Likelihood Ratio test.
    assert "CLR" in out.method or "conditional" in out.method.lower() or "moreira" in out.method.lower()


# ---------------------------------------------------------------------------
# sp.tF_adjustment — Lee-McCrary-Moreira-Porter (2022) critical value.
# ---------------------------------------------------------------------------

def test_tF_adjustment_monotone():
    """Critical value should weakly decrease as the first-stage F grows."""
    c_small = sp.tF_adjustment(5.0)
    c_medium = sp.tF_adjustment(50.0)
    c_large = sp.tF_adjustment(200.0)
    assert c_small >= c_medium >= c_large
    # At very strong IV the tF cv converges to the usual 1.96.
    assert c_large == pytest.approx(1.96, abs=0.05)


def test_tF_adjustment_matches_underlying():
    from statspai.diagnostics.weak_iv import tF_critical_value
    for F in [5.0, 10.0, 50.0, 200.0]:
        assert sp.tF_adjustment(F) == pytest.approx(tF_critical_value(F), rel=1e-12)


# ---------------------------------------------------------------------------
# P1 coverage property: Anderson-Rubin CI covers the true β under weak IV.
# ---------------------------------------------------------------------------

def test_anderson_rubin_covers_true_beta():
    """AR's *raison d'être* is validity under weak instruments.

    Generate a single moderately-weak IV DGP with known β=0.5, check that
    the AR confidence set contains 0.5 and the CLR set does too.
    """
    rng = np.random.default_rng(2026)
    n = 500
    Z = rng.normal(size=n)
    # First stage with a modest coefficient π = 0.3 (weak-ish but not pathological).
    u = rng.normal(size=n)
    v = rng.normal(size=n)
    D = 0.3 * Z + u + 0.3 * v  # endogeneity via shared v
    beta_true = 0.5
    y = beta_true * D + v + rng.normal(size=n) * 0.3

    df = pd.DataFrame({"y": y, "D": D, "Z": Z})

    ar = sp.anderson_rubin_ci(y="y", endog="D", instruments="Z", data=df)
    clr = sp.conditional_lr_ci(y="y", endog="D", instruments="Z", data=df)

    assert not ar.is_empty, "AR set is empty — numerical failure"
    assert ar.lower - 1e-6 <= beta_true <= ar.upper + 1e-6, (
        f"AR CI [{ar.lower:.3f}, {ar.upper:.3f}] excludes β=0.5"
    )
    assert not clr.is_empty
    assert clr.lower - 1e-6 <= beta_true <= clr.upper + 1e-6, (
        f"CLR CI [{clr.lower:.3f}, {clr.upper:.3f}] excludes β=0.5"
    )
