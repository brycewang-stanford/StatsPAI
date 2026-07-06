"""Analytical parity: sp.rate and sp.honest_variance known-truth anchors.

Both aggregate a fitted ``sp.causal_forest``; neither has a cross-package
reference, so the honest grade is ``analytical-only`` — verified against a
known DGP truth and against exact structural identities.

``sp.honest_variance``
    The reported point ``ate`` is *by construction* the sample mean of the
    forest's own CATE predictions, so ``ate == mean(forest.effect(X))``
    holds to machine precision (an exact identity, BLAS-portable because
    both sides use the same predictions). On a homogeneous constant-effect
    DGP the mean CATE recovers the true effect within a bounded band, and
    the half-sample SE / CI are well-formed (SE >= 0, CI brackets the ATE).

``sp.rate`` (Yadlowsky et al. 2023 Rank-Average Treatment Effect)
    AUTOC measures whether the prioritisation score (the forest CATE)
    correctly orders the *true* heterogeneous effect. On a DGP with strong
    heterogeneity ``tau(x) = 1 + 2 x0`` the forest prioritises correctly, so
    AUTOC is positive and its CI excludes zero. On a constant-effect DGP
    there is nothing to prioritise, so AUTOC is ~0 and its CI contains zero.
    RATE/TOC values are not bit-portable across BLAS backends, so the anchor
    is the sign / null structure, not a pinned value.

Forests are seeded (``random_state``) so the test is deterministic.
"""

from __future__ import annotations

import numpy as np

import statspai as sp


def _sim_hte(n, seed, tau_const=None):
    """Heterogeneous (or, with tau_const, homogeneous) treatment DGP."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    tau = np.full(n, tau_const) if tau_const is not None else 1.0 + 2.0 * X[:, 0]
    T = rng.integers(0, 2, size=n)
    Y0 = X[:, 1] + rng.standard_normal(n) * 0.5
    Y = np.where(T == 1, Y0 + tau, Y0)
    return X, T, Y


def test_honest_variance_ate_equals_mean_cate_identity():
    X, T, Y = _sim_hte(n=400, seed=1)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=50, random_state=42)
    hv = sp.honest_variance(cf, X=X, n_splits=25, seed=0)

    mean_cate = float(np.asarray(cf.effect(X)).ravel().mean())
    # Exact structural identity: the honest-variance point estimate IS the
    # mean of the forest's CATE predictions.
    assert abs(hv["ate"] - mean_cate) <= 1e-12
    # Well-formed uncertainty: non-negative SE, CI brackets the point.
    assert hv["se"] >= 0.0
    assert hv["ci_low"] <= hv["ate"] <= hv["ci_high"]


def test_honest_variance_recovers_constant_effect():
    # Homogeneous effect = 3.0: mean CATE must recover it within a band.
    X, T, Y = _sim_hte(n=1500, seed=2, tau_const=3.0)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=100, random_state=0)
    hv = sp.honest_variance(cf, X=X, n_splits=25, seed=0)
    # Forest CATE is a smoothed/regularised estimate; 0.3 absolute band is
    # comfortably tight for a constant-3.0 effect and excludes 0 / 2 / 4.
    assert abs(hv["ate"] - 3.0) <= 0.3


def test_rate_autoc_positive_under_heterogeneity():
    X, T, Y = _sim_hte(n=1200, seed=5)  # tau = 1 + 2 x0, strong HTE
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=200, random_state=0)
    out = sp.rate(cf, X=X, Y=Y, T=T, target="AUTOC")
    # Correct prioritisation ⇒ strictly positive AUTOC whose CI excludes 0.
    assert out["estimate"] > 0.0
    assert out["se"] > 0.0
    assert out["ci_low"] > 0.0


def test_rate_autoc_null_under_constant_effect():
    # No heterogeneity ⇒ no prioritisation value ⇒ AUTOC ~ 0, CI contains 0.
    X, T, Y = _sim_hte(n=1500, seed=2, tau_const=3.0)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=100, random_state=0)
    out = sp.rate(cf, X=X, Y=Y, T=T, target="AUTOC")
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
