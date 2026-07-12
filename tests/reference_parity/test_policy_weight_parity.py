"""Analytical parity: Mogstad-Santos-Torgovitsky policy weight functions.

The MST (2018) framework represents a target parameter as an integral of the
marginal treatment effect against a weight function ``w(u)`` over the
unobservable resistance ``u in [0, 1]``. Each ``sp.policy_weight_*`` builder
returns such a ``w`` with a known closed form:

* ``policy_weight_ate`` is the constant weight ``1`` (integrates to 1).
* ``policy_weight_subsidy(u_lo, u_hi)`` is the indicator of ``[u_lo, u_hi]``
  (integrates to ``u_hi - u_lo``).
* ``policy_weight_marginal(u_star, bw)`` concentrates around ``u_star`` on a
  window of half-width ``bw``.

Analytical evidence tier (closed-form weight identities on a deterministic
grid).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

U = np.linspace(1e-3, 1 - 1e-3, 4000)


def test_ate_weight_is_constant_one():
    w = sp.policy_weight_ate()(U)
    assert np.allclose(w, 1.0)
    assert float(np.trapezoid(w, U)) == pytest.approx(1.0, abs=1e-2)


def test_subsidy_weight_is_window_indicator():
    lo, hi = 0.3, 0.7
    w = sp.policy_weight_subsidy(lo, hi)(U)
    inside = (U >= lo) & (U <= hi)
    assert np.allclose(w[inside], 1.0)
    assert np.allclose(w[~inside], 0.0)
    assert float(np.trapezoid(w, U)) == pytest.approx(hi - lo, abs=1e-2)


def test_marginal_weight_concentrates_at_u_star():
    w = sp.policy_weight_marginal(u_star=0.5, bandwidth=0.05)(U)
    peak_u = float(U[np.argmax(w)])
    assert peak_u == pytest.approx(0.5, abs=0.05)
    # Support lives within the [u_star - bw, u_star + bw] window.
    support = U[w > 0]
    assert support.min() >= 0.5 - 0.05 - 1e-2
    assert support.max() <= 0.5 + 0.05 + 1e-2


def test_marginal_wider_bandwidth_has_wider_support():
    narrow = sp.policy_weight_marginal(u_star=0.5, bandwidth=0.02)(U)
    wide = sp.policy_weight_marginal(u_star=0.5, bandwidth=0.10)(U)
    assert (wide > 0).sum() > (narrow > 0).sum()


def test_observed_prte_weight_is_finite():
    rng = np.random.default_rng(0)
    propensity = rng.uniform(0, 1, 2000)
    w = sp.policy_weight_observed_prte(propensity, shift=0.1)(U)
    assert w.shape == U.shape
    assert np.all(np.isfinite(w))
