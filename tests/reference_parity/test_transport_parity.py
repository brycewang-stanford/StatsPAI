"""Analytical parity: transportability / generalizability of a trial effect.

``sp.transport_generalize`` reweights an RCT to a target population. Two
known-truth checks pin it:

* Under a *homogeneous* treatment effect, the transported effect equals the
  source effect (and the true constant) regardless of covariate shift — there
  is nothing to reweight.
* Under a *heterogeneous* effect that rises in ``x``, transporting to a target
  shifted toward high ``x`` moves the estimate strictly above the source
  effect.

Analytical evidence tier (known-truth recovery / ordering on deterministic
DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _rct(seed, effect_fn, n=3000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    treat = (rng.random(n) < 0.5).astype(int)
    y = 1.0 + 0.5 * x + effect_fn(x) * treat + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "treat": treat, "y": y})


def _target(seed, loc=2.0, n=3000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(loc, 1, n)})


def test_transport_homogeneous_effect_is_invariant():
    tau = 3.0
    rct = _rct(0, lambda x: np.full_like(x, tau))
    target = _target(1)
    res = sp.transport_generalize(
        rct, target, features=["x"], treatment="treat", outcome="y"
    )
    # Nothing to reweight: source, transported, and truth all coincide.
    assert float(res.effect_source) == pytest.approx(tau, abs=0.3)
    assert float(res.effect_transported) == pytest.approx(tau, abs=0.4)


def test_transport_shifts_toward_target_under_heterogeneity():
    # tau(x) = 1 + x: source (x~0) has mean effect ~1; target (x~2) ~3.
    rct = _rct(0, lambda x: 1.0 + x)
    target = _target(1, loc=2.0)
    res = sp.transport_generalize(
        rct, target, features=["x"], treatment="treat", outcome="y"
    )
    assert float(res.effect_source) == pytest.approx(1.0, abs=0.3)
    # Transport pulls the estimate up toward the target population's effect.
    assert float(res.effect_transported) > float(res.effect_source) + 1.0


def test_transport_effective_sample_size_positive():
    rct = _rct(0, lambda x: 1.0 + x)
    target = _target(1, loc=2.0)
    res = sp.transport_generalize(
        rct, target, features=["x"], treatment="treat", outcome="y"
    )
    assert 0.0 < float(res.ess) <= len(rct)
