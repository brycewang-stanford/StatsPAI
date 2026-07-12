"""Analytical parity: linear-in-means peer effects.

A linear-in-means DGP ``y_i = a + b x_i + g (W x)_i + eps`` (contextual peer
effect only, no endogenous feedback) has known coefficients. ``sp.peer_effects``
recovers the direct effect ``b`` and the contextual peer effect ``g``, and
estimates the endogenous peer effect near zero (there is none). It must also
accept a native StatsPAI ``W`` object (regression guard for the ``W.full()[0]``
first-row coercion bug). Analytical evidence tier (known-truth recovery on a
deterministic DGP).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _linear_in_means(seed=0, n=400, k=5, b=2.0, g=1.5):
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n)])
    W = sp.knn_weights(coords, k=k)
    Wr = np.asarray(W.full(), dtype=float)
    Wr = Wr / Wr.sum(axis=1, keepdims=True)
    x = rng.normal(0, 1, n)
    y = 1.0 + b * x + g * (Wr @ x) + rng.normal(0, 0.3, n)
    return W, pd.DataFrame({"y": y, "x": x})


def test_peer_effects_recovers_direct_and_contextual():
    W, df = _linear_in_means()
    res = sp.peer_effects(df, y="y", covariates=["x"], W=W, include_contextual=True)
    assert float(res.direct["x"]) == pytest.approx(2.0, abs=0.1)
    assert float(res.contextual_peer["x"]) == pytest.approx(1.5, abs=0.2)


def test_peer_effects_no_spurious_endogenous_effect():
    W, df = _linear_in_means()
    res = sp.peer_effects(df, y="y", covariates=["x"], W=W, include_contextual=True)
    # DGP has no endogenous (reflection) feedback term.
    assert abs(float(res.endogenous_peer)) < 0.1


def test_peer_effects_accepts_native_W_object():
    """Regression guard: a native StatsPAI ``W`` must match its dense array
    (the W coercion previously grabbed only the first row)."""
    W, df = _linear_in_means()
    r_native = sp.peer_effects(df, y="y", covariates=["x"], W=W)
    r_array = sp.peer_effects(
        df, y="y", covariates=["x"], W=np.asarray(W.full())
    )
    np.testing.assert_allclose(
        r_native.coefficients["coef"].to_numpy(),
        r_array.coefficients["coef"].to_numpy(),
        rtol=1e-10,
    )
