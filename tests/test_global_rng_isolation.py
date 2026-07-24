"""Estimators that seed the global RNG must not leak the reseed to the caller.

Several back ends (NOTEARS structure learning, DeepIV, the neural-causal
models, the DML cross-fit engine) call ``np.random.seed(...)`` /
``torch.manual_seed(...)`` because the third-party libraries they wrap read
from the global RNG. The ``preserve_global_rng`` decorator snapshots and
restores the caller's global RNG state around those calls, so a user's own
``np.random`` stream is not silently reset mid-session (CLAUDE.md §7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.utils._rng import preserve_global_rng


def test_decorator_restores_numpy_global_state():
    @preserve_global_rng
    def reseeds():
        np.random.seed(999)
        return np.random.rand()

    np.random.seed(0)
    expected_stream = [np.random.rand() for _ in range(3)]

    np.random.seed(0)
    _ = reseeds()  # internally reseeds to 999
    got_stream = [np.random.rand() for _ in range(3)]

    assert got_stream == expected_stream


def test_decorator_is_transparent_to_return_and_exceptions():
    @preserve_global_rng
    def boom():
        np.random.seed(1)
        raise ValueError("x")

    state_before = np.random.get_state()[1].copy()
    with pytest.raises(ValueError):
        boom()
    # State restored even though the wrapped call raised.
    assert np.array_equal(np.random.get_state()[1], state_before)


def test_notears_does_not_pollute_caller_rng():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    X["b"] += 0.8 * X["a"]
    X["c"] += 0.7 * X["b"]

    np.random.seed(0)
    expected_next = [np.random.rand() for _ in range(2)]

    np.random.seed(0)
    sp.notears(X, random_state=7)
    got_next = [np.random.rand() for _ in range(2)]

    assert got_next == expected_next


def test_notears_determinism_unchanged():
    """Same random_state must still give byte-identical results (the isolation
    changes only the after-effect on the global stream, not the draws)."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    X["b"] += 0.8 * X["a"]
    r1 = sp.notears(X, random_state=3)
    r2 = sp.notears(X, random_state=3)
    assert str(r1) == str(r2)
