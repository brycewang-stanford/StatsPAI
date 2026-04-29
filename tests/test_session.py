"""Tests for ``sp.session(seed=...)`` deterministic-RNG context manager.

Pins: determinism (same seed → same draws), restoration (state outside
the block is untouched), snapshot-only mode (``seed=None``), and JAX /
Torch interop being lazy (no auto-import).
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
#  Top-level export
# ---------------------------------------------------------------------------


class TestExport:

    def test_callable(self):
        assert callable(sp.session)

    def test_in_all(self):
        assert "session" in sp.__all__


# ---------------------------------------------------------------------------
#  Determinism: identical seeds → identical RNG draws
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_numpy_legacy_global_is_seeded(self):
        with sp.session(seed=42):
            a = np.random.randn(5)
        with sp.session(seed=42):
            b = np.random.randn(5)
        assert np.array_equal(a, b)

    def test_python_random_is_seeded(self):
        with sp.session(seed=42):
            a = [random.random() for _ in range(5)]
        with sp.session(seed=42):
            b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_diverge(self):
        with sp.session(seed=1):
            a = np.random.randn(5)
        with sp.session(seed=2):
            b = np.random.randn(5)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
#  Restoration: state outside the block is untouched
# ---------------------------------------------------------------------------


class TestRestoration:

    def test_numpy_global_state_restored(self):
        np.random.seed(99)
        before = np.random.get_state()
        with sp.session(seed=42):
            np.random.randn(10)  # mutate inside the block
        after = np.random.get_state()
        # State tuple: compare each element since arrays don't ``==``
        # cleanly.
        for x, y in zip(before, after):
            if hasattr(x, "__len__") and not isinstance(x, str):
                assert np.array_equal(x, y)
            else:
                assert x == y

    def test_python_random_state_restored(self):
        random.seed(99)
        before = random.getstate()
        with sp.session(seed=42):
            random.random()
        after = random.getstate()
        assert before == after

    def test_state_restored_even_after_exception_inside_block(self):
        # Regression guard for the try/finally restoration contract:
        # an exception raised inside the ``with`` body must not
        # prevent the snapshot from being restored on exit.
        np.random.seed(99)
        before = np.random.get_state()
        with pytest.raises(ValueError, match="boom"):
            with sp.session(seed=42):
                np.random.randn(50)
                raise ValueError("boom")
        after = np.random.get_state()
        for x, y in zip(before, after):
            if hasattr(x, "__len__") and not isinstance(x, str):
                assert np.array_equal(x, y)
            else:
                assert x == y

    def test_post_session_draws_match_prior_seed(self):
        # Seed 99, draw twice. Expected: the two draws (across a
        # session in between) should be identical to draws from a
        # fresh seed=99 run that sees no session at all.
        np.random.seed(99)
        np.random.randn(3)  # advance to a known mid-stream state
        with sp.session(seed=42):
            np.random.randn(50)  # mutate inside, but state is restored
        post_session = np.random.randn(3)

        np.random.seed(99)
        np.random.randn(3)
        baseline = np.random.randn(3)
        assert np.array_equal(post_session, baseline)


# ---------------------------------------------------------------------------
#  ``seed=None`` snapshot-only mode
# ---------------------------------------------------------------------------


class TestDefaultRngScope:
    """``np.random.default_rng()`` is NOT covered by ``sp.session`` —
    documenting that contract here so a future "fix" doesn't
    accidentally seed PCG64 globally without warning."""

    def test_default_rng_inside_session_is_not_seeded(self):
        with sp.session(seed=42):
            a = np.random.default_rng().normal(size=5)
        with sp.session(seed=42):
            b = np.random.default_rng().normal(size=5)
        # Two independent default_rng() calls inside same-seed
        # sessions must NOT be forced to be equal — they pull from
        # OS entropy.
        assert not np.array_equal(a, b)

    def test_explicit_default_rng_seed_is_deterministic(self):
        # The documented escape hatch: pass state.seed to default_rng.
        with sp.session(seed=42) as state:
            a = np.random.default_rng(state.seed).normal(size=5)
        with sp.session(seed=42) as state:
            b = np.random.default_rng(state.seed).normal(size=5)
        assert np.array_equal(a, b)


class TestSeedNone:

    def test_snapshot_only_no_reseed(self):
        np.random.seed(7)
        with sp.session(seed=None):
            inside = np.random.randn(3)
        np.random.seed(7)
        outside = np.random.randn(3)
        # ``seed=None`` means we keep the prior global state, so the
        # draws inside should equal a fresh seed=7 stream.
        assert np.array_equal(inside, outside)

    def test_snapshot_only_still_restores(self):
        np.random.seed(7)
        np.random.randn(2)  # advance global state
        snapshot = np.random.get_state()
        with sp.session(seed=None):
            np.random.randn(50)
        after = np.random.get_state()
        for x, y in zip(snapshot, after):
            if hasattr(x, "__len__") and not isinstance(x, str):
                assert np.array_equal(x, y)
            else:
                assert x == y


# ---------------------------------------------------------------------------
#  Lazy backend interop — torch / jax never auto-imported
# ---------------------------------------------------------------------------


class TestLazyBackends:

    def test_session_does_not_import_torch(self):
        was_loaded_before = "torch" in sys.modules
        with sp.session(seed=42):
            pass
        if not was_loaded_before:
            assert "torch" not in sys.modules, (
                "sp.session must not trigger torch import")

    def test_session_does_not_import_jax(self):
        was_loaded_before = "jax" in sys.modules
        with sp.session(seed=42):
            pass
        if not was_loaded_before:
            assert "jax" not in sys.modules, (
                "sp.session must not trigger jax import")


# ---------------------------------------------------------------------------
#  PYTHONHASHSEED
# ---------------------------------------------------------------------------


class TestPythonHashSeed:

    def test_off_by_default(self):
        prior = os.environ.get("PYTHONHASHSEED")
        with sp.session(seed=42):
            inside = os.environ.get("PYTHONHASHSEED")
        assert inside == prior

    def test_opt_in_sets_and_restores(self):
        prior = os.environ.get("PYTHONHASHSEED")
        os.environ.pop("PYTHONHASHSEED", None)
        with sp.session(seed=42, pythonhashseed=True):
            assert os.environ["PYTHONHASHSEED"] == "42"
        assert "PYTHONHASHSEED" not in os.environ
        # Restore the global the test runner inherited.
        if prior is not None:
            os.environ["PYTHONHASHSEED"] = prior


# ---------------------------------------------------------------------------
#  Yielded SessionState
# ---------------------------------------------------------------------------


class TestSessionState:

    def test_yields_state_with_seed(self):
        with sp.session(seed=42) as state:
            assert state.seed == 42

    def test_state_is_an_object_not_a_dict(self):
        # Tests can hold a reference; agents can stash it for
        # downstream calls (e.g. JAX key plumbing).
        with sp.session(seed=42) as state:
            assert hasattr(state, "seed")
            assert hasattr(state, "jax_key")
