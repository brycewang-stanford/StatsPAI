"""Tests for the v1.10 ``causal/`` → ``forest/`` rename + deprecation shim.

The package was renamed because ``causal/`` only ever housed four
forest-based estimators (CausalForest / IVForest / MultiArmForest +
forest inference helpers).  The old name is kept as a deprecation
shim for one minor cycle.

Guard the four contracts:

  1. ``statspai.forest`` is the canonical name and exposes the full
     public surface.
  2. ``statspai.causal`` still works (back-compat) but emits a
     :class:`DeprecationWarning`.
  3. Both top-level (``from statspai.causal import X``) and
     submodule (``from statspai.causal.causal_forest import X``)
     imports keep resolving — submodule shims preserve the second
     form.
  4. ``sp.causal_forest`` / ``sp.CausalForest`` etc. at the
     top-level continue to work unchanged.
"""
from __future__ import annotations

import warnings

import pytest

import statspai as sp


# ─── Canonical: statspai.forest ─────────────────────────────────────────


def test_forest_subpackage_exists():
    import statspai.forest as forest
    assert forest.__name__ == "statspai.forest"


def test_forest_public_surface():
    from statspai.forest import (
        CausalForest, causal_forest,
        calibration_test, test_calibration, rate, honest_variance,
        multi_arm_forest, MultiArmForestResult,
        iv_forest, IVForestResult,
    )
    assert callable(CausalForest)
    assert callable(causal_forest)
    assert callable(calibration_test)
    assert callable(iv_forest)


def test_forest_submodule_paths():
    from statspai.forest.causal_forest import CausalForest
    from statspai.forest.forest_inference import (
        calibration_test, test_calibration, rate, honest_variance,
    )
    from statspai.forest.iv_forest import iv_forest, IVForestResult
    from statspai.forest.multi_arm_forest import (
        multi_arm_forest, MultiArmForestResult,
    )
    assert callable(CausalForest)
    assert callable(iv_forest)


# ─── Top-level sp.X passthroughs unchanged ──────────────────────────────


def test_top_level_names_unchanged():
    assert callable(sp.causal_forest)
    assert callable(sp.CausalForest)
    assert callable(sp.iv_forest)
    assert callable(sp.multi_arm_forest)
    assert callable(sp.calibration_test)
    assert callable(sp.test_calibration)
    assert callable(sp.rate)
    assert callable(sp.honest_variance)


# ─── Deprecation shim: statspai.causal ──────────────────────────────────


def test_causal_package_emits_deprecation_warning():
    """Importing ``statspai.causal`` should emit a DeprecationWarning."""
    import importlib
    # Force a fresh import path: drop the cached module then re-import.
    import sys
    sys.modules.pop("statspai.causal", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import statspai.causal  # noqa: F401
    deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)
                    and "statspai.causal" in str(x.message)]
    assert deprecations, (
        f"Expected DeprecationWarning mentioning statspai.causal; got: "
        f"{[str(x.message) for x in w]}"
    )


def test_causal_top_level_back_compat_imports():
    """``from statspai.causal import X`` must still work."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from statspai.causal import (
            CausalForest, causal_forest,
            calibration_test, test_calibration, rate, honest_variance,
            multi_arm_forest, MultiArmForestResult,
            iv_forest, IVForestResult,
        )
    assert callable(CausalForest)
    assert callable(causal_forest)


def test_causal_submodule_back_compat_imports():
    """``from statspai.causal.causal_forest import X`` must still work."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from statspai.causal.causal_forest import CausalForest
        from statspai.causal.forest_inference import (
            calibration_test, test_calibration, rate, honest_variance,
        )
        from statspai.causal.iv_forest import iv_forest, IVForestResult
        from statspai.causal.multi_arm_forest import (
            multi_arm_forest, MultiArmForestResult,
        )
    assert callable(CausalForest)
    assert callable(iv_forest)


def test_causal_and_forest_export_same_objects():
    """Identity check — the shim re-exports, doesn't reimplement."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import statspai.causal as old
        import statspai.forest as new
    assert old.CausalForest is new.CausalForest
    assert old.causal_forest is new.causal_forest
    assert old.iv_forest is new.iv_forest
    assert old.multi_arm_forest is new.multi_arm_forest
    assert old.calibration_test is new.calibration_test
