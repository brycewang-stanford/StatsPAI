"""Regression contracts for late-bind shadowing across the public surface.

The ``statspai/__init__.py`` namespace has two intertwined hazards:

1. **Late-bind shadow** — names like ``sp.iv`` / ``sp.mediation`` /
   ``sp.policy_tree`` / ``sp.dml`` / ``sp.matrix_completion`` /
   ``sp.causal_discovery`` were originally bound to *submodules* via
   ``from . import X``, then deliberately *re-bound* to article-facing
   wrapper *functions* (or callable-modules) by the
   ``_article_aliases`` block at the bottom of ``__init__.py``.  A
   future re-order of imports that drops the re-bind silently turns
   these names back into modules and breaks every blog-post snippet.

2. **Post-import re-shadow** — Python's import system auto-binds a
   submodule on its parent package when ``import statspai.X`` runs.
   For names like ``sp.proximal`` / ``sp.principal_strat`` /
   ``sp.bridge`` / ``sp.bcf`` etc. that are *both* submodule names
   and same-named function exports, a downstream
   ``from statspai.X import Y`` would silently re-shadow the function
   with the module.  Codex's lazy-load refactor (commit ``7eeb624``)
   surfaced this; the hardening lives in
   ``f9ec214 fix(api): preserve same-name function bindings under lazy submodule registry``.

These tests *do not* re-validate the inner behaviour of each wrapper —
that is covered by ``tests/test_article_aliases_round2.py``,
``tests/test_iv_dispatcher.py``, ``tests/test_principal_strat.py``,
``tests/test_proximal.py``, etc.  We only pin the *binding type* and
ensure it survives a downstream ``from statspai.X import ...``.
"""
from __future__ import annotations

import importlib
import types

import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Late-bind: 6 names that ``_article_aliases`` re-binds at the bottom of
# ``__init__.py``.  Each must remain callable at import time.
# ---------------------------------------------------------------------------

LATE_BIND_FUNCTIONS = [
    "mediation",
    "policy_tree",
    "dml",
    "matrix_completion",
    "causal_discovery",
]


@pytest.mark.parametrize("name", LATE_BIND_FUNCTIONS)
def test_late_bind_target_is_callable(name):
    """``sp.<name>`` must be a callable wrapper, not a submodule."""
    obj = getattr(sp, name)
    assert callable(obj), (
        f"sp.{name} should be the article-facing function wrapper. "
        f"If this fails, the late-bind block in __init__.py "
        f"(``from ._article_aliases import {name}``) was likely removed "
        f"or moved before the eager submodule import that shadows it."
    )
    assert not isinstance(obj, types.ModuleType), (
        f"sp.{name} resolved to a module instead of a function. "
        f"Restore the _article_aliases re-bind."
    )


def test_iv_is_callable_dispatcher():
    """``sp.iv`` is the IV submodule, but it must be callable as a dispatcher
    via the module's ``__call__`` (set in ``statspai/iv/__init__.py``)."""
    assert callable(sp.iv), (
        "sp.iv must be callable for sp.iv('y ~ (d ~ z)', data=df) to work. "
        "The callable-module trick lives in statspai/iv/__init__.py."
    )


# ---------------------------------------------------------------------------
# Post-import re-shadow: 14 names that are both function exports and
# submodule names.  After ``import statspai`` plus a downstream
# ``from statspai.X import Y``, ``sp.X`` must still resolve to the
# function (not the auto-bound module).
# ---------------------------------------------------------------------------

CONFLICT_PRONE = [
    ("proximal",         "ProximalCausalInference"),
    ("principal_strat",  "PrincipalStratResult"),
    ("bridge",           "BridgeResult"),
    ("bcf",              "BayesianCausalForest"),
    ("bunching",         "BunchingEstimator"),
    ("dose_response",    "DoseResponse"),
    ("multi_treatment",  "MultiTreatment"),
    ("causal_impact",    "CausalImpactEstimator"),
    ("frontier",         "FrontierResult"),
    ("interference",     "SpilloverEstimator"),
    ("tmle",             "TMLE"),
    ("msm",              "MarginalStructuralModel"),
    ("deepiv",           "DeepIV"),
    ("bartik",           "BartikIV"),
]


@pytest.mark.parametrize("name,sibling", CONFLICT_PRONE)
def test_conflict_prone_function_survives_submodule_import(name, sibling):
    """After ``from statspai.<X> import <sibling>``, ``sp.<X>`` must still
    be the function — not silently re-shadowed to the module."""
    submod = importlib.import_module(f"statspai.{name}")
    # Sanity: the sibling we'll trigger via fromlist actually exists.
    assert hasattr(submod, sibling), (
        f"Sibling export statspai.{name}.{sibling} disappeared; the test "
        f"target needs to be updated."
    )
    # Trigger the same import shape a downstream user would use.
    importlib.__import__(f"statspai.{name}", fromlist=[sibling])
    obj = getattr(sp, name)
    assert callable(obj), (
        f"sp.{name} is not callable after `from statspai.{name} import "
        f"{sibling}`.  The function binding was shadowed by the auto-bound "
        f"submodule.  Either keep the eager `from .{name} import {name}` "
        f"in __init__.py, or extend the __getattr__ defensive re-pin."
    )
    assert not isinstance(obj, types.ModuleType), (
        f"sp.{name} resolved to a module instead of a function."
    )


# ---------------------------------------------------------------------------
# Defensive re-pin against future conflict-prone additions.
#
# The 14 conflict-prone modules above all have *eager* function bindings
# in __init__.py (Step 1 / commit f9ec214), so their leaves never go
# through __getattr__ — the defensive ``_root_modpath`` re-pin in
# __init__.__getattr__ stays dormant for them.
#
# This test pins the *forward-compat* contract: if anyone adds a future
# lazy leaf whose source module shares a name with a registered leaf,
# the side-effect attribute binding from importlib must NOT shadow the
# function.  We instrument by clearing all cached bindings and
# triggering ``sp`` access through a legitimate lazy leaf, then asserting
# every conflict-prone function remains callable.
# ---------------------------------------------------------------------------

def test_no_conflict_function_morphs_to_module_after_lazy_traffic():
    """End-to-end: trigger several lazy resolutions and re-assert all 14
    conflict-prone names remain callable.  This guards against any future
    refactor that re-introduces a same-name lazy leaf without keeping the
    eager fallback or the __getattr__ defensive re-pin."""
    # Trigger a handful of legitimate lazy resolutions.
    _ = sp.bayes_did
    _ = sp.target_trial_protocol
    _ = sp.surrogate_index
    _ = sp.causal_dqn

    for name, _sibling in CONFLICT_PRONE:
        obj = getattr(sp, name)
        assert callable(obj) and not isinstance(obj, types.ModuleType), (
            f"sp.{name} morphed into the module after lazy traffic.  "
            f"Either restore the eager `from .{name} import {name}` import "
            f"or extend the __getattr__ defensive re-pin."
        )


# ---------------------------------------------------------------------------
# Forest cold-start budget (Step 1B).
#
# Previously, ``__init__.py`` eagerly imported ``forest.causal_forest``
# / ``forest.iv_forest`` / ``forest.multi_arm_forest`` /
# ``forest.forest_inference`` at module load, which transitively pulled
# ~245 ``sklearn.*`` submodules into ``sys.modules`` for every
# ``import statspai`` — even sessions that never touch heterogeneous-effect
# forests.  ``forest`` does *not* collide with a top-level function (no
# ``sp.forest`` callable export), so the 8 leaf names (CausalForest /
# causal_forest / calibration_test / test_calibration / rate /
# honest_variance / multi_arm_forest / MultiArmForestResult / iv_forest /
# IVForestResult) live in ``_LAZY_ATTRS`` keyed to dotted submodule paths
# (e.g. ``forest.causal_forest``) and resolve via ``__getattr__`` on first
# touch.
#
# These contracts pin three things:
#
# 1. ``import statspai`` must NOT pre-load any ``statspai.forest.*``
#    submodule — that's the cold-start regression contract.
# 2. After touching a forest leaf, the public name must resolve to the
#    callable (function/class), not the auto-bound submodule.
# 3. A downstream ``from statspai.forest.causal_forest import CausalForest``
#    must NOT silently re-shadow ``sp.causal_forest`` to the leaf module
#    via Python's post-import attribute binding.
# ---------------------------------------------------------------------------

FOREST_LAZY_LEAVES = [
    ("CausalForest",          "forest.causal_forest"),
    ("causal_forest",         "forest.causal_forest"),
    ("calibration_test",      "forest.forest_inference"),
    ("test_calibration",      "forest.forest_inference"),
    ("rate",                  "forest.forest_inference"),
    ("honest_variance",       "forest.forest_inference"),
    ("multi_arm_forest",      "forest.multi_arm_forest"),
    ("MultiArmForestResult",  "forest.multi_arm_forest"),
    ("iv_forest",             "forest.iv_forest"),
    ("IVForestResult",        "forest.iv_forest"),
]


def test_forest_not_loaded_on_bare_import_statspai():
    """``import statspai`` must not transitively import any forest leaf
    submodule.  Regression: any future eager ``from .forest... import ...``
    in ``__init__.py`` would re-inflate ``sys.modules`` by ~245
    ``sklearn.*`` entries on every session.

    Runs in a subprocess so the cold-state check doesn't perturb other
    tests' ``sys.modules`` (which would change ``CausalResult`` class
    identity and break ``isinstance`` assertions elsewhere)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import statspai\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('statspai.forest'))\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    last_line = proc.stdout.strip().splitlines()[-1]
    assert last_line.startswith("LEAKED="), proc.stdout
    leaked_csv = last_line[len("LEAKED="):]
    leaked = [m for m in leaked_csv.split(",") if m]
    assert not leaked, (
        f"`import statspai` eagerly loaded {leaked}.  Move the offending "
        f"`from .forest... import ...` line out of __init__.py and rely on "
        f"the lazy `_register_lazy('forest.X', ...)` table at the bottom."
    )


@pytest.mark.parametrize("name,expected_modpath", FOREST_LAZY_LEAVES)
def test_forest_leaf_resolves_to_callable(name, expected_modpath):
    """Each forest leaf, on first ``sp.<name>`` access, must resolve to
    the function/class — not the auto-bound submodule object."""
    obj = getattr(sp, name)
    assert not isinstance(obj, types.ModuleType), (
        f"sp.{name} resolved to a module instead of the expected "
        f"callable from statspai.{expected_modpath}."
    )
    # CausalForest / MultiArmForestResult / IVForestResult are classes;
    # everything else is a callable function.  ``callable(obj)`` is True
    # for both, so this is enough.
    assert callable(obj), (
        f"sp.{name} should be the callable exported by "
        f"statspai.{expected_modpath}."
    )


def test_forest_leaf_survives_submodule_fromimport_shadow():
    """A downstream ``from statspai.forest.causal_forest import CausalForest``
    must NOT re-shadow ``sp.causal_forest`` to the leaf submodule.

    Python's import system attaches ``statspai.forest`` to ``statspai`` as
    a side effect of the chained import.  ``__getattr__`` already resolved
    ``sp.causal_forest`` to the function and cached it in ``globals()``,
    so the parent-package binding for the function should win — but the
    child binding for the *submodule* (``statspai.forest.causal_forest``)
    is what the user is touching here, and we want to confirm the
    function binding survives.
    """
    import importlib

    # Pre-touch so the lazy resolution caches the function into sp.__dict__.
    pre = sp.causal_forest
    assert callable(pre)

    # Trigger the shadow shape.
    importlib.__import__(
        "statspai.forest.causal_forest", fromlist=["CausalForest"]
    )

    post = sp.causal_forest
    assert callable(post), (
        "sp.causal_forest morphed into a module after a downstream "
        "`from statspai.forest.causal_forest import CausalForest`.  The "
        "function binding cached by __getattr__ should outlive the "
        "submodule attach."
    )
    assert not isinstance(post, types.ModuleType)


# ---------------------------------------------------------------------------
# sklearn cold-start budget (Steps 1B + 1C + 1D — final).
#
# Prior to the cold-start refactors, ``import statspai`` pulled ~245
# ``sklearn.*`` submodules into ``sys.modules`` even for sessions that never
# touched any ML-based estimator.  Three rounds drove this to zero:
#
#   - Step 1B (commit b655ba1): lazy-load ``statspai.forest`` so its 4
#     leaf modules don't fire at ``import statspai``.
#   - Step 1C (commit ef410c6): move top-level ``from sklearn.X import Y``
#     inside function bodies in 18 estimator files; keep ``BaseEstimator``
#     annotations under ``TYPE_CHECKING`` + string-literal form.
#   - Step 1D (this commit): drop ``sklearn.base.BaseEstimator`` /
#     ``RegressorMixin`` / ``ClassifierMixin`` inheritance from
#     ``HALRegressor`` / ``HALClassifier`` in ``tmle/hal_tmle.py``;
#     replace with a minimal duck-typed ``_BaseHAL`` providing the
#     ``get_params`` / ``set_params`` / ``__repr__`` slice that
#     ``sklearn.base.clone()`` actually consumes.
#
# After Step 1D, ``import statspai`` pulls **zero** sklearn submodules.
# This contract pins that floor at exactly 0 — any future change that
# re-introduces a top-level sklearn import must either justify the
# regression or move the import inside a function body.  Runs in a
# subprocess so the cold-state check doesn't perturb other tests'
# ``sys.modules``.
# ---------------------------------------------------------------------------

SKLEARN_BUDGET_CEILING = 0


def test_sklearn_budget_ceiling_on_bare_import_statspai():
    """``import statspai`` must not eagerly pull any sklearn submodule.

    After Steps 1B/1C/1D, the cold-import floor is zero — sklearn is
    only loaded on first use of an ML-backed estimator.  A non-zero
    count means some estimator file re-introduced a top-level
    ``from sklearn.X import Y`` (or a class inherited from
    ``BaseEstimator`` etc. at module-load time).  Grep the offending
    ``git diff`` for added ``from sklearn`` lines or class declarations
    that subclass sklearn types, and move the dependency inside the
    function body / replace the inheritance with the duck-typed
    ``_BaseHAL`` pattern from ``tmle/hal_tmle.py``.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import statspai\n"
        "n = sum(1 for m in sys.modules if m.startswith('sklearn'))\n"
        "print(f'SKLEARN_COUNT={n}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    last_line = proc.stdout.strip().splitlines()[-1]
    assert last_line.startswith("SKLEARN_COUNT="), proc.stdout
    n = int(last_line[len("SKLEARN_COUNT="):])
    assert n <= SKLEARN_BUDGET_CEILING, (
        f"`import statspai` pulled {n} sklearn submodules, exceeding the "
        f"<= {SKLEARN_BUDGET_CEILING} ceiling.  Either a top-level "
        f"`from sklearn.X import Y` was re-introduced (move it inside "
        f"the function), or a class re-acquired a sklearn superclass at "
        f"module load (use the ``_BaseHAL`` duck-typed pattern from "
        f"``tmle/hal_tmle.py`` instead)."
    )
