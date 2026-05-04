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
# Defensive re-pin: when ``__getattr__`` resolves a leaf whose source module
# *also* exports a same-named function, the function must end up bound on
# ``sp`` after the lazy resolution side-effect.  We exercise the path that
# the ``_LAZY_ATTRS`` defensive branch in ``__init__.__getattr__`` was
# written to protect.
# ---------------------------------------------------------------------------

def test_lazy_leaf_does_not_shadow_same_name_function():
    """Resolving ``sp.bidirectional_pci`` (lazy leaf in
    statspai.proximal) must not cause ``sp.proximal`` to morph into the
    module.  This validates the ``_root_modpath`` re-pin in __getattr__."""
    # Force a fresh state by deleting any cached binding.
    if "bidirectional_pci" in sp.__dict__:
        del sp.__dict__["bidirectional_pci"]
    if "proximal" in sp.__dict__ and isinstance(sp.__dict__["proximal"], types.ModuleType):
        # Restore the function — the module-level eager import already happened.
        from statspai.proximal import proximal as _prox_fn
        sp.__dict__["proximal"] = _prox_fn

    _ = sp.bidirectional_pci  # triggers the lazy resolution path

    assert callable(sp.proximal), (
        "Resolving sp.bidirectional_pci shadowed sp.proximal with the "
        "submodule.  The defensive re-pin in __init__.__getattr__ is broken."
    )
