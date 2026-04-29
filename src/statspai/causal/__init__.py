"""
DEPRECATED — use :mod:`statspai.forest` instead.

This package was renamed in v1.10 to ``statspai.forest`` because it
only ever housed the four forest-based causal estimators
(``causal_forest`` / ``iv_forest`` / ``multi_arm_forest`` / forest
inference helpers).  The ``causal`` name was misleading — it
implied a top-level causal-inference namespace when in fact the
content is a single-method family.

This shim keeps both ``from statspai.causal import X`` and
``from statspai.causal.causal_forest import X`` working by aliasing
the deprecated submodule paths through :data:`sys.modules` to the
real :mod:`statspai.forest` modules.  Users see one
:class:`DeprecationWarning` on first import; everything else
continues working.

Plan to migrate within one minor cycle.
"""
from __future__ import annotations

import sys
import warnings

warnings.warn(
    "statspai.causal is deprecated and will be removed in a future "
    "release; use statspai.forest instead "
    "(e.g. `from statspai.forest import CausalForest`).",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public names (functions / classes) at the package level.
# This also has the side effect of loading the ``statspai.forest.*``
# submodules into :data:`sys.modules` (forest/__init__.py does
# ``from .causal_forest import ...`` etc.).
from ..forest import (  # noqa: F401
    CausalForest, causal_forest,
    calibration_test, test_calibration, rate, honest_variance,
    multi_arm_forest, MultiArmForestResult,
    iv_forest, IVForestResult,
)

# Alias the deprecated submodule paths to the real forest modules
# *without* re-importing them (which would shadow the function names
# with the submodule attributes — the same trap that broke ``sp.iv``).
# After ``from ..forest import ...`` above, every submodule is already
# in sys.modules; we just point the deprecated paths at the same
# objects.
sys.modules["statspai.causal.causal_forest"] = (
    sys.modules["statspai.forest.causal_forest"]
)
sys.modules["statspai.causal.forest_inference"] = (
    sys.modules["statspai.forest.forest_inference"]
)
sys.modules["statspai.causal.iv_forest"] = (
    sys.modules["statspai.forest.iv_forest"]
)
sys.modules["statspai.causal.multi_arm_forest"] = (
    sys.modules["statspai.forest.multi_arm_forest"]
)


__all__ = [
    "CausalForest",
    "causal_forest",
    "calibration_test",
    "test_calibration",
    "rate",
    "honest_variance",
    "multi_arm_forest", "MultiArmForestResult",
    "iv_forest", "IVForestResult",
]
