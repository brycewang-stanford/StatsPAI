"""Cross-engine validation for StatsPAI.

``sp.cross_validate`` runs one estimand through several *independent* engines
(StatsPAI native, pyfixest, linearmodels, DoubleML, R's fixest via Rscript,
Stata via batch ``do``) and reports whether they agree — turning the
cross-package-reproducibility discipline of Scott Cunningham's "estimate it two
ways and check they match" into a single callable for humans and agents.

Public surface
--------------
- :func:`cross_validate` — the dispatcher.
- :class:`CrossValidationResult` — the verdict + per-engine table.
- :class:`EstimandSpec`, :class:`EngineEstimate`, :class:`TolerancePolicy` —
  building blocks, exposed for advanced use and testing.
"""

from __future__ import annotations

from ._agreement import EngineEstimate, TolerancePolicy
from ._result import CrossValidationResult
from ._spec import EstimandSpec
from .cross_validate import cross_validate

__all__ = [
    "cross_validate",
    "CrossValidationResult",
    "EstimandSpec",
    "EngineEstimate",
    "TolerancePolicy",
]
