"""
Unified exception taxonomy for StatsPAI.

Every exception in this module carries a ``recovery_hint`` that tells
agents (and humans) how to react — not just *what* went wrong. The
taxonomy exists so that agent code can ``except`` on a *kind of
failure* (identification, convergence, data, numerical stability)
without having to pattern-match on error messages.

Hierarchy
---------

::

    StatsPAIError
    ├── AssumptionViolation          # parallel trends, exclusion, overlap, …
    │   └── IdentificationFailure    # stronger: no valid estimand
    ├── DataInsufficient             # n too small, too few clusters, thin support
    ├── ConvergenceFailure           # optimizer / MCMC / EM did not converge
    ├── NumericalInstability         # singular design, near-zero variance, NaN
    └── MethodIncompatibility        # method ≠ data design / options conflict

Each instance exposes:

* ``recovery_hint`` — short, actionable instruction
* ``diagnostics`` — dict of machine-readable details
* ``alternative_functions`` — ranked list of ``sp.xxx`` alternatives

Agent-oriented example
----------------------

>>> from statspai.exceptions import AssumptionViolation
>>> try:
...     result = sp.did(df, y="y", treat="t", time="p")
... except AssumptionViolation as e:
...     print(e.recovery_hint)           # "Run sp.pretrends_test(...)"
...     print(e.diagnostics)              # {"test": "pretrends", "pvalue": 0.003}
...     for alt in e.alternative_functions:
...         print(alt)                    # "sp.callaway_santanna", ...

Note
----
Most estimators in StatsPAI **still raise** ``ValueError`` /
``RuntimeError`` for historical reasons. This module is intentionally
additive — new code should raise a ``StatsPAIError`` subclass, and old
code will be migrated to the taxonomy without breaking the old catches
(``ValueError`` subclasses are preserved; see each class below).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional


class StatsPAIError(Exception):
    """Base class for all StatsPAI exceptions.

    Agents should catch this to handle any StatsPAI-specific failure
    while letting genuinely unexpected errors (e.g. ``AttributeError``
    from misuse) propagate.

    Parameters
    ----------
    message : str
        Human-readable error message.
    recovery_hint : str, optional
        One-sentence, actionable suggestion for how to recover.
    diagnostics : dict, optional
        Machine-readable details (test statistic, threshold, variable
        names, etc.) so an agent can branch on specifics.
    alternative_functions : list of str, optional
        Ranked list of ``sp.xxx`` names the caller may try next.
    """

    #: Short identifier surfaced in ``to_dict()`` and agent summaries.
    code: str = "statspai_error"

    def __init__(
        self,
        message: str,
        *,
        recovery_hint: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
        alternative_functions: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.recovery_hint = recovery_hint
        self.diagnostics: Dict[str, Any] = dict(diagnostics or {})
        self.alternative_functions: List[str] = list(alternative_functions or [])

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.recovery_hint:
            return f"{self.message}\n  ↳ recovery: {self.recovery_hint}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dict for agent consumption."""
        return {
            "kind": self.code,
            "class": type(self).__name__,
            "message": self.message,
            "recovery_hint": self.recovery_hint,
            "diagnostics": dict(self.diagnostics),
            "alternative_functions": list(self.alternative_functions),
        }


# --------------------------------------------------------------------- #
#  Domain-specific errors
# --------------------------------------------------------------------- #


class AssumptionViolation(StatsPAIError, ValueError):
    """An identifying assumption is violated by the data.

    Typical triggers:

    * DID → parallel trends rejected
    * RD → bunching / manipulation at the cutoff
    * IV → weak instrument / exclusion restriction failure
    * Matching / IPW → insufficient overlap / extreme weights

    Subclasses :class:`ValueError` for backwards compatibility with
    code that already catches ``ValueError``.
    """

    code = "assumption_violation"


class IdentificationFailure(AssumptionViolation):
    """The target estimand is not identified under the chosen design.

    Stronger than :class:`AssumptionViolation`: no amount of re-tuning
    within the current method will fix it — the caller must switch
    method or add structure.
    """

    code = "identification_failure"


class DataInsufficient(StatsPAIError, ValueError):
    """Sample or design is too small / sparse for the chosen method.

    Typical triggers:

    * n below the method's minimum (e.g. CS DID with 1 period)
    * too few clusters for cluster-robust inference
    * thin support near the RD cutoff
    * too few treated units for synthetic control
    """

    code = "data_insufficient"


class ConvergenceFailure(StatsPAIError, RuntimeError):
    """An iterative algorithm did not converge.

    Covers optimizer (BFGS / IPOPT), EM, MCMC (rhat / ESS thresholds),
    iterative HDFE demeaning, and cross-fitting loops.
    """

    code = "convergence_failure"


class NumericalInstability(StatsPAIError, RuntimeError):
    """Computation hit a numerical corner case.

    Covers singular / near-singular design matrices, near-zero weight
    denominators, NaNs in sandwich variance, and similar.
    """

    code = "numerical_instability"


class MethodIncompatibility(StatsPAIError, ValueError):
    """The requested method is incompatible with the data / options.

    Typical triggers:

    * panel method called on cross-section
    * staggered estimator called with only one treatment cohort
    * ``robust="hac"`` with no time index
    """

    code = "method_incompatibility"


# --------------------------------------------------------------------- #
#  Warnings
# --------------------------------------------------------------------- #


class StatsPAIWarning(UserWarning):
    """Base warning class for soft (non-raising) StatsPAI diagnostics.

    Instances are not exceptions; they are raised via ``warnings.warn``.
    A warning carries the same ``recovery_hint`` / ``diagnostics`` shape
    as an error so agent code can handle both consistently via
    ``warnings.catch_warnings``.
    """

    #: Short identifier surfaced in agent summaries.
    code: str = "statspai_warning"

    def __init__(
        self,
        message: str = "",
        *,
        recovery_hint: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
        alternative_functions: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.recovery_hint = recovery_hint
        self.diagnostics: Dict[str, Any] = dict(diagnostics or {})
        self.alternative_functions: List[str] = list(alternative_functions or [])

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - mirrors error
        return {
            "kind": self.code,
            "class": type(self).__name__,
            "message": self.message,
            "recovery_hint": self.recovery_hint,
            "diagnostics": dict(self.diagnostics),
            "alternative_functions": list(self.alternative_functions),
        }


class ConvergenceWarning(StatsPAIWarning):
    """Soft convergence warning — fit produced output but diagnostics
    suggest instability (e.g. rhat > 1.01, ESS < 400)."""

    code = "convergence_warning"


class AssumptionWarning(StatsPAIWarning):
    """Soft assumption warning — a test flags concern but the caller
    chose to proceed (e.g. borderline pre-trend p-value)."""

    code = "assumption_warning"


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #


def warn(category: type, message: str, *, stacklevel: int = 2, **kwargs: Any) -> None:
    """Emit a :class:`StatsPAIWarning` with the rich-payload shape.

    Mirrors :func:`warnings.warn` but constructs the warning instance
    so the ``recovery_hint`` / ``diagnostics`` / ``alternative_functions``
    fields survive the round-trip through :mod:`warnings`.

    Parameters
    ----------
    category : type
        A :class:`StatsPAIWarning` subclass.
    message : str
        Human-readable message.
    stacklevel : int, default 2
        Passed through to :func:`warnings.warn`.
    **kwargs
        Forwarded to the warning constructor
        (``recovery_hint`` / ``diagnostics`` / ``alternative_functions``).
    """
    if not (isinstance(category, type) and issubclass(category, StatsPAIWarning)):
        raise TypeError(
            "category must be a StatsPAIWarning subclass; "
            f"got {category!r}"
        )
    warnings.warn(category(message, **kwargs), stacklevel=stacklevel)


__all__ = [
    "StatsPAIError",
    "AssumptionViolation",
    "IdentificationFailure",
    "DataInsufficient",
    "ConvergenceFailure",
    "NumericalInstability",
    "MethodIncompatibility",
    "StatsPAIWarning",
    "ConvergenceWarning",
    "AssumptionWarning",
    "warn",
]
