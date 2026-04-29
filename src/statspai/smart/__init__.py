"""
Smart Workflow Engine — StatsPAI's unique differentiator.

No other econometrics package offers these:

- recommend()           — DAG + data → estimator selection
- compare_estimators()  — run multiple methods, compare, diagnose
- assumption_audit()    — comprehensive assumption testing by method
- sensitivity_dashboard() — multi-dimensional sensitivity analysis
- pub_ready()           — journal-specific publication readiness checklist
- replicate()           — famous paper replication with built-in data
"""

from .recommend import recommend, RecommendationResult
from .compare import compare_estimators, ComparisonResult
from .assumptions import assumption_audit, AssumptionResult
from .audit import audit
from .brief import brief
from .citations import bib_for, render_citation
from .detect_design import detect_design
from .examples import examples
from .preflight import preflight
from .session import session
from .sensitivity import sensitivity_dashboard, SensitivityDashboard
from .publication import pub_ready, PubReadyResult
from .replicate import replicate, list_replications
from .identification import (
    check_identification,
    IdentificationReport,
    DiagnosticFinding,
    IdentificationError,
)

# verify_recommendation / verify_benchmark are lazily imported via
# __getattr__ below so that the expensive stability-check machinery
# only loads when actually used. This keeps ``recommend(..., verify=False)``
# genuinely zero-overhead at import time, matching the docstring promise.

__all__ = [
    "recommend", "RecommendationResult",
    "compare_estimators", "ComparisonResult",
    "assumption_audit", "AssumptionResult",
    "audit",
    "bib_for",
    "brief",
    "detect_design",
    "examples",
    "preflight",
    "render_citation",
    "session",
    "sensitivity_dashboard", "SensitivityDashboard",
    "pub_ready", "PubReadyResult",
    "replicate", "list_replications",
    "verify_recommendation", "verify_benchmark",
    "check_identification", "IdentificationReport", "DiagnosticFinding",
    "IdentificationError",
]


def __getattr__(name):
    # Cache into globals() after first resolve so subsequent attribute
    # accesses bypass __getattr__ entirely (matches the top-level
    # statspai/__init__.py pattern).
    if name == "verify_recommendation":
        from .verify import verify_recommendation as _vr
        globals()["verify_recommendation"] = _vr
        return _vr
    if name == "verify_benchmark":
        from .benchmark import verify_benchmark as _vb
        globals()["verify_benchmark"] = _vb
        return _vb
    raise AttributeError(f"module 'statspai.smart' has no attribute {name!r}")
