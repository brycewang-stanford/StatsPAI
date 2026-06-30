"""
Smart Workflow Engine.

Registered workflow helpers for planning, diagnostics, sensitivity, and
replication support:

- recommend()           — DAG + data → estimator selection
- compare_estimators()  — run multiple methods, compare, diagnose
- assumption_audit()    — comprehensive assumption testing by method
- sensitivity_dashboard() — multi-dimensional sensitivity analysis
- pub_ready()           — journal-specific publication readiness checklist
- replicate()           — famous paper replication with built-in data
"""

from typing import Any

from .assumptions import AssumptionResult, assumption_audit
from .audit import audit
from .brief import brief
from .citations import bib_for, bibtex, render_citation
from .compare import ComparisonResult, compare_estimators
from .detect_design import detect_design
from .examples import examples
from .identification import (
    DiagnosticFinding,
    IdentificationError,
    IdentificationReport,
    check_identification,
)
from .intake import IntakeResult, design_intake
from .methods_appendix import MethodSpec, methods_appendix
from .preflight import preflight
from .publication import PubReadyResult, pub_ready
from .recommend import RecommendationResult, recommend
from .replicate import list_replications, replicate
from .sensitivity import SensitivityDashboard, sensitivity_dashboard
from .session import session

# verify_recommendation / verify_benchmark are lazily imported via
# __getattr__ below so that the expensive stability-check machinery
# only loads when actually used. This keeps ``recommend(..., verify=False)``
# genuinely zero-overhead at import time, matching the docstring promise.

__all__ = [
    "recommend",
    "RecommendationResult",
    "compare_estimators",
    "ComparisonResult",
    "assumption_audit",
    "AssumptionResult",
    "audit",
    "bib_for",
    "bibtex",
    "brief",
    "methods_appendix",
    "MethodSpec",
    "detect_design",
    "examples",
    "IntakeResult",
    "design_intake",
    "preflight",
    "render_citation",
    "session",
    "sensitivity_dashboard",
    "SensitivityDashboard",
    "pub_ready",
    "PubReadyResult",
    "replicate",
    "list_replications",
    "verify_recommendation",
    "verify_benchmark",
    "check_identification",
    "IdentificationReport",
    "DiagnosticFinding",
    "IdentificationError",
]


def __getattr__(name: str) -> Any:
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
