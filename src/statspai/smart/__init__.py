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
from .sensitivity import sensitivity_dashboard, SensitivityDashboard
from .publication import pub_ready, PubReadyResult
from .replicate import replicate, list_replications
from .verify import verify_recommendation

__all__ = [
    "recommend", "RecommendationResult",
    "compare_estimators", "ComparisonResult",
    "assumption_audit", "AssumptionResult",
    "sensitivity_dashboard", "SensitivityDashboard",
    "pub_ready", "PubReadyResult",
    "replicate", "list_replications",
    "verify_recommendation",
]
