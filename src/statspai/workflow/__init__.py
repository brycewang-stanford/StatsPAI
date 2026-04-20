"""End-to-end causal-inference workflow orchestrator.

``sp.causal(df, y=, treatment=, ...)`` stitches the full analysis
pipeline into one call: diagnose identification -> recommend an
estimator -> fit it -> run the standard robustness suite -> produce
a publication-ready HTML / Markdown / LaTeX report.

Unique to StatsPAI — Stata and R both leave it to the user to
remember the sequence of checks.  This module is the ``agent-native``
differentiation materialised as an API.
"""
from .causal_workflow import (
    causal,
    CausalWorkflow,
)

__all__ = ['causal', 'CausalWorkflow']
