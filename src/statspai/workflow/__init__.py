"""End-to-end causal-inference workflow orchestrator.

``sp.causal(df, y=, treatment=, ...)`` stitches the full analysis
pipeline into one call: diagnose identification -> recommend an
estimator -> fit it -> run the standard robustness suite -> produce
an HTML / Markdown / LaTeX report.

This module materialises the ``agent-native`` workflow as an API while
keeping each stage's statistical assumptions explicit.
"""

from .causal_workflow import (
    causal,
    CausalWorkflow,
)
from .paper import paper, PaperDraft, parse_question

__all__ = ["causal", "CausalWorkflow", "paper", "PaperDraft", "parse_question"]
