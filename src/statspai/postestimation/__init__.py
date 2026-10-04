"""
Post-estimation tools for StatsPAI.

Provides:
- margins(): Average Marginal Effects (AME), Marginal Effects at the Mean (MEM)
- marginsplot(): Visualize marginal effects
- test(): Wald / F test for linear restrictions (beta1 = beta2, joint significance)
- lincom(): Linear combinations of coefficients with inference
- nlcom(): Nonlinear combinations by the delta method

Equivalent to Stata's ``margins``, ``test``, ``lincom`` commands.
"""

from .contract import postestimation_contract, postestimation_report
from .hypothesis import lincom, test
from .margins import (
    contrast,
    event_study_table,
    margins,
    margins_at,
    margins_at_plot,
    margins_table,
    marginsplot,
    pwcompare,
)
from .nlcom import nlcom

__all__ = [
    "margins",
    "margins_table",
    "event_study_table",
    "marginsplot",
    "margins_at",
    "margins_at_plot",
    "contrast",
    "pwcompare",
    "test",
    "lincom",
    "nlcom",
    "postestimation_contract",
    "postestimation_report",
]
