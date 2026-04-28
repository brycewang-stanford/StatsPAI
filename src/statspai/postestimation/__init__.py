"""
Post-estimation tools for StatsPAI.

Provides:
- margins(): Average Marginal Effects (AME), Marginal Effects at the Mean (MEM)
- marginsplot(): Visualize marginal effects
- test(): Wald / F test for linear restrictions (beta1 = beta2, joint significance)
- lincom(): Linear combinations of coefficients with inference

Equivalent to Stata's ``margins``, ``test``, ``lincom`` commands.
"""

from .margins import (
    margins, margins_table, event_study_table, marginsplot, margins_at,
    margins_at_plot, contrast, pwcompare,
)
from .hypothesis import test, lincom

__all__ = [
    'margins', 'margins_table', 'event_study_table',
    'marginsplot', 'margins_at', 'margins_at_plot',
    'contrast', 'pwcompare', 'test', 'lincom',
]
