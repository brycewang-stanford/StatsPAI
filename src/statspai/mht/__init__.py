"""
Multiple Hypothesis Testing (MHT) module for StatsPAI.

Provides corrections for simultaneous inference across many outcomes or
subgroups --- the single most common gap when Python users replicate
empirical economics workflows that rely on Stata's ``rwolf`` or ``wyoung``.

Estimators and utilities:

- **Romano-Wolf stepdown** (Romano & Wolf 2005, 2016) --- bootstrap
  FWER control that exploits dependence across test statistics.
- **Westfall-Young stepdown maxT** (Westfall & Young 1993) ---
  ``westfall_young()``: FWER control from the design's own
  re-randomizations (within strata, by cluster), as Stata
  ``wyoung, permute()``.
- **Bonferroni**, **Holm** (1979), **Benjamini-Hochberg** (1995) ---
  classical non-resampling adjustments included for comparison.
- ``sharpened_qvalues()`` --- Benjamini-Krieger-Yekutieli (2006)
  two-stage FDR q-values as Anderson's (2008) Stata do-file computes them.
- ``adjust_pvalues()`` --- convenience dispatcher across all methods.
- ``summary_index()`` --- Kling-Liebman-Katz (2007) and Anderson (2008)
  summary indices of several outcomes (Stata ``swindex``).
"""

from .index import SummaryIndexResult, summary_index
from .romano_wolf import (
    RomanoWolfResult,
    adjust_pvalues,
    benjamini_hochberg,
    bonferroni,
    holm,
    romano_wolf,
)
from .sharpened import sharpened_qvalues
from .westfall_young import westfall_young

__all__ = [
    "romano_wolf",
    "RomanoWolfResult",
    "adjust_pvalues",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    "westfall_young",
    "sharpened_qvalues",
    "summary_index",
    "SummaryIndexResult",
]
