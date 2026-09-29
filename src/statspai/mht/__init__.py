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
- ``adjust_pvalues()`` --- convenience dispatcher across all methods.
"""

from .romano_wolf import (
    RomanoWolfResult,
    adjust_pvalues,
    benjamini_hochberg,
    bonferroni,
    holm,
    romano_wolf,
)
from .westfall_young import westfall_young

__all__ = [
    "romano_wolf",
    "RomanoWolfResult",
    "adjust_pvalues",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    "westfall_young",
]
