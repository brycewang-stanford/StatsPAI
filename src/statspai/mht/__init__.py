"""
Multiple Hypothesis Testing (MHT) module for StatsPAI.

Provides corrections for simultaneous inference across many outcomes or
subgroups --- the single most common gap when Python users replicate
empirical economics workflows that rely on Stata's ``rwolf`` or ``wyoung``.

Estimators and utilities:

- **Romano-Wolf stepdown** (Romano & Wolf 2005, 2016) --- bootstrap
  FWER control that exploits dependence across test statistics.
- **Westfall-Young maxT** (Westfall & Young 1993) --- single-step
  resampling-based FWER control.
- **Bonferroni**, **Holm** (1979), **Benjamini-Hochberg** (1995) ---
  classical non-resampling adjustments included for comparison.
- ``adjust_pvalues()`` --- convenience dispatcher across all methods.
"""

from .romano_wolf import (
    romano_wolf,
    RomanoWolfResult,
    adjust_pvalues,
    bonferroni,
    holm,
    benjamini_hochberg,
)

__all__ = [
    "romano_wolf",
    "RomanoWolfResult",
    "adjust_pvalues",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
]
