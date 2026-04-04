"""
Bounds and Partial Identification for causal effects.

When point identification is not possible (e.g., due to sample selection,
non-compliance, or missing data), bounds provide informative intervals
for the treatment effect.

Methods
-------
- **Lee Bounds** : Bounds on ATE under sample selection (Lee 2009)
- **Manski Bounds** : Worst-case bounds with minimal assumptions (Manski 1990)

References
----------
Lee, D. S. (2009).
Training, Wages, and Sample Selection: Estimating Sharp Bounds on
Treatment Effects. RES, 76(3), 1071-1102.

Manski, C. F. (1990).
Nonparametric Bounds on Treatment Effects.
AER P&P, 80(2), 319-323.
"""

from .lee_manski import lee_bounds, manski_bounds

__all__ = ['lee_bounds', 'manski_bounds']
