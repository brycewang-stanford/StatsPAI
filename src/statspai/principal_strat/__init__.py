"""
Principal Stratification (Frangakis & Rubin 2002).

Principal strata classify units by the joint potential values of a
post-treatment variable (e.g. compliance type or survival status).
Methods provided:

* ``principal_strat(..., method='monotonicity')`` — sharp bounds on
  complier/always-taker/never-taker ATEs under monotonicity (Angrist,
  Imbens & Rubin 1996; Abadie 2002) + point-identified LATE.
* ``principal_strat(..., method='principal_score')`` — covariate-
  based weighting estimator (Jo & Stuart 2009; Ding & Lu 2017) that
  point-identifies stratum-specific effects when principal ignorability
  holds.

Also ships :func:`survivor_average_causal_effect` (SACE bounds —
Zhang & Rubin 2003) as a specialized entry point for the classical
truncation-by-death problem.
"""

from .principal_strat import (
    principal_strat,
    PrincipalStratResult,
    survivor_average_causal_effect,
)

__all__ = [
    'principal_strat',
    'PrincipalStratResult',
    'survivor_average_causal_effect',
]
