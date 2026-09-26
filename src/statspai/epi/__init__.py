"""
Epidemiology domain primitives (``sp.epi``).

Fills the gap the article calls out — statspai already has the heavy
epidemiological causal machinery (IPW, G-formula, MSM, target trial),
but lacked the entry-level statistical primitives that clinicians,
epidemiologists, and public-health researchers reach for first.

Modelled after R's ``epiR``, ``epitools``, and ``fmsb``.

>>> import statspai as sp
>>> round(sp.epi.odds_ratio(50, 20, 30, 40).estimate, 3)
3.333
>>> round(sp.epi.relative_risk(50, 950, 10, 990).estimate, 3)
5.0
>>> tables_2x2xK = [[[10, 20], [5, 25]], [[8, 15], [6, 30]]]
>>> sp.epi.mantel_haenszel(tables_2x2xK).n_strata
2
>>> events, pop = [10, 20, 30], [1000, 2000, 1500]
>>> standard_weights = [0.3, 0.4, 0.3]
>>> round(sp.epi.direct_standardize(events, pop, standard_weights).rate, 4)
0.013
>>> sp.epi.bradford_hill(strength=1.0, temporality=1.0, consistency=0.5).total
2.5
"""

from .bradford_hill import VIEWPOINTS as BRADFORD_HILL_VIEWPOINTS
from .bradford_hill import BradfordHillResult, bradford_hill
from .diagnostic import (
    DiagnosticTestResult,
    KappaResult,
    ROCResult,
    auc,
    cohen_kappa,
    diagnostic_test,
    roc_curve,
    sensitivity_specificity,
)
from .measures import (
    ARResult,
    IRRResult,
    NNTResult,
    OR2x2Result,
    RD2x2Result,
    RR2x2Result,
    attributable_risk,
    incidence_rate_ratio,
    number_needed_to_treat,
    odds_ratio,
    prevalence_ratio,
    relative_risk,
    risk_difference,
)
from .standardize import (
    SMRResult,
    StandardizedRateResult,
    direct_standardize,
    indirect_standardize,
)
from .stratified import MantelHaenszelResult, breslow_day_test, mantel_haenszel

__all__ = [
    # Association measures
    "OR2x2Result",
    "RR2x2Result",
    "RD2x2Result",
    "ARResult",
    "IRRResult",
    "NNTResult",
    "odds_ratio",
    "relative_risk",
    "risk_difference",
    "attributable_risk",
    "incidence_rate_ratio",
    "number_needed_to_treat",
    "prevalence_ratio",
    # Stratified analysis
    "MantelHaenszelResult",
    "mantel_haenszel",
    "breslow_day_test",
    # Standardization
    "StandardizedRateResult",
    "SMRResult",
    "direct_standardize",
    "indirect_standardize",
    # Bradford-Hill
    "BradfordHillResult",
    "bradford_hill",
    "BRADFORD_HILL_VIEWPOINTS",
    # Clinical diagnostics
    "DiagnosticTestResult",
    "ROCResult",
    "KappaResult",
    "diagnostic_test",
    "sensitivity_specificity",
    "roc_curve",
    "auc",
    "cohen_kappa",
]
