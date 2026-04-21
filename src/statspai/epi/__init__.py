"""
Epidemiology domain primitives (``sp.epi``).

Fills the gap the article calls out — statspai already has the heavy
epidemiological causal machinery (IPW, G-formula, MSM, target trial),
but lacked the entry-level statistical primitives that clinicians,
epidemiologists, and public-health researchers reach for first.

Modelled after R's ``epiR``, ``epitools``, and ``fmsb``.

>>> import statspai as sp
>>> sp.epi.odds_ratio(50, 20, 30, 40)
>>> sp.epi.relative_risk(50, 950, 10, 990)
>>> sp.epi.mantel_haenszel(tables_2x2xK)
>>> sp.epi.direct_standardize(events, pop, standard_weights)
>>> sp.epi.bradford_hill(strength=1.0, temporality=1.0, consistency=0.5, ...)
"""

from .measures import (
    OR2x2Result,
    RR2x2Result,
    RD2x2Result,
    ARResult,
    IRRResult,
    NNTResult,
    odds_ratio,
    relative_risk,
    risk_difference,
    attributable_risk,
    incidence_rate_ratio,
    number_needed_to_treat,
    prevalence_ratio,
)
from .stratified import (
    MantelHaenszelResult,
    mantel_haenszel,
    breslow_day_test,
)
from .standardize import (
    StandardizedRateResult,
    SMRResult,
    direct_standardize,
    indirect_standardize,
)
from .bradford_hill import (
    BradfordHillResult,
    bradford_hill,
    VIEWPOINTS as BRADFORD_HILL_VIEWPOINTS,
)
from .diagnostic import (
    DiagnosticTestResult, ROCResult, KappaResult,
    diagnostic_test, sensitivity_specificity,
    roc_curve, auc, cohen_kappa,
)

__all__ = [
    # Association measures
    "OR2x2Result", "RR2x2Result", "RD2x2Result", "ARResult",
    "IRRResult", "NNTResult",
    "odds_ratio", "relative_risk", "risk_difference",
    "attributable_risk", "incidence_rate_ratio",
    "number_needed_to_treat", "prevalence_ratio",
    # Stratified analysis
    "MantelHaenszelResult", "mantel_haenszel", "breslow_day_test",
    # Standardization
    "StandardizedRateResult", "SMRResult",
    "direct_standardize", "indirect_standardize",
    # Bradford-Hill
    "BradfordHillResult", "bradford_hill", "BRADFORD_HILL_VIEWPOINTS",
    # Clinical diagnostics
    "DiagnosticTestResult", "ROCResult", "KappaResult",
    "diagnostic_test", "sensitivity_specificity",
    "roc_curve", "auc", "cohen_kappa",
]
