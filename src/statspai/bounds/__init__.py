"""
Bounds and Partial Identification for causal effects.

When point identification is not possible (e.g., due to sample selection,
non-compliance, or missing data), bounds provide informative intervals
for the treatment effect.

Methods
-------
- **Lee Bounds** : Bounds on ATE under sample selection (Lee 2009)
- **Manski Bounds** : Worst-case bounds with minimal assumptions (Manski 1990)
- **Horowitz-Manski Bounds** : Tighter bounds conditioning on covariates (2000)
- **IV Bounds** : Bounds under imperfect instruments (Nevo & Rosen 2012)
- **Oster Delta** : Coefficient stability / identified set (Oster 2019)
- **Selection Bounds** : Lee bounds with covariates (Lee 2009, conditional)
- **Breakdown Frontier** : Assumption robustness frontier (Masten & Poirier 2021)

References
----------
Lee, D. S. (2009).
Training, Wages, and Sample Selection: Estimating Sharp Bounds on
Treatment Effects. RES, 76(3), 1071-1102.

Manski, C. F. (1990).
Nonparametric Bounds on Treatment Effects.
AER P&P, 80(2), 319-323.

Horowitz, J. L. & Manski, C. F. (2000).
Nonparametric Analysis of Randomized Experiments with Missing
Covariate and Outcome Data. JASA, 95(449), 77-84.

Nevo, A. & Rosen, A. M. (2012).
Identification with Imperfect Instruments. RES, 79(3), 1104-1127.

Oster, E. (2019).
Unobservable Selection and Coefficient Stability.
JBES, 37(2), 187-204.

Masten, M. A. & Poirier, A. (2021).
Salvaging Falsified Instrumental Variable Models.
Econometrica, 89(3), 1449-1469.
"""

from .lee_manski import lee_bounds, manski_bounds
from .partial_id import (
    BoundsResult,
    horowitz_manski,
    iv_bounds,
    oster_delta,
    selection_bounds,
    breakdown_frontier,
)
from .balke_pearl import balke_pearl, BalkePearlResult
from .ml_bounds import ml_bounds, MLBoundsResult

__all__ = [
    'lee_bounds',
    'manski_bounds',
    'BoundsResult',
    'horowitz_manski',
    'iv_bounds',
    'oster_delta',
    'selection_bounds',
    'breakdown_frontier',
    'balke_pearl', 'BalkePearlResult',
    'ml_bounds', 'MLBoundsResult',
]
