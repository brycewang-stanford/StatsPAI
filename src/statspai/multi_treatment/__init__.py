"""
Multi-valued Treatment Effects.

Estimates causal effects when the treatment takes more than two
values (e.g., different drug dosages, multiple policy options).

Uses inverse probability weighting with multinomial propensity scores.

References
----------
Cattaneo, M. D. (2010).
Efficient semiparametric estimation of multi-valued treatment effects.
Journal of Econometrics, 155(2), 138-154.

Lechner, M. (2001).
Identification and estimation of causal effects of multiple treatments
under the conditional independence assumption.
Econometric Evaluation of Labour Market Policies, 43-58.
"""

from .multi_ipw import multi_treatment, MultiTreatment

__all__ = ['multi_treatment', 'MultiTreatment']
