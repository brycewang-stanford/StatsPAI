"""
Targeted Maximum Likelihood Estimation (TMLE) with Super Learner.

TMLE is a doubly robust, semiparametrically efficient estimator for
causal effects that combines initial outcome regression with a targeted
bias-correction step using the propensity score.

Components
----------
- **TMLE** : Full TMLE estimator for ATE/ATT with targeting step
- **SuperLearner** : Ensemble learner for nuisance parameter estimation

References
----------
van der Laan, M. J. & Rose, S. (2011).
Targeted Learning: Causal Inference for Observational and Experimental Data.
Springer Series in Statistics.

van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007).
Super Learner. Statistical Applications in Genetics and Molecular Biology, 6(1).
"""

from .tmle import tmle, TMLE
from .super_learner import super_learner, SuperLearner
from .ltmle import ltmle, LTMLEResult
from .ltmle_survival import ltmle_survival, LTMLESurvivalResult
from .hal_tmle import hal_tmle, HALRegressor, HALClassifier

__all__ = [
    'tmle',
    'TMLE',
    'super_learner',
    'SuperLearner',
    'ltmle',
    'LTMLEResult',
    'ltmle_survival',
    'LTMLESurvivalResult',
    'hal_tmle',
    'HALRegressor',
    'HALClassifier',
]
