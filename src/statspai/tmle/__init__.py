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
Springer Series in Statistics. [@vanderlaan2011targeted]

van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007).
Super Learner. Statistical Applications in Genetics and Molecular Biology, 6(1). [@vanderlaan2007super]
"""

from .ctmle import ctmle
from .hal_tmle import HALClassifier, HALRegressor, hal_tmle
from .ltmle import LTMLEResult, ltmle
from .ltmle_survival import LTMLESurvivalResult, ltmle_survival
from .super_learner import SuperLearner, super_learner
from .tmle import TMLE, tmle

__all__ = [
    "tmle",
    "TMLE",
    "super_learner",
    "SuperLearner",
    "ltmle",
    "LTMLEResult",
    "ltmle_survival",
    "LTMLESurvivalResult",
    "hal_tmle",
    "ctmle",
    "HALRegressor",
    "HALClassifier",
]
