"""
Causal Mediation Analysis module for StatsPAI.

Implements modern causal mediation analysis following Imai, Keele, and
Tingley (2010), decomposing total treatment effects into:
- Average Causal Mediation Effect (ACME): indirect effect through mediator
- Average Direct Effect (ADE): direct effect not through mediator

Also supports the classical Baron-Kenny approach for comparison.

References
----------
Imai, K., Keele, L., and Tingley, D. (2010).
"A General Approach to Causal Mediation Analysis."
*Psychological Methods*, 15(4), 309-334. [@imai2010general]

Baron, R.M. and Kenny, D.A. (1986).
"The Moderator-Mediator Variable Distinction in Social Psychological Research."
*Journal of Personality and Social Psychology*, 51(6), 1173-1182. [@baron1986moderator]
"""

from .four_way import FourWayResult, four_way_decomposition
from .iv_mediate import IVMediateResult, iv_mediate
from .mediate import MediationAnalysis, mediate, mediate_interventional
from .paths import mediate_paths
from .sensitivity import MediateSensitivityResult, mediate_sensitivity

__all__ = [
    "mediate",
    "MediationAnalysis",
    "mediate_interventional",
    "mediate_paths",
    "mediate_sensitivity",
    "MediateSensitivityResult",
    "four_way_decomposition",
    "FourWayResult",
    "iv_mediate",
    "IVMediateResult",
]
