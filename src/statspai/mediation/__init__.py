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
*Psychological Methods*, 15(4), 309-334.

Baron, R.M. and Kenny, D.A. (1986).
"The Moderator-Mediator Variable Distinction in Social Psychological Research."
*Journal of Personality and Social Psychology*, 51(6), 1173-1182.
"""

from .mediate import mediate, MediationAnalysis

__all__ = ['mediate', 'MediationAnalysis']
