"""
Dynamic panel GMM estimators for StatsPAI.

Provides:
- Arellano-Bond (1991) first-differenced GMM
- Blundell-Bond (1998) system GMM
- Arellano-Bond test for serial correlation (AR(1)/AR(2))
- Hansen/Sargan test for overidentifying restrictions
"""

from .arellano_bond import xtabond

__all__ = ['xtabond']
