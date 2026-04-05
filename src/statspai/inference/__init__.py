"""
Inference module for StatsPAI.

Provides robust inference methods that work across all estimators:
- Wild Cluster Bootstrap (Cameron, Gelbach & Miller 2008)
- Randomization Inference (Fisher 1935; Young 2019)
"""

from .wild_bootstrap import wild_cluster_bootstrap
from .aipw import aipw
from .randomization import ri_test
from .ipw import ipw
from .bootstrap import bootstrap, BootstrapResult

__all__ = [
    'wild_cluster_bootstrap',
    'aipw',
    'ri_test',
    'ipw',
    'bootstrap',
    'BootstrapResult',
]
