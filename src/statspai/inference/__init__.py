"""
Inference module for StatsPAI.

Provides robust inference methods that work across all estimators:
- Wild Cluster Bootstrap (Cameron, Gelbach & Miller 2008)
- Randomization Inference (Fisher 1935; Young 2019)
"""

from .wild_bootstrap import wild_cluster_bootstrap
from .aipw import aipw
from .randomization import ri_test, fisher_exact, FisherResult
from .ipw import ipw
from .bootstrap import bootstrap, BootstrapResult
from .twoway_cluster import twoway_cluster
from .conley import conley
from .pate import pate, PATEEstimator
from .jackknife import jackknife_se, cr2_se, wild_cluster_boot

__all__ = [
    'wild_cluster_bootstrap',
    'aipw',
    'ri_test',
    'fisher_exact',
    'FisherResult',
    'ipw',
    'bootstrap',
    'BootstrapResult',
    'twoway_cluster',
    'conley',
    'pate',
    'PATEEstimator',
    'jackknife_se',
    'cr2_se',
    'wild_cluster_boot',
]
