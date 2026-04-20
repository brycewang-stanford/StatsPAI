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
from .g_computation import g_computation
from .front_door import front_door
from .bootstrap import bootstrap, BootstrapResult
from .twoway_cluster import twoway_cluster
from .conley import conley
from .pate import pate, PATEEstimator
from .jackknife import jackknife_se, cr2_se, wild_cluster_boot
from .wild_subcluster import subcluster_wild_bootstrap, wild_cluster_ci_inv
from .multiway_cluster import (
    multiway_cluster_vcov,
    cluster_robust_se,
    cr3_jackknife_vcov,
)

__all__ = [
    'wild_cluster_bootstrap',
    'aipw',
    'ri_test',
    'fisher_exact',
    'FisherResult',
    'ipw',
    'g_computation',
    'front_door',
    'bootstrap',
    'BootstrapResult',
    'twoway_cluster',
    'conley',
    'pate',
    'PATEEstimator',
    'jackknife_se',
    'cr2_se',
    'wild_cluster_boot',
    'subcluster_wild_bootstrap',
    'wild_cluster_ci_inv',
    'multiway_cluster_vcov',
    'cluster_robust_se',
    'cr3_jackknife_vcov',
]
