"""
Inference module for StatsPAI.

Provides robust inference methods for supported estimator results:
- Wild Cluster Bootstrap (Cameron, Gelbach & Miller 2008)
- Randomization Inference (Fisher 1935; Young 2019)
"""

from .aipw import aipw
from .bootstrap import BootstrapResult, bootstrap
from .conley import conley
from .front_door import front_door
from .g_computation import g_computation
from .ipw import ipw
from .iv_wild import iv_wild_bootstrap
from .jackknife import cr2_se, jackknife_se, wild_cluster_boot
from .meta_analysis import MetaAnalysisResult, meta_analysis
from .multiway_cluster import (
    cluster_robust_se,
    cr3_jackknife_vcov,
    multiway_cluster_vcov,
)
from .pate import PATEEstimator, pate
from .randomization import FisherResult, fisher_exact, ri_test
from .twoway_cluster import twoway_cluster
from .wild_bootstrap import wild_cluster_bootstrap
from .wild_subcluster import subcluster_wild_bootstrap, wild_cluster_ci_inv

__all__ = [
    "wild_cluster_bootstrap",
    "aipw",
    "ri_test",
    "fisher_exact",
    "FisherResult",
    "ipw",
    "g_computation",
    "front_door",
    "bootstrap",
    "BootstrapResult",
    "twoway_cluster",
    "conley",
    "pate",
    "PATEEstimator",
    "jackknife_se",
    "cr2_se",
    "wild_cluster_boot",
    "subcluster_wild_bootstrap",
    "wild_cluster_ci_inv",
    "multiway_cluster_vcov",
    "cluster_robust_se",
    "cr3_jackknife_vcov",
    "meta_analysis",
    "MetaAnalysisResult",
]
