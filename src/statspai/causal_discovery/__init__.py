"""
Causal Discovery: Learning causal structure from observational data.

Algorithms
----------
- **NOTEARS** : NO TEARS continuous optimisation for DAG learning
  (Zheng et al. 2018). Formulates structure learning as a smooth
  optimisation problem with an acyclicity constraint.

- **PC Algorithm** : Constraint-based causal discovery using conditional
  independence tests (Spirtes, Glymour, Scheines 2000). Learns a CPDAG
  (completed partially directed acyclic graph).

References
----------
Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018).
DAGs with NO TEARS: Continuous Optimization for Structure Learning.
Advances in Neural Information Processing Systems, 31. [@zheng2018dags]

Spirtes, P., Glymour, C., & Scheines, R. (2000).
Causation, Prediction, and Search (2nd ed.). MIT Press. [@spirtes2000causation]
"""

from .notears import notears, NOTEARS
from .pc import pc_algorithm, PCAlgorithm
from .lingam import lingam, LiNGAMResult
from .ges import ges, GESResult
from .fci import fci, FCIResult
from .icp import icp, nonlinear_icp, ICPResult
from .pcmci import pcmci, PCMCIResult, partial_corr_pvalue
from .lpcmci import lpcmci, LPCMCIResult
from .dynotears import dynotears, DYNOTEARSResult

__all__ = [
    'notears',
    'NOTEARS',
    'pc_algorithm',
    'PCAlgorithm',
    'lingam',
    'LiNGAMResult',
    'ges',
    'GESResult',
    'fci',
    'FCIResult',
    'icp',
    'nonlinear_icp',
    'ICPResult',
    'pcmci',
    'PCMCIResult',
    'partial_corr_pvalue',
    'lpcmci',
    'LPCMCIResult',
    'dynotears',
    'DYNOTEARSResult',
]
