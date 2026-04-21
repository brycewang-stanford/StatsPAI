"""
Dynamic Treatment Regimes (DTR).

Estimates optimal sequential treatment rules when treatments are
assigned over multiple time periods.

Methods
-------
- **G-estimation** : Structural nested mean model for optimal DTR
  (Robins 1986, 2004).

References
----------
Robins, J. M. (2004).
Optimal Structural Nested Models for Optimal Sequential Decisions.
In Proceedings of the Second Seattle Symposium in Biostatistics, 189-326.

Murphy, S. A. (2003).
Optimal Dynamic Treatment Regimes.
JRSS-B, 65(2), 331-355.
"""

from .g_estimation import g_estimation, GEstimation
from .q_learning import q_learning, QLearningResult
from .a_learning import a_learning, ALearningResult
from .snmm import snmm, SNMMResult

__all__ = [
    'g_estimation', 'GEstimation',
    'q_learning', 'QLearningResult',
    'a_learning', 'ALearningResult',
    'snmm', 'SNMMResult',
]
