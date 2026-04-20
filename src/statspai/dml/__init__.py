"""
Double/Debiased Machine Learning module for StatsPAI.

Implements the Chernozhukov et al. (2018) framework with separate
per-model estimator classes (PLR / IRM / PLIV / IIVM) sharing a common
cross-fitting infrastructure.

Public entry points:

* :func:`dml` — dispatcher, selects the model via ``model=`` string.
* :class:`DoubleML` — legacy façade, delegates to a per-model class.
* :class:`DoubleMLPLR`, :class:`DoubleMLIRM`, :class:`DoubleMLPLIV`,
  :class:`DoubleMLIIVM` — direct per-model entry points.

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., and Robins, J. (2018). "Double/Debiased Machine Learning for
Treatment and Structural Parameters." *Econometrics Journal*, 21(1), C1-C68.
"""

from .double_ml import dml, DoubleML
from .plr import DoubleMLPLR
from .irm import DoubleMLIRM
from .pliv import DoubleMLPLIV
from .iivm import DoubleMLIIVM

__all__ = [
    'dml',
    'DoubleML',
    'DoubleMLPLR',
    'DoubleMLIRM',
    'DoubleMLPLIV',
    'DoubleMLIIVM',
]
