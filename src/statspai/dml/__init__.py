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
Treatment and Structural Parameters." *Econometrics Journal*, 21(1),
C1-C68. [@chernozhukov2018double]
"""

from ._diagnostics import DMLDiagnostics, dml_diagnostics
from ._sensitivity import DMLSensitivityResult, dml_sensitivity
from .did import dml_did
from .double_ml import DoubleML, dml
from .dynamic_dml import DynamicDMLResult, dynamic_dml
from .iivm import DoubleMLIIVM
from .irm import DoubleMLIRM
from .model_averaging import (
    DMLAveragingResult,
    dml_model_averaging,
    model_averaging_dml,
)
from .oof import OOFBundle, OOFPredictions
from .panel_dml import DMLPanelResult, dml_panel
from .pliv import DoubleMLPLIV
from .plr import DoubleMLPLR

__all__ = [
    "dml",
    "DoubleML",
    "DoubleMLPLR",
    "DoubleMLIRM",
    "DoubleMLPLIV",
    "DoubleMLIIVM",
    "dml_model_averaging",
    "model_averaging_dml",
    "DMLAveragingResult",
    # v1.7 long-panel DML (Clarke & Polselli 2025)
    "dml_panel",
    "dml_did",
    "dynamic_dml",
    "DynamicDMLResult",
    "DMLPanelResult",
    # v1.13 sensitivity + diagnostics (Chernozhukov-Cinelli-Newey 2022)
    "dml_sensitivity",
    "DMLSensitivityResult",
    "dml_diagnostics",
    "DMLDiagnostics",
    "OOFBundle",
    "OOFPredictions",
]
