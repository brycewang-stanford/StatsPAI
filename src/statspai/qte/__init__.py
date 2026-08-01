"""
Quantile Treatment Effects (QTE) module for StatsPAI.

Provides estimators for:
- **Quantile DID** (Athey & Imbens 2006) — DID at each quantile
- **QTE via Quantile Regression** (Firpo 2007) — conditional QTE with controls
- **QTE via Distribution** — propensity-score reweighting approach

References
----------
Athey, S. & Imbens, G. W. (2006).
    Identification and Inference in Nonlinear Difference-in-Differences Models.
    *Econometrica*, 74(2), 431-497. [@athey2006identification]

Firpo, S. (2007).
    Efficient Semiparametric Estimation of Quantile Treatment Effects.
    *Econometrica*, 75(1), 259-276. [@firpo2007efficient]
"""

from .beyond_average import BeyondAverageResult, beyond_average_late

# v0.10 distributional / panel QTE frontier
from .dist_iv import DistIVResult, dist_iv, kan_dlate
from .distributional import DTEResult, distributional_te
from .hd_panel import HDPanelQTEResult, qte_hd_panel
from .panel_qtet import panel_qtet
from .qte import QTEResult, qdid, qte

__all__ = [
    "qdid",
    "qte",
    "QTEResult",
    "distributional_te",
    "DTEResult",
    "dist_iv",
    "kan_dlate",
    "DistIVResult",
    "qte_hd_panel",
    "HDPanelQTEResult",
    "beyond_average_late",
    "BeyondAverageResult",
    "panel_qtet",
]
