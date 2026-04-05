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
    *Econometrica*, 74(2), 431-497.

Firpo, S. (2007).
    Efficient Semiparametric Estimation of Quantile Treatment Effects.
    *Econometrica*, 75(1), 259-276.
"""

from .qte import qdid, qte, QTEResult

__all__ = ["qdid", "qte", "QTEResult"]
