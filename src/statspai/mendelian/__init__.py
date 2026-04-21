"""
Mendelian Randomization methods.

Uses genetic variants as instrumental variables to estimate causal effects
in epidemiological/health economics studies.

Core estimators (``mr``):
  - IVW (inverse-variance weighted)
  - MR-Egger regression
  - Weighted / penalized median

Diagnostics (``diagnostics``):
  - Cochran's Q / Rucker's Q' (heterogeneity)
  - MR-Egger intercept test (directional pleiotropy)
  - Leave-one-out sensitivity
  - Steiger directionality test
  - MR-PRESSO global test + outlier correction
  - Radial MR
"""

from .mr import (
    mendelian_randomization, MRResult,
    mr_egger, mr_ivw, mr_median, mr_plot,
)
from .diagnostics import (
    HeterogeneityResult, PleiotropyResult, LeaveOneOutResult,
    SteigerResult, MRPressoResult, RadialResult,
    mr_heterogeneity, mr_pleiotropy_egger, mr_leave_one_out,
    mr_steiger, mr_presso, mr_radial,
)
from .extras import (
    ModeBasedResult, FStatisticResult,
    mr_mode, mr_f_statistic, mr_funnel_plot, mr_scatter_plot,
)
from .multivariable import (
    mr_multivariable, mr_mediation, mr_bma,
    MVMRResult, MediationMRResult, MRBMAResult,
)
from .dispatcher import mr, available_methods as mr_available_methods

__all__ = [
    # Estimators
    "mendelian_randomization", "MRResult",
    "mr_egger", "mr_ivw", "mr_median", "mr_plot",
    # Diagnostics
    "HeterogeneityResult", "PleiotropyResult", "LeaveOneOutResult",
    "SteigerResult", "MRPressoResult", "RadialResult",
    "mr_heterogeneity", "mr_pleiotropy_egger", "mr_leave_one_out",
    "mr_steiger", "mr_presso", "mr_radial",
    # Extras
    "ModeBasedResult", "FStatisticResult",
    "mr_mode", "mr_f_statistic", "mr_funnel_plot", "mr_scatter_plot",
    # Multivariable / mediation / BMA (Yao et al. 2026 roadmap)
    "mr_multivariable", "mr_mediation", "mr_bma",
    "MVMRResult", "MediationMRResult", "MRBMAResult",
    # v1.5 unified dispatcher
    "mr", "mr_available_methods",
]
