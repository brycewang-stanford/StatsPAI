"""
Nonparametric estimation methods.

Provides local polynomial regression (lpoly), kernel density estimation (kdensity),
nonparametric regression (npregress), and conditional inference trees (ctree).
"""

from .conditional_tree import CTreeResult, ctree
from .kdensity import KDensityResult, kdensity
from .lpoly import LPolyResult, lpoly
from .lprobust import (
    LProbustPoint,
    lpbwselect_ce_rot,
    lpbwselect_imse_dpi,
    lpbwselect_imse_rot,
    lpbwselect_mse_dpi,
    lpbwselect_mse_rot,
    lprobust_at_point,
)

__all__ = [
    "ctree",
    "CTreeResult",
    "lpoly",
    "LPolyResult",
    "kdensity",
    "KDensityResult",
    "lprobust_at_point",
    "lpbwselect_mse_dpi",
    "lpbwselect_mse_rot",
    "lpbwselect_imse_dpi",
    "lpbwselect_imse_rot",
    "lpbwselect_ce_rot",
    "LProbustPoint",
]
