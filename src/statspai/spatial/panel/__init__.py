"""Spatial panel econometrics (R ``splm`` / ``spreg.ML_LagFE`` etc.)."""
from .estimator import spatial_panel, SpatialPanelResult

__all__ = ["spatial_panel", "SpatialPanelResult"]
