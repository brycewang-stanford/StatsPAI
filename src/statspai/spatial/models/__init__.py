"""Spatial regression models."""

# Re-export legacy ML implementations (pre-refactor) so existing imports
# `from statspai.spatial.models import sar, sem, sdm, SpatialModel` keep
# working while new submodules (_logdet, _base, ...) are added alongside.
from ._legacy import sar, sem, sdm, SpatialModel  # noqa: F401

__all__ = ["sar", "sem", "sdm", "SpatialModel"]
