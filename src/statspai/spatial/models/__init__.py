"""Spatial regression models."""

# New sparse-backed implementations (Tasks 12-14). Drop-in replacement for
# the legacy dense estimators in `_legacy`; accept ndarray, scipy.sparse, or
# a `statspai.spatial.weights.W` object. Keep the legacy `SpatialModel`
# facade re-exported for the small number of callers that use it directly.
from .ml import sar, sem, sdm                       # noqa: F401
from ._legacy import SpatialModel                   # noqa: F401

__all__ = ["sar", "sem", "sdm", "SpatialModel"]
