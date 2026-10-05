"""Multivariate descriptive methods: principal components and factor analysis."""

from .factor import FactorResult, factor
from .pca import PCAResult, pca

__all__ = ["pca", "PCAResult", "factor", "FactorResult"]
