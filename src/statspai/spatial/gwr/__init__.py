"""Geographically Weighted Regression (GWR) and Multiscale GWR (MGWR).

Provides PySAL mgwr-equivalent local regression, with AICc / CV bandwidth
selection via golden-section search. Implementation is pure numpy/scipy —
no compiled extensions — so it works out of the box on any StatsPAI install.
"""
from .gwr import gwr, GWRResult
from .bandwidth import gwr_bandwidth
from .mgwr import mgwr, MGWRResult

__all__ = ["gwr", "GWRResult", "gwr_bandwidth", "mgwr", "MGWRResult"]
