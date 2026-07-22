"""Spatial econometrics — StatsPAI's answer to R's ``spatialreg + spdep + mgwr``
and Python's PySAL.

Three layers, flat-imported for convenience:

Weights
-------
``W``, ``queen_weights``, ``rook_weights``, ``knn_weights``,
``distance_band``, ``kernel_weights``, ``block_weights``.

Exploratory spatial data analysis (ESDA)
---------------------------------------
``moran``, ``moran_local``, ``geary``, ``getis_ord_g``, ``getis_ord_local``,
``join_counts``, ``moran_plot``, ``lisa_cluster_map``.

Regression
----------
- ML estimators: ``sar``, ``sem``, ``sdm``, ``slx``, ``sac``
- Diagnostics: ``lm_tests``, ``moran_residuals``
- Effects: ``impacts`` (LeSage-Pace direct / indirect / total)

All regression estimators accept a :class:`W` object, a ``scipy.sparse``
matrix, or an ``(n, n)`` ndarray for the weights argument.

Examples
--------
>>> import statspai as sp
>>> w = sp.knn_weights(coords, k=6); w.transform = "R"
>>> moran = sp.moran(df["y"], w)
>>> result = sp.sar(w, data=df, formula='y ~ x1 + x2')
>>> eff = sp.impacts(result)
>>> lms = sp.lm_tests("y ~ x1 + x2", df, w)
"""

from .did import SpatialDiDResult, spatial_did
from .esda import geary, getis_ord_g, getis_ord_local, join_counts, moran, moran_local
from .esda.plots import lisa_cluster_map, moran_plot
from .gwr import GWRResult, MGWRResult, gwr, gwr_bandwidth, mgwr
from .iv import SpatialIVResult, spatial_iv
from .models import SpatialModel, sar, sdm, sem
from .models.diagnostics import lm_tests, moran_residuals
from .models.gmm import sar_gmm, sarar_gmm, sem_gmm
from .models.impacts import impacts
from .models.ml import sac, slx
from .panel import SpatialPanelResult, spatial_panel
from .utils import distance_to_feature, line_length_in_polygon, share_within_buffer
from .weights import (
    W,
    block_weights,
    distance_band,
    kernel_weights,
    knn_weights,
    queen_weights,
    rook_weights,
)

__all__ = [
    # weights
    "W",
    "queen_weights",
    "rook_weights",
    "knn_weights",
    "distance_band",
    "kernel_weights",
    "block_weights",
    # esda
    "moran",
    "moran_local",
    "geary",
    "getis_ord_g",
    "getis_ord_local",
    "join_counts",
    "moran_plot",
    "lisa_cluster_map",
    # models
    "sar",
    "sem",
    "sdm",
    "slx",
    "sac",
    "sar_gmm",
    "sem_gmm",
    "sarar_gmm",
    "SpatialModel",
    # diagnostics / effects
    "lm_tests",
    "moran_residuals",
    "impacts",
    # GWR
    "gwr",
    "mgwr",
    "gwr_bandwidth",
    "GWRResult",
    "MGWRResult",
    # panel
    "spatial_panel",
    "SpatialPanelResult",
    # DiD / IV
    "spatial_did",
    "SpatialDiDResult",
    "spatial_iv",
    "SpatialIVResult",
    # GIS pre-processing (optional geopandas extra)
    "line_length_in_polygon",
    "share_within_buffer",
    "distance_to_feature",
]
