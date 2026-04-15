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
from .models import sar, sem, sdm, SpatialModel
from .models.ml import slx, sac
from .models.gmm import sar_gmm, sem_gmm, sarar_gmm
from .models.diagnostics import lm_tests, moran_residuals
from .models.impacts import impacts
from .weights import (
    W, queen_weights, rook_weights, knn_weights,
    distance_band, kernel_weights, block_weights,
)
from .esda import (
    moran, moran_local, geary, getis_ord_g, getis_ord_local, join_counts,
)
from .esda.plots import moran_plot, lisa_cluster_map

__all__ = [
    # weights
    "W",
    "queen_weights", "rook_weights", "knn_weights",
    "distance_band", "kernel_weights", "block_weights",
    # esda
    "moran", "moran_local", "geary",
    "getis_ord_g", "getis_ord_local", "join_counts",
    "moran_plot", "lisa_cluster_map",
    # models
    "sar", "sem", "sdm", "slx", "sac",
    "sar_gmm", "sem_gmm", "sarar_gmm",
    "SpatialModel",
    # diagnostics / effects
    "lm_tests", "moran_residuals", "impacts",
]
