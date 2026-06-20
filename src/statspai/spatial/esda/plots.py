"""Visualisations for ESDA."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..weights.core import W
from .moran import moran, moran_local

try:
    import geopandas as _gpd
except ImportError:
    _gpd = None


def moran_plot(y: Any, w: W, ax: Optional[Any] = None) -> Any:
    """Moran scatter: z vs spatial lag Wz. Slope of OLS line = Moran's I.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.uniform(size=(60, 2))
    >>> w = sp.knn_weights(coords, k=5)
    >>> y = coords[:, 0] + rng.normal(scale=0.3, size=60)
    >>> ax = sp.moran_plot(y, w)
    >>> ax.figure.savefig("moran.png")  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float).ravel()
    z = y - y.mean()
    Wz = w.sparse @ z
    res = moran(y, w, permutations=0)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(z, Wz, s=18, alpha=0.7)
    xs = np.linspace(z.min(), z.max(), 50)
    ax.plot(xs, res.I * xs, "-", color="C3", linewidth=1.5, label=f"I = {res.I:.3f}")
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("z (centered)")
    ax.set_ylabel("Wz (spatial lag)")
    ax.set_title("Moran scatter plot")
    ax.legend()
    return ax


def lisa_cluster_map(
    y: Any,
    w: W,
    gdf: Any,
    ax: Optional[Any] = None,
    p_threshold: float = 0.05,
) -> Any:
    """Classify each observation HH/LL/HL/LH/Not significant and colour on a GDF.

    Requires ``geopandas`` (install with ``pip install geopandas``); the
    ``gdf`` must align row-for-row with ``y`` and ``w``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.uniform(size=(60, 2))
    >>> w = sp.knn_weights(coords, k=5)
    >>> y = coords[:, 0] + rng.normal(scale=0.3, size=60)
    >>> import geopandas as gpd  # doctest: +SKIP
    >>> from shapely.geometry import Point  # doctest: +SKIP
    >>> pts = [Point(xy) for xy in coords]  # doctest: +SKIP
    >>> gdf = gpd.GeoDataFrame(geometry=pts)  # doctest: +SKIP
    >>> ax = sp.lisa_cluster_map(y, w, gdf)  # doctest: +SKIP
    >>> ax.figure.savefig("lisa.png")  # doctest: +SKIP
    """
    if _gpd is None:
        raise ImportError(
            "geopandas is required for lisa_cluster_map. "
            "Install with `pip install geopandas`."
        )
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float).ravel()
    z = y - y.mean()
    Wz = w.sparse @ z
    local = moran_local(y, w, permutations=499, seed=0)
    p = local["p_sim"]
    labels = np.full(len(y), "NS", dtype=object)
    if p is not None:
        hi_z = z > 0
        hi_lag = Wz > 0
        labels[(p <= p_threshold) & hi_z & hi_lag] = "HH"
        labels[(p <= p_threshold) & ~hi_z & ~hi_lag] = "LL"
        labels[(p <= p_threshold) & hi_z & ~hi_lag] = "HL"
        labels[(p <= p_threshold) & ~hi_z & hi_lag] = "LH"
    cmap = {
        "HH": "#d62728",
        "LL": "#1f77b4",
        "HL": "#ff9896",
        "LH": "#9ecae1",
        "NS": "#d9d9d9",
    }
    colours = [cmap[lab] for lab in labels]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    gdf = gdf.copy()
    gdf["_lisa"] = labels
    gdf.plot(color=colours, ax=ax, edgecolor="white", linewidth=0.3)
    ax.set_title("LISA cluster map")
    ax.set_axis_off()
    return ax
