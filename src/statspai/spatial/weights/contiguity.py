"""Contiguity weights (queen / rook). Requires geopandas + shapely."""
from __future__ import annotations

from .core import W

try:
    import geopandas as _gpd
except ImportError:
    _gpd = None


def _require_gpd():
    if _gpd is None:
        raise ImportError(
            "geopandas is required for contiguity weights. "
            "Install with `pip install geopandas shapely`."
        )


def _contiguity(gdf, criterion: str) -> W:
    _require_gpd()
    if criterion not in {"queen", "rook"}:
        raise ValueError("criterion must be 'queen' or 'rook'")
    geoms = list(gdf.geometry.values)
    n = len(geoms)
    sindex = gdf.sindex
    neighbors = {i: [] for i in range(n)}
    for i in range(n):
        candidates = list(sindex.intersection(geoms[i].bounds))
        for j in candidates:
            if j == i:
                continue
            inter = geoms[i].intersection(geoms[j])
            if inter.is_empty:
                continue
            if criterion == "queen":
                neighbors[i].append(int(j))
            else:
                if inter.geom_type in {"LineString", "MultiLineString"} or (
                    hasattr(inter, "length") and inter.length > 0
                ):
                    neighbors[i].append(int(j))
    return W(neighbors)


def queen_weights(gdf) -> W:
    return _contiguity(gdf, "queen")


def rook_weights(gdf) -> W:
    return _contiguity(gdf, "rook")
