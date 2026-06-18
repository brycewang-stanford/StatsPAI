"""Contiguity weights (queen / rook). Requires geopandas + shapely."""

from __future__ import annotations

from typing import Any, Dict, List

from .core import W

try:
    import geopandas as _gpd
except ImportError:
    _gpd = None


def _require_gpd() -> None:
    if _gpd is None:
        raise ImportError(
            "geopandas is required for contiguity weights. "
            "Install with `pip install geopandas shapely`."
        )


def _contiguity(gdf: Any, criterion: str) -> W:
    _require_gpd()
    if criterion not in {"queen", "rook"}:
        raise ValueError("criterion must be 'queen' or 'rook'")
    geoms = list(gdf.geometry.values)
    n = len(geoms)
    sindex = gdf.sindex
    neighbors: Dict[int, List[int]] = {i: [] for i in range(n)}
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


def queen_weights(gdf: Any) -> W:
    """Queen-contiguity spatial weights from polygon geometries.

    Two areal units are queen-contiguous if their geometries share at least
    one boundary point (an edge *or* a single vertex). Requires ``geopandas``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Layer of polygon geometries (accessed via ``gdf.geometry`` and the
        spatial index ``gdf.sindex``).

    Returns
    -------
    W
        Spatial weights object whose neighbour lists encode queen contiguity.

    Raises
    ------
    ImportError
        If ``geopandas`` is not installed.

    Examples
    --------
    >>> import statspai as sp
    >>> import geopandas as gpd  # doctest: +SKIP
    >>> gdf = gpd.read_file("counties.shp")  # doctest: +SKIP
    >>> w = sp.queen_weights(gdf)  # doctest: +SKIP

    See Also
    --------
    rook_weights : Edge-only contiguity (excludes shared vertices).
    """
    return _contiguity(gdf, "queen")


def rook_weights(gdf: Any) -> W:
    """Rook-contiguity spatial weights from polygon geometries.

    Two areal units are rook-contiguous if their geometries share a boundary
    segment of positive length (a shared edge); units touching only at a
    single vertex are *not* neighbours. Requires ``geopandas``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Layer of polygon geometries.

    Returns
    -------
    W
        Spatial weights object whose neighbour lists encode rook contiguity.

    Raises
    ------
    ImportError
        If ``geopandas`` is not installed.

    Examples
    --------
    >>> import statspai as sp
    >>> import geopandas as gpd  # doctest: +SKIP
    >>> w = sp.rook_weights(gpd.read_file("counties.shp"))  # doctest: +SKIP

    See Also
    --------
    queen_weights : Contiguity that also counts shared vertices.
    """
    return _contiguity(gdf, "rook")
