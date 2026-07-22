"""Thin GIS pre-processing helpers for building spatial panels.

These wrap ``geopandas`` for the three overlay operations that show up over and
over when turning historical shapefiles (CHGIS, China historical GIS, and
friends) into an analysis-ready panel: length of a linear feature inside each
polygon, share of point features inside a buffer of a linear feature, and
distance from each point to the nearest target feature.

They deliberately do **not** reimplement any GIS primitive. Their value-add is
CRS discipline: computing a length, a buffer, or a distance in *degrees*
(EPSG:4326) instead of a projected CRS silently produces plausible-looking but
meaningless numbers. Every helper here refuses to guess -- it raises with the
CRS it found and a concrete suggestion, and it never reprojects behind your
back.

``geopandas`` is an optional extra: ``pip install statspai[spatial]``.

See ``docs/guides/gis_panel_construction.md`` for the end-to-end workflow.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

try:
    import geopandas as _gpd
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    _gpd = None

__all__ = [
    "line_length_in_polygon",
    "share_within_buffer",
    "distance_to_feature",
]

_METRE_ALIASES = {"metre", "meter", "m", "metres", "meters"}
_UNIT_SCALE = {"km": 1000.0, "m": 1.0}

_CRS_HINT = (
    "Reproject explicitly before calling, e.g. "
    "gdf.to_crs('EPSG:3395') for a global metric CRS, or a national equal-area "
    "/ equidistant projection such as EPSG:4479 (China Geodetic Coordinate "
    "System 2000, metres) or an Albers equal-area definition appropriate to "
    "your study region. Pass allow_geographic=True only if you have already "
    "verified the units are metres."
)


def _require_gpd() -> None:
    if _gpd is None:
        raise ImportError(
            "geopandas is required for statspai.spatial.utils. "
            "Install with `pip install statspai[spatial]` "
            "(or `pip install geopandas shapely`)."
        )


def _check_geometry(gdf: Any, name: str, expected: Optional[set] = None) -> None:
    """Raise on missing, empty, invalid, or wrong-typed geometries."""
    if not hasattr(gdf, "geometry"):
        raise TypeError(f"{name} must be a GeoDataFrame, got {type(gdf).__name__}.")
    geom = gdf.geometry
    if len(geom) == 0:
        raise ValueError(f"{name} is empty; nothing to compute.")
    n_missing = int(geom.isna().sum())
    if n_missing:
        raise ValueError(
            f"{name} has {n_missing} missing geometries. Drop or repair them "
            "before calling (silently skipping rows would corrupt the panel)."
        )
    n_empty = int(geom.is_empty.sum())
    if n_empty:
        raise ValueError(f"{name} has {n_empty} empty geometries; drop or repair them.")
    n_invalid = int((~geom.is_valid).sum())
    if n_invalid:
        raise ValueError(
            f"{name} has {n_invalid} invalid geometries (self-intersections or "
            "unclosed rings). Repair them first, e.g. gdf.geometry = "
            "gdf.geometry.buffer(0)."
        )
    if expected is not None:
        found = set(geom.geom_type.unique())
        if not found <= expected:
            raise TypeError(
                f"{name} must contain {sorted(expected)} geometries, "
                f"found {sorted(found)}."
            )


def _check_crs(gdf: Any, name: str, allow_geographic: bool) -> None:
    """Require a defined, projected, metre-based CRS unless opted out."""
    crs = gdf.crs
    if crs is None:
        raise ValueError(
            f"{name} has no CRS set. A length/buffer/distance computed on an "
            "undefined CRS is meaningless. Set it explicitly, e.g. "
            "gdf.set_crs('EPSG:4326').to_crs('EPSG:3395'). " + _CRS_HINT
        )
    if allow_geographic:
        return
    if crs.is_geographic:
        raise ValueError(
            f"{name} is in a geographic CRS ({crs.name!r}, "
            f"EPSG:{crs.to_epsg()}), whose units are degrees. Lengths, buffers "
            "and distances computed in degrees are not metric and vary with "
            "latitude. " + _CRS_HINT
        )
    unit = (gdf.crs.axis_info[0].unit_name or "").lower()
    if unit not in _METRE_ALIASES:
        raise ValueError(
            f"{name} is projected but its linear unit is {unit!r}, not metres "
            f"(CRS {crs.name!r}). statspai returns metres/kilometres, so "
            "reproject to a metre-based CRS first. " + _CRS_HINT
        )


def _check_same_crs(left: Any, right: Any, left_name: str, right_name: str) -> None:
    if left.crs != right.crs:
        raise ValueError(
            f"CRS mismatch: {left_name} is {left.crs.name!r} and {right_name} "
            f"is {right.crs.name!r}. Reproject one to match the other "
            "explicitly -- statspai will not reproject silently."
        )


def _scale(unit: str) -> float:
    if unit not in _UNIT_SCALE:
        raise ValueError(f"unit must be one of {sorted(_UNIT_SCALE)}, got {unit!r}.")
    return _UNIT_SCALE[unit]


def _dissolve(gdf: Any) -> Any:
    """Single geometry covering the whole layer (geopandas 0.14 / 1.x)."""
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union  # pragma: no cover - geopandas < 1.0


def _prepare(
    left: Any,
    right: Any,
    left_name: str,
    right_name: str,
    allow_geographic: bool,
    left_types: Optional[set] = None,
    right_types: Optional[set] = None,
) -> None:
    _require_gpd()
    _check_geometry(left, left_name, left_types)
    _check_geometry(right, right_name, right_types)
    _check_crs(left, left_name, allow_geographic)
    _check_crs(right, right_name, allow_geographic)
    _check_same_crs(left, right, left_name, right_name)


def line_length_in_polygon(
    lines_gdf: Any,
    polygons_gdf: Any,
    polygon_id: Optional[str] = None,
    unit: str = "km",
    allow_geographic: bool = False,
) -> pd.DataFrame:
    """Length of linear features falling inside each polygon.

    The canonical use is "kilometres of canal (or railway, or road) per
    county", the exposure variable in a historical-GIS panel.

    Parameters
    ----------
    lines_gdf : geopandas.GeoDataFrame
        Linear features. Must be ``LineString`` / ``MultiLineString``.
    polygons_gdf : geopandas.GeoDataFrame
        Areal units. Must be ``Polygon`` / ``MultiPolygon``.
    polygon_id : str, optional
        Column in ``polygons_gdf`` identifying each unit. Defaults to the
        index of ``polygons_gdf``.
    unit : {'km', 'm'}, default 'km'
        Output unit.
    allow_geographic : bool, default False
        Opt out of the projected-CRS requirement. Only set this if you have
        verified the CRS units are metres.

    Returns
    -------
    pandas.DataFrame
        One row per polygon with the id column and ``line_length_<unit>``.
        Polygons with no intersecting line get ``0.0``, not ``NaN``.

    Raises
    ------
    ValueError
        If either frame has no CRS, a geographic CRS (and ``allow_geographic``
        is False), a non-metre projected CRS, mismatched CRS, or missing /
        empty / invalid geometries.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> canal = canal.to_crs("EPSG:4479")  # doctest: +SKIP
    >>> counties = counties.to_crs("EPSG:4479")  # doctest: +SKIP
    >>> sp.line_length_in_polygon(canal, counties, polygon_id="county_id")
    ... # doctest: +SKIP
    """
    _prepare(
        lines_gdf,
        polygons_gdf,
        "lines_gdf",
        "polygons_gdf",
        allow_geographic,
        left_types={"LineString", "MultiLineString"},
        right_types={"Polygon", "MultiPolygon"},
    )
    scale = _scale(unit)
    out_col = f"line_length_{unit}"

    polys = polygons_gdf.copy()
    if polygon_id is None:
        polygon_id = "_polygon_index"
        polys[polygon_id] = polys.index
    elif polygon_id not in polys.columns:
        raise KeyError(f"polygon_id {polygon_id!r} not in polygons_gdf columns.")

    inter = _gpd.overlay(
        lines_gdf[["geometry"]],
        polys[[polygon_id, "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    if len(inter):
        lengths = inter.geometry.length.groupby(inter[polygon_id]).sum() / scale
    else:
        lengths = pd.Series(dtype="float64")
    ids = polys[polygon_id]
    values = ids.map(lengths).fillna(0.0).astype("float64")
    return pd.DataFrame({polygon_id: ids.to_numpy(), out_col: values.to_numpy()})


def share_within_buffer(
    points_gdf: Any,
    lines_gdf: Any,
    buffer_km: float = 10.0,
    group_col: Optional[str] = None,
    allow_geographic: bool = False,
) -> Any:
    """Share of point features lying within a buffer of the linear features.

    The canonical use is "share of market towns within 10 km of the canal",
    either overall or per county.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        Point features (e.g. market towns).
    lines_gdf : geopandas.GeoDataFrame
        Linear features to buffer (e.g. the canal).
    buffer_km : float, default 10.0
        Buffer radius in kilometres. Must be strictly positive.
    group_col : str, optional
        Column in ``points_gdf`` to group by (e.g. ``'county_id'``). If
        omitted the share is computed over all points.
    allow_geographic : bool, default False
        Opt out of the projected-CRS requirement.

    Returns
    -------
    float or pandas.DataFrame
        Without ``group_col``, the overall share in ``[0, 1]``. With it, a
        DataFrame with ``group_col``, ``n_points``, ``n_within``, ``share``.

    Raises
    ------
    ValueError
        On non-positive ``buffer_km``, or any of the CRS / geometry problems
        listed in :func:`line_length_in_polygon`.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> sp.share_within_buffer(towns, canal, buffer_km=10, group_col="county_id")
    ... # doctest: +SKIP
    """
    _prepare(
        points_gdf,
        lines_gdf,
        "points_gdf",
        "lines_gdf",
        allow_geographic,
        left_types={"Point"},
        right_types={"LineString", "MultiLineString"},
    )
    if not buffer_km > 0:
        raise ValueError(f"buffer_km must be > 0, got {buffer_km!r}.")

    buffer = _dissolve(lines_gdf).buffer(buffer_km * 1000.0)
    inside = points_gdf.geometry.intersects(buffer)

    if group_col is None:
        return float(inside.mean())
    if group_col not in points_gdf.columns:
        raise KeyError(f"group_col {group_col!r} not in points_gdf columns.")

    groups = points_gdf[group_col]
    n_points = inside.groupby(groups).size()
    n_within = inside.groupby(groups).sum().astype("int64")
    return pd.DataFrame(
        {
            group_col: n_points.index.to_numpy(),
            "n_points": n_points.to_numpy().astype("int64"),
            "n_within": n_within.to_numpy(),
            "share": (n_within / n_points).to_numpy().astype("float64"),
        }
    )


def distance_to_feature(
    points_gdf: Any,
    target_gdf: Any,
    unit: str = "km",
    allow_geographic: bool = False,
) -> pd.Series:
    """Distance from each point to the nearest target feature.

    The canonical use is "distance from each county seat to the canal", the
    continuous treatment-intensity variable in a historical-GIS design.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        Point features.
    target_gdf : geopandas.GeoDataFrame
        Target layer of any geometry type; the nearest distance to the union
        of its features is returned. Points inside a target polygon get ``0``.
    unit : {'km', 'm'}, default 'km'
        Output unit.
    allow_geographic : bool, default False
        Opt out of the projected-CRS requirement.

    Returns
    -------
    pandas.Series
        Named ``distance_<unit>``, indexed like ``points_gdf``.

    Raises
    ------
    ValueError
        On any of the CRS / geometry problems listed in
        :func:`line_length_in_polygon`.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> sp.distance_to_feature(county_seats, canal, unit="km")  # doctest: +SKIP
    """
    _prepare(
        points_gdf,
        target_gdf,
        "points_gdf",
        "target_gdf",
        allow_geographic,
        left_types={"Point"},
    )
    scale = _scale(unit)
    target = _dissolve(target_gdf)
    dist = points_gdf.geometry.distance(target) / scale
    return pd.Series(
        dist.to_numpy(),
        index=points_gdf.index,
        name=f"distance_{unit}",
        dtype="float64",
    )
