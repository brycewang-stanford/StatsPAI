"""Analytic tests for the GIS pre-processing helpers.

Every expected value below is known in closed form: geometries are laid out on
a metric grid (EPSG:3395, metres) so lengths, buffer membership and distances
are plain Euclidean quantities.
"""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Point, Polygon  # noqa: E402

from statspai.spatial.utils import (  # noqa: E402
    distance_to_feature,
    line_length_in_polygon,
    share_within_buffer,
)

METRIC = "EPSG:3395"  # World Mercator, axis unit = metre
GEOGRAPHIC = "EPSG:4326"  # WGS84, axis unit = degree


def _square(x0, y0, side):
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


@pytest.fixture
def counties():
    """Two adjacent 1 km / 1.5 km wide cells plus one disjoint cell."""
    polys = [
        Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)]),
        Polygon([(1000, 0), (2500, 0), (2500, 1000), (1000, 1000)]),
        _square(0, 2000, 1000),
    ]
    return gpd.GeoDataFrame(
        {"county_id": ["A", "B", "C"], "geometry": polys}, crs=METRIC
    )


@pytest.fixture
def canal():
    """Straight 2.5 km line at y = 500, crossing counties A and B."""
    return gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 500), (2500, 500)])]}, crs=METRIC
    )


# --------------------------------------------------------------------------
# line_length_in_polygon
# --------------------------------------------------------------------------


def test_line_length_matches_analytic_km(counties, canal):
    # A spans x in [0, 1000] -> 1.0 km; B spans x in [1000, 2500] -> 1.5 km;
    # C is disjoint from the line -> exactly 0.0 (not NaN).
    out = line_length_in_polygon(canal, counties, polygon_id="county_id")
    got = dict(zip(out["county_id"], out["line_length_km"]))
    assert set(out.columns) == {"county_id", "line_length_km"}
    np.testing.assert_allclose(got["A"], 1.0)
    np.testing.assert_allclose(got["B"], 1.5)
    np.testing.assert_allclose(got["C"], 0.0)
    # Total length inside all polygons equals the full 2.5 km line.
    np.testing.assert_allclose(out["line_length_km"].sum(), 2.5)


def test_line_length_metres_unit_and_index_fallback(counties, canal):
    out = line_length_in_polygon(canal, counties, unit="m")
    np.testing.assert_allclose(sorted(out["line_length_m"]), [0.0, 1000.0, 1500.0])
    assert "_polygon_index" in out.columns


def test_line_length_bad_unit(counties, canal):
    with pytest.raises(ValueError, match="unit must be one of"):
        line_length_in_polygon(canal, counties, unit="miles")


def test_line_length_unknown_id_column(counties, canal):
    with pytest.raises(KeyError, match="polygon_id"):
        line_length_in_polygon(canal, counties, polygon_id="nope")


# --------------------------------------------------------------------------
# share_within_buffer
# --------------------------------------------------------------------------


@pytest.fixture
def axis_canal():
    return gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (10000, 0)])]}, crs=METRIC
    )


@pytest.fixture
def towns():
    """4 towns: 2 within 1 km of the x-axis line, 2 outside."""
    pts = [
        Point(1000, 500),  # 0.5 km  -> inside
        Point(2000, 999),  # 0.999 km -> inside
        Point(3000, 1001),  # 1.001 km -> outside
        Point(4000, 5000),  # 5 km    -> outside
    ]
    return gpd.GeoDataFrame(
        {"county_id": ["A", "A", "B", "B"], "geometry": pts}, crs=METRIC
    )


def test_share_within_buffer_overall(axis_canal, towns):
    # 2 of 4 points lie within 1 km of the line.
    assert share_within_buffer(towns, axis_canal, buffer_km=1.0) == pytest.approx(0.5)
    # Widen to 2 km: 3 of 4 (the 5 km town is still out).
    assert share_within_buffer(towns, axis_canal, buffer_km=2.0) == pytest.approx(0.75)
    # 6 km captures all four.
    assert share_within_buffer(towns, axis_canal, buffer_km=6.0) == pytest.approx(1.0)


def test_share_within_buffer_grouped(axis_canal, towns):
    out = share_within_buffer(towns, axis_canal, buffer_km=1.0, group_col="county_id")
    got = dict(zip(out["county_id"], out["share"]))
    assert got["A"] == pytest.approx(1.0)  # both A towns within 1 km
    assert got["B"] == pytest.approx(0.0)  # neither B town within 1 km
    assert list(out["n_points"]) == [2, 2]
    assert list(out["n_within"]) == [2, 0]


def test_share_within_buffer_rejects_nonpositive(axis_canal, towns):
    with pytest.raises(ValueError, match="buffer_km must be > 0"):
        share_within_buffer(towns, axis_canal, buffer_km=0.0)


def test_share_within_buffer_unknown_group(axis_canal, towns):
    with pytest.raises(KeyError, match="group_col"):
        share_within_buffer(towns, axis_canal, group_col="nope")


# --------------------------------------------------------------------------
# distance_to_feature
# --------------------------------------------------------------------------


def test_distance_to_line_matches_analytic_km(axis_canal):
    pts = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(5000, 2500),  # 2.5 km above the line
                Point(5000, -3000),  # 3 km below
                Point(5000, 0),  # on the line
                Point(13000, 0),  # 3 km past the (10000, 0) endpoint
            ]
        },
        crs=METRIC,
    )
    out = distance_to_feature(pts, axis_canal)
    assert out.name == "distance_km"
    np.testing.assert_allclose(out.to_numpy(), [2.5, 3.0, 0.0, 3.0])


def test_distance_to_polygon_is_zero_inside(counties):
    pts = gpd.GeoDataFrame({"geometry": [Point(500, 500), Point(0, -2000)]}, crs=METRIC)
    out = distance_to_feature(pts, counties, unit="m")
    # First point is inside county A; second is 2000 m below its southern edge.
    np.testing.assert_allclose(out.to_numpy(), [0.0, 2000.0])
    assert out.name == "distance_m"


# --------------------------------------------------------------------------
# CRS guards -- the whole point of these wrappers
# --------------------------------------------------------------------------


def test_geographic_crs_raises_for_every_helper():
    line = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 0)])]}, crs=GEOGRAPHIC
    )
    poly = gpd.GeoDataFrame(
        {"geometry": [_square(0, -0.5, 1)], "id": [1]}, crs=GEOGRAPHIC
    )
    pts = gpd.GeoDataFrame({"geometry": [Point(0.5, 0.1)]}, crs=GEOGRAPHIC)

    for call in (
        lambda: line_length_in_polygon(line, poly),
        lambda: share_within_buffer(pts, line),
        lambda: distance_to_feature(pts, line),
    ):
        with pytest.raises(ValueError, match="geographic CRS"):
            call()


def test_geographic_crs_message_names_crs_and_suggests_projection():
    line = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 0)])]}, crs=GEOGRAPHIC
    )
    pts = gpd.GeoDataFrame({"geometry": [Point(0.5, 0.1)]}, crs=GEOGRAPHIC)
    with pytest.raises(ValueError) as exc:
        distance_to_feature(pts, line)
    msg = str(exc.value)
    assert "WGS 84" in msg and "4326" in msg  # names the CRS it found
    assert "degrees" in msg
    assert "to_crs" in msg and "EPSG:" in msg  # suggests a fix


def test_allow_geographic_opts_out():
    line = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 0)])]}, crs=GEOGRAPHIC
    )
    pts = gpd.GeoDataFrame({"geometry": [Point(0.0, 2.0)]}, crs=GEOGRAPHIC)
    out = distance_to_feature(pts, line, unit="m", allow_geographic=True)
    # 2 "metres" == 2 degrees: nonsense units, but the opt-in was explicit.
    np.testing.assert_allclose(out.to_numpy(), [2.0])


def test_missing_crs_raises():
    line = gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1000, 0)])]})
    pts = gpd.GeoDataFrame({"geometry": [Point(0, 500)]})
    with pytest.raises(ValueError, match="no CRS set"):
        distance_to_feature(pts, line)


def test_mismatched_crs_raises(axis_canal):
    pts = gpd.GeoDataFrame({"geometry": [Point(0, 500)]}, crs="EPSG:3857")
    with pytest.raises(ValueError, match="CRS mismatch"):
        distance_to_feature(pts, axis_canal)


def test_non_metre_projected_crs_raises(axis_canal):
    # EPSG:2225 = NAD83 / California zone 1, axis unit = US survey foot.
    feet = "EPSG:2225"
    line = axis_canal.set_crs(feet, allow_override=True)
    pts = gpd.GeoDataFrame({"geometry": [Point(0, 500)]}, crs=feet)
    with pytest.raises(ValueError, match="linear unit"):
        distance_to_feature(pts, line)


# --------------------------------------------------------------------------
# Geometry validation
# --------------------------------------------------------------------------


def test_missing_geometry_raises(axis_canal):
    pts = gpd.GeoDataFrame({"geometry": [Point(0, 500), None]}, crs=METRIC)
    with pytest.raises(ValueError, match="missing geometries"):
        distance_to_feature(pts, axis_canal)


def test_invalid_geometry_raises(canal):
    bowtie = Polygon([(0, 0), (1000, 1000), (1000, 0), (0, 1000)])
    polys = gpd.GeoDataFrame({"geometry": [bowtie]}, crs=METRIC)
    with pytest.raises(ValueError, match="invalid geometries"):
        line_length_in_polygon(canal, polys)


def test_empty_frame_raises(axis_canal):
    pts = gpd.GeoDataFrame({"geometry": []}, crs=METRIC)
    with pytest.raises(ValueError, match="is empty"):
        distance_to_feature(pts, axis_canal)


def test_wrong_geometry_type_raises(counties, canal):
    # Polygons passed where lines are expected.
    with pytest.raises(TypeError, match="LineString"):
        line_length_in_polygon(counties, counties)
    # Lines passed where points are expected.
    with pytest.raises(TypeError, match="Point"):
        distance_to_feature(canal, counties)


def test_not_a_geodataframe_raises(axis_canal):
    with pytest.raises(TypeError, match="GeoDataFrame"):
        distance_to_feature(object(), axis_canal)


def test_without_geopandas_raises(monkeypatch):
    import statspai.spatial.utils as mod

    monkeypatch.setattr(mod, "_gpd", None)
    with pytest.raises(ImportError, match="geopandas"):
        mod.distance_to_feature(object(), object())
