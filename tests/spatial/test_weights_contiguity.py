import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon
from statspai.spatial.weights.contiguity import queen_weights, rook_weights


@pytest.fixture
def three_squares():
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
    ]
    return gpd.GeoDataFrame({"id": [0, 1, 2], "geometry": polys})


def test_queen_vs_rook_edge_vs_corner(three_squares):
    wq = queen_weights(three_squares)
    wr = rook_weights(three_squares)
    assert set(wq.neighbors[1]) == {0, 2}
    assert set(wr.neighbors[1]) == {0, 2}
    assert wq.neighbors[0] == wr.neighbors[0]


def test_contiguity_without_geopandas_raises(monkeypatch):
    import statspai.spatial.weights.contiguity as mod
    monkeypatch.setattr(mod, "_gpd", None)
    with pytest.raises(ImportError, match="geopandas"):
        mod.queen_weights(object())
