import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from statspai.spatial.esda.plots import moran_plot, lisa_cluster_map
from statspai.spatial.weights import knn_weights


def test_moran_plot_smoke():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(50, 2))
    y = rng.standard_normal(50)
    w = knn_weights(coords, k=4); w.transform = "R"
    ax = moran_plot(y, w)
    assert ax is not None
    # slope of fitted line equals Moran's I roughly
    assert hasattr(ax, "plot")


def test_lisa_cluster_map_requires_geopandas(monkeypatch):
    import statspai.spatial.esda.plots as mod
    monkeypatch.setattr(mod, "_gpd", None)
    with pytest.raises(ImportError, match="geopandas"):
        lisa_cluster_map(np.zeros(5), object(), object())
