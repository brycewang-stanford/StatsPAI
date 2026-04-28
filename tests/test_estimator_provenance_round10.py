"""Round-10 provenance: time series + diagnostics (76 → 82/925)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp


def test_arima_provenance():
    from statspai.timeseries.arima import arima
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=200).cumsum())
    r = arima(y, order=(1, 0, 0))
    assert sp.get_provenance(r).function == "sp.timeseries.arima"


def test_garch_provenance():
    from statspai.timeseries.garch import garch
    rng = np.random.default_rng(1)
    r = garch(rng.normal(size=200), p=1, q=1)
    assert sp.get_provenance(r).function == "sp.timeseries.garch"


def test_its_provenance():
    from statspai.timeseries.its import its
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(size=100), "t": range(100)})
    r = its(df, y="y", time="t", intervention=50)
    assert sp.get_provenance(r).function == "sp.timeseries.its"


def test_local_projections_provenance():
    from statspai.timeseries.local_projections import local_projections
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"y": rng.normal(size=200),
                        "shock": rng.normal(size=200)})
    r = local_projections(df, outcome="y", shock="shock", horizons=4)
    assert sp.get_provenance(r).function == "sp.timeseries.local_projections"


def test_mccrary_test_provenance():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=500)})
    r = sp.mccrary_test(df, x="x", c=0)
    assert sp.get_provenance(r).function == "sp.diagnostics.mccrary_test"


def test_rddensity_provenance():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(size=500)})
    r = sp.rddensity(df, x="x", c=0)
    assert sp.get_provenance(r).function == "sp.diagnostics.rddensity"
