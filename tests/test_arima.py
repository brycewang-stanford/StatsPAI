"""ARIMA tests."""
import numpy as np, pytest
from statspai.timeseries.arima import arima


@pytest.fixture(scope="module")
def rw():
    return np.cumsum(np.random.default_rng(0).standard_normal(200))


def test_arima_fit(rw):
    res = arima(rw, order=(1, 1, 0))
    assert res.n == 200
    assert np.isfinite(res.aic)


def test_arima_forecast_shape(rw):
    res = arima(rw, order=(1, 1, 0))
    fc = res.forecast(10)
    assert len(fc) == 10
    assert "forecast" in fc.columns


def test_arima_auto_selects(rw):
    res = arima(rw, auto=True, max_p=3, max_q=2, max_d=2)
    assert res.order is not None
    assert res.aicc < 560  # should beat a bad model


def test_exported():
    import statspai as sp
    assert callable(sp.arima)
