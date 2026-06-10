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


def test_arima_standard_errors(rw):
    res = arima(rw, order=(1, 1, 0))
    # se is exposed, aligned with params, positive and finite
    assert res.se is not None
    assert list(res.se.index) == list(res.params.index)
    assert np.all(np.isfinite(res.se.to_numpy()))
    assert np.all(res.se.to_numpy() > 0)
    # std_errors is an alias for se
    assert res.std_errors.equals(res.se)


def test_arima_conf_int_and_pvalues(rw):
    res = arima(rw, order=(2, 0, 0))
    ci = res.conf_int(alpha=0.05)
    assert list(ci.columns) == ["lower", "upper"]
    assert list(ci.index) == list(res.params.index)
    # params lie inside their own CI; bounds ordered
    assert np.all(ci["lower"].to_numpy() <= res.params.to_numpy())
    assert np.all(res.params.to_numpy() <= ci["upper"].to_numpy())
    assert np.all(ci["lower"].to_numpy() < ci["upper"].to_numpy())
    # pvalues in [0, 1], z = params / se
    pv = res.pvalues
    assert np.all((pv.to_numpy() >= 0) & (pv.to_numpy() <= 1))
    np.testing.assert_allclose(res.tvalues.to_numpy(), (res.params / res.se).to_numpy())


def test_arima_se_matches_statsmodels(rw):
    # the exposed se must equal statsmodels' bse on the underlying fit
    res = arima(rw, order=(1, 1, 1))
    np.testing.assert_allclose(
        res.se.to_numpy(), np.asarray(res._model.bse, dtype=float), rtol=1e-12, atol=0
    )


def test_arima_css_ml_alias_matches_statsmodels_innovations_mle(rw):
    from statsmodels.tsa.arima.model import ARIMA as SMARIMA

    res = arima(rw, order=(2, 0, 0), method="css_ml")
    sm = SMARIMA(
        rw,
        order=(2, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(method="innovations_mle")
    explicit = arima(rw, order=(2, 0, 0), method="innovations_mle")
    np.testing.assert_allclose(res.params.to_numpy(), sm.params, rtol=1e-12, atol=0)
    np.testing.assert_allclose(res.se.to_numpy(), sm.bse, rtol=1e-12, atol=0)
    assert res.log_likelihood == pytest.approx(float(sm.llf))
    np.testing.assert_allclose(
        explicit.params.to_numpy(), res.params.to_numpy(), rtol=0, atol=0
    )


def test_arima_rejects_unknown_method(rw):
    with pytest.raises(ValueError, match="method"):
        arima(rw, order=(1, 0, 0), method="bogus")


def test_exported():
    import statspai as sp

    assert callable(sp.arima)
