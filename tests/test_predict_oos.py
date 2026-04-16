"""Out-of-sample prediction tests for OLS and IV."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def ols_dgp():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.standard_normal(n)
    train = pd.DataFrame({"y": y[:150], "x1": x1[:150], "x2": x2[:150]})
    test = pd.DataFrame({"x1": x1[150:], "x2": x2[150:]})
    return train, test, y[150:]


def test_ols_predict_in_sample_returns_fitted(ols_dgp):
    train, _, _ = ols_dgp
    model = sp.OLSRegression(formula="y ~ x1 + x2", data=train) if hasattr(sp, "OLSRegression") else None
    # Use the public regress entry — it returns EconometricResults; we need the
    # estimator for predict, so go through sp.regression.ols.OLSRegression:
    from statspai.regression.ols import OLSRegression
    m = OLSRegression(formula="y ~ x1 + x2", data=train)
    m.fit()
    yhat = m.predict()
    assert yhat.shape == (150,)
    assert np.isfinite(yhat).all()


def test_ols_predict_out_of_sample(ols_dgp):
    train, test, y_true_oos = ols_dgp
    from statspai.regression.ols import OLSRegression
    m = OLSRegression(formula="y ~ x1 + x2", data=train)
    m.fit()
    yhat = m.predict(test)
    assert yhat.shape == (50,)
    # R² on held-out data should be high given the DGP
    ss_res = float(((y_true_oos - yhat) ** 2).sum())
    ss_tot = float(((y_true_oos - y_true_oos.mean()) ** 2).sum())
    assert 1 - ss_res / ss_tot > 0.8


def test_ols_predict_confidence_interval(ols_dgp):
    train, test, _ = ols_dgp
    from statspai.regression.ols import OLSRegression
    m = OLSRegression(formula="y ~ x1 + x2", data=train)
    m.fit()
    out = m.predict(test, what="confidence", alpha=0.05)
    assert list(out.columns) == ["yhat", "lower", "upper"]
    assert (out["lower"] < out["yhat"]).all()
    assert (out["yhat"] < out["upper"]).all()


def test_ols_prediction_interval_wider_than_confidence(ols_dgp):
    train, test, _ = ols_dgp
    from statspai.regression.ols import OLSRegression
    m = OLSRegression(formula="y ~ x1 + x2", data=train)
    m.fit()
    ci = m.predict(test, what="confidence")
    pi = m.predict(test, what="prediction")
    assert (pi["upper"] - pi["lower"] > ci["upper"] - ci["lower"]).all()


def test_ols_predict_missing_column_raises(ols_dgp):
    train, _, _ = ols_dgp
    from statspai.regression.ols import OLSRegression
    m = OLSRegression(formula="y ~ x1 + x2", data=train)
    m.fit()
    bad = pd.DataFrame({"x1": [0.0]})    # missing x2
    with pytest.raises(Exception):
        m.predict(bad)


# ---------------------------------------------------------------------- IV
@pytest.fixture(scope="module")
def iv_dgp():
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    e = rng.standard_normal(n)
    # Endogenous: x correlates with e via shared shock
    x = 0.8 * z + 0.5 * e + 0.3 * rng.standard_normal(n)
    y = 1.0 + 2.0 * x + e
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    train, test = df.iloc[:300].reset_index(drop=True), df.iloc[300:].reset_index(drop=True)
    return train, test


def test_iv_predict_out_of_sample(iv_dgp):
    train, test = iv_dgp
    from statspai.regression.iv import IVRegression
    m = IVRegression(formula="y ~ (x ~ z)", data=train)
    m.fit()
    yhat = m.predict(test.drop(columns=["y"]))
    assert yhat.shape == (100,)
    # The structural-form predictor should beat simply predicting the mean
    ss_res = float(((test["y"].values - yhat) ** 2).sum())
    ss_null = float(((test["y"].values - train["y"].mean()) ** 2).sum())
    assert ss_res < ss_null
