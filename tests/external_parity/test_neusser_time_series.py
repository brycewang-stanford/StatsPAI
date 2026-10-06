"""Neusser (2016), *Time Series Econometrics*: the examples on the book's data.

Opt-in: the companion data are not redistributed. Convert them with
``python tests/external_parity/neusser_convert.py <folder>`` and set
``STATSPAI_NEUSSER_DIR=<folder>``.

The book prints figures and a few rounded tables, and its data page holds
no code beyond the MATLAB Kalman filter of section 17.4
(``test_neusser_quarterly_gdp.py``). The references here are therefore R
4.5.2 (stats, vars 1.6-1, urca 1.3-4) and Stata 18 run on the same files
on 2026-10-06; each number below was copied from that output, and the
tolerance is the precision at which it was printed unless stated.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_NEUSSER_DIR")
DATA = Path(ROOT) / "_statspai" if ROOT else None
pytestmark = pytest.mark.skipif(
    DATA is None or not (DATA / "bipbeispiel.csv").is_file(),
    reason="set STATSPAI_NEUSSER_DIR to the converted book folder "
    "(tests/external_parity/neusser_convert.py)",
)
NAN = np.nan


def _read(name: str) -> pd.DataFrame:
    assert DATA is not None
    return pd.read_csv(DATA / name)


# ---------------------------------------------------------------- section 5.6
@pytest.fixture(scope="module")
def growth():
    b = _read("bipbeispiel.csv")["bip"].dropna().to_numpy()
    return 100 * (np.log(b[4:]) - np.log(b[:-4]))  # year-on-year, 91 quarters


def test_arma_for_swiss_gdp_growth(growth):
    # R: arima(x, c(1, 0, 3), method = "ML") and c(2, 0, 0)
    fit = sp.arima(growth, order=(1, 0, 3))
    assert fit.log_likelihood == pytest.approx(-107.049619863, abs=1e-5)
    got = fit.params[["ar.L1", "ma.L1", "ma.L2", "ma.L3", "const"]].to_numpy()
    np.testing.assert_allclose(
        got, [0.4837, 0.5794, 0.6182, 0.5166, 1.2696], atol=2e-4
    )
    ar2 = sp.arima(growth, order=(2, 0, 0))
    assert ar2.log_likelihood == pytest.approx(-112.9774277, abs=1e-5)
    # R: predict(f, 9)$pred for the ARMA(1,3)
    path = fit.forecast(9)["forecast"].to_numpy()
    np.testing.assert_allclose(
        path[:4], [-0.4246140, 0.7296399, 1.3833839, 1.3246348], atol=5e-5
    )


def test_information_criteria_never_worse_than_r(growth):
    # R's AIC for ARMA(p, q), p, q = 0..5 (arima, method = "ML"). A lower
    # AIC is a higher likelihood for the same model, so ours may be lower
    # but not higher. ARMA(5,2) is the cell that a start from zero
    # coefficients was added for: 238.26 before, R's 232.06 after.
    r_aic = np.array(
        [
            [361.9401, 287.4160, 260.7859, 233.8592, 228.1336, 227.9454],
            [240.7786, 236.5568, 235.8700, 226.0992, 228.0183, 229.9937],
            [233.9549, 233.6315, 240.3833, 228.0363, 230.0171, 231.9610],
            [233.2266, 235.1092, 237.2962, 229.9241, 231.8439, 234.0796],
            [234.8852, 234.9569, 236.6754, 231.4766, 233.4729, 230.3356],
            [234.2387, 236.4795, 232.0551, 233.4712, 232.0893, 232.2151],
        ]
    )
    for p in range(6):
        for q in range(6):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aic = sp.arima(growth, order=(p, 0, q)).aic
            assert aic <= r_aic[p, q] + 1e-3, (p, q)
    assert sp.arima(growth, order=(5, 0, 2)).aic == pytest.approx(232.0551, abs=1e-3)


# ---------------------------------------------------------------- section 7
def test_zivot_andrews_on_us_gdp():
    # urca: ur.za(y, model = "both", lag = 2)
    res = sp.zivot_andrews(_read("varus_t.csv"), "y", model="both", lags=2, trim=0)
    assert res.statistic == pytest.approx(-3.94955290136, rel=1e-9)
    assert res.break_index == 57
    assert not res.reject["10%"]


# ---------------------------------------------------------------- section 8.4
@pytest.fixture(scope="module")
def smi():
    return _read("smi_ret.csv")


def test_garch_models_for_the_swiss_market_index(smi):
    # Stata: arch r, arch(1) garch(1/2)   (log likelihood -5598.548)
    g21 = sp.garch("r", data=smi, p=2, q=1, vce="opg")
    assert g21.log_likelihood == pytest.approx(-5598.548, abs=2e-3)
    np.testing.assert_allclose(g21.beta, [0.3709151, 0.3744472], atol=5e-4)
    # nested model: arch(1) garch(1), -5599.372
    assert sp.garch("r", data=smi).log_likelihood == pytest.approx(
        -5599.372, abs=2e-3
    )
    assert g21.log_likelihood > -5599.372


def test_ar_garch_with_student_t_errors(smi):
    # Stata: arch r, ar(1) arch(1) garch(1) distribution(t)
    fit = sp.garch("r", data=smi, ar=1, dist="t", vce="opg")
    assert fit.log_likelihood == pytest.approx(-5418.403, abs=2e-3)
    assert fit.nu == pytest.approx(7.286736, abs=2e-3)
    assert fit.std_errors["nu"] == pytest.approx(0.5262404, rel=1e-3)
    got = fit.params[["mu", "ar[1]", "alpha[1]", "beta[1]", "omega"]].to_numpy()
    np.testing.assert_allclose(
        got, [0.081359, 0.0337462, 0.124716, 0.8483254, 0.0388909], atol=5e-5
    )


# ---------------------------------------------------------------- section 11.3
def test_consumer_sentiment_leads_gdp_by_one_quarter():
    lead = _read("leading.csv")
    res = sp.xcorr("gdp", "seco", data=lead, lags=8, prewhiten="ar", ar_order=8)
    outside = res.table.index[res.table["outside"]].tolist()
    # with sp.xcorr(x, y), lag h is corr(x[t + h], y[t]): sentiment one
    # quarter earlier moves with GDP growth
    assert 1 in outside
    # the book's own residual columns are AR(8) residuals of each series
    both = lead.dropna(subset=["res_gdp", "res_seco"])
    book = sp.xcorr("res_gdp", "res_seco", data=both, lags=8)
    table = book.table
    assert table.loc[1, "xcorr"] == pytest.approx(0.323, abs=1e-3)
    assert table["outside"].sum() == 1


# ---------------------------------------------------------------- section 15
def test_advertising_and_sales_recursive_var():
    # Stata: var adver sales, lags(1/2); irf table oirf, stderr
    fit = sp.var(_read("adv_log.csv"), lags=2)
    out = sp.irf(fit, periods=4, ci="asymptotic")
    np.testing.assert_allclose(
        out["irf"]["adver -> sales"],
        [0.055966, 0.056498, 0.020776, -0.006723, -0.021854],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out["se"]["adver -> sales"],
        [0.013027, 0.02095, 0.026859, 0.02976, 0.030669],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out["se"]["sales -> adver"],
        [0.0, 0.025069, 0.025659, 0.027701, 0.029145],
        atol=1e-6,
    )


def test_blanchard_quah_long_run_identification():
    # vars: BQ(VAR(bq, p = 2, type = "const"))$B, which scales the residual
    # covariance by T - Kp - 1 = 219 where ours (and Stata's svar) use T = 224
    fit = sp.var(_read("bq_clean.csv"), lags=2)
    res = sp.svar(fit, long_run=[[NAN, 0], [NAN, NAN]])
    theirs = np.array([[2.78072559698, -2.4528291555], [0.00239333129, 0.2855693125]])
    np.testing.assert_allclose(
        res.impact.to_numpy() * np.sqrt(224 / 219), theirs, rtol=1e-8
    )
    bands = res.irf(40, cumulative=True, ci="bootstrap", reps=100, seed=0)
    last = bands[(bands.shock == "shock2") & (bands.response == "dgdp")].iloc[-1]
    # the restriction holds in every bootstrap sample
    assert abs(last["upper"]) < 0.05 and abs(last["lower"]) < 0.05


# ---------------------------------------------------------------- section 16.5
def test_cointegration_of_consumption_investment_and_output():
    z = _read("coint_t.csv")
    # urca: ca.jo(z, type = "trace", ecdet = "none", K = 2)
    rank = sp.johansen(z, lags=1)
    np.testing.assert_allclose(
        rank.test_stats, [96.473217, 37.597512, 9.009360, 1.190802], atol=1e-6
    )
    assert rank.rank == 2 and rank.n_used == 234
    H = np.array([[1, 0, -1, 0], [0, 1, -1, 0], [0, 0, 0, 1]], dtype=float).T
    # blrtest: consumption and investment enter only relative to output
    ratios = sp.johansen_lrtest(z, rank=2, beta=H)
    assert ratios.statistic == pytest.approx(2.83533241468, rel=1e-9)
    assert ratios.pvalue == pytest.approx(0.2422788, abs=1e-7)
    # bh5lrtest: the consumption-output ratio alone is stationary
    known = sp.johansen_lrtest(z, rank=2, beta_known=[1, 0, -1, 0])
    assert known.statistic == pytest.approx(22.8232696865, rel=1e-9)
    # alrtest: the real interest rate does not adjust
    weak = sp.johansen_lrtest(z, rank=2, loading=np.eye(4)[:, :3])
    assert weak.statistic == pytest.approx(27.2493816976, rel=1e-9)
