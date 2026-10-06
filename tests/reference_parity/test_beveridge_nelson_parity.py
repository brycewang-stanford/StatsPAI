"""``sp.beveridge_nelson`` against its definition, computed in R.

No CRAN package computes the Beveridge-Nelson decomposition of an AR model
(checked 2026-10: none listed), so there is no cross-package parity here.
The reference is the definition evaluated by brute force in another
language: ``_generate_beveridge_nelson_R.R`` fits the AR(p) for the first
difference with ``lm()``, iterates the fitted equation 400 periods ahead
from every date and sets ``trend[t] = forecast(y[t+400]) - 400 * drift``.
``sp.beveridge_nelson`` uses the closed form
``cycle[t] = -e1' F (I - F)^-1 z[t]`` instead. Data: the committed
synthetic ARIMA(2,1,0) with drift in ``_fixtures/beveridge_nelson.csv``.

Tolerance. 1e-9 relative on coefficients, drift, variance and trend (level
near 100-230): the forecasts converge geometrically, so after 400 steps
the truncation error is far below rounding; observed 3e-14. 1e-9 absolute
on the cycle, which is a difference of two levels of order 100.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.beveridge_nelson import beveridge_nelson

FIX = Path(__file__).parent / "_fixtures"
TOL = 1e-9


@pytest.fixture(scope="module")
def y() -> pd.Series:
    return pd.read_csv(FIX / "beveridge_nelson.csv")["y"]


@pytest.fixture(scope="module")
def ref() -> dict:
    text = (FIX / "beveridge_nelson_R.json").read_text(encoding="utf-8")
    return json.loads(text)


@pytest.mark.parametrize("key, order", [("ar1", 1), ("ar2", 2), ("ar4", 4)])
def test_closed_form_equals_long_horizon_forecast(y, ref, key, order):
    r = ref[key]
    bn = beveridge_nelson(y, order=order)
    trend = np.array([np.nan if v is None else v for v in r["trend"]])
    seen = ~np.isnan(trend)
    assert np.array_equal(np.isnan(bn.trend.to_numpy()), ~seen)
    assert seen.sum() == y.size - order
    ours = bn.trend.to_numpy()[seen]
    assert np.max(np.abs(ours / trend[seen] - 1)) < TOL
    cycle = y.to_numpy()[seen] - trend[seen]
    assert np.max(np.abs(bn.cycle.to_numpy()[seen] - cycle)) < TOL
    assert abs(bn.intercept / r["intercept"] - 1) < TOL
    assert np.max(np.abs(bn.ar_coefs / np.atleast_1d(r["phi"]) - 1)) < TOL
    assert abs(bn.drift / r["drift"] - 1) < TOL
    assert abs(bn.sigma2 / r["sigma2"] - 1) < TOL
    assert abs(bn.long_run_multiplier / r["psi1"] - 1) < TOL
    # variance ratio psi(1)^2 / sum psi_j^2, the sum from R's ARMAtoMA
    assert abs(bn.variance_ratio / (r["psi1"] ** 2 / r["gamma0_unit"]) - 1) < TOL


# --- ARMA models --------------------------------------------------------------
#
# Reference: ``beveridge_nelson_arma_R.json`` from
# ``_generate_beveridge_nelson_arma_R.R`` on the committed synthetic
# ARIMA(1,1,1) with drift in ``beveridge_nelson_arma.csv``. R (4.5.2,
# ``stats::arima``, exact ML) fits each ARMA(p, q) to the first difference
# and then evaluates the definition by brute force: at every date it hands
# the differences so far to ``arima(fixed = estimates)`` and sums 1500
# forecasts from ``predict`` (``KalmanForecast``).
#
# Two comparisons, because two things are being checked.
#
# * The decomposition given the model is deterministic. The closed form
#   ``-e1' F (I - F)^-1 E_t(s[t])`` is evaluated at R's estimates and
#   compared with R's long-horizon forecasts: 1e-9 relative on the trend
#   (level near 100-200) and 1e-9 absolute on the cycle. Observed 5e-15
#   and 6e-13.
# * End to end, ``sp.beveridge_nelson(order=(p, q))`` estimates the model
#   itself. Two optimisers on the same exact likelihood: the maximised
#   log-likelihoods agree to 1e-8 (observed 7e-11) and the estimates to
#   1e-4 (observed 4e-6), hence 1e-4 on the cycle (observed 7e-6).
#
# R's ``residuals()`` of an ``arima`` fit are the prediction errors divided
# by the square root of their variance relative to sigma2; ours are the
# prediction errors. The two coincide once the filter has settled and
# differ by that factor at the first date, ``sqrt(gamma(0) / sigma2)``.

ARMA_CASES = ["ma1", "arma11", "arma21", "arma12", "ar2"]


@pytest.fixture(scope="module")
def ya() -> np.ndarray:
    return pd.read_csv(FIX / "beveridge_nelson_arma.csv")["y"].to_numpy()


@pytest.fixture(scope="module")
def ref_arma() -> dict:
    text = (FIX / "beveridge_nelson_arma_R.json").read_text(encoding="utf-8")
    return json.loads(text)


def _r_trend(r: dict) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in r["trend"]])


@pytest.mark.parametrize("key", ARMA_CASES)
def test_arma_closed_form_equals_r_forecasts_at_r_estimates(ya, ref_arma, key):
    from statspai.timeseries.beveridge_nelson import _arma_cycle

    r = ref_arma[key]
    est = {
        "mean": r["mean"],
        "phi": np.array(r["phi"], dtype=float).reshape(-1),
        "theta": np.array(r["theta"], dtype=float).reshape(-1),
        "sigma2": r["sigma2"],
    }
    cycle, innov, psi1, ratio = _arma_cycle(np.diff(ya), est)
    trend = _r_trend(r)
    seen = ~np.isnan(trend)
    assert seen.sum() == ya.size - r["first"]
    ours = (ya[1:] - cycle)[seen[1:]]
    assert np.max(np.abs(ours / trend[seen] - 1)) < TOL
    assert np.max(np.abs(cycle[seen[1:]] - (ya[seen] - trend[seen]))) < TOL
    assert abs(psi1 / r["psi1"] - 1) < TOL
    # variance ratio psi(1)^2 / sum psi_j^2, the sum from R's ARMAtoMA
    assert abs(ratio / (r["psi1"] ** 2 / r["gamma0_unit"]) - 1) < TOL
    resid = np.array(r["residuals"])
    assert np.max(np.abs(innov[100:] - resid[100:])) < TOL
    assert abs(innov[0] / resid[0] - np.sqrt(r["gamma0_unit"])) < TOL


@pytest.mark.parametrize("key", ARMA_CASES)
def test_arma_end_to_end_against_r(ya, ref_arma, key):
    r = ref_arma[key]
    bn = beveridge_nelson(ya, order=(r["p"], r["q"]))
    assert abs(bn.loglik - r["loglik"]) < 1e-8
    loose = 1e-4  # optimiser tolerance on both sides
    assert abs(bn.drift - r["mean"]) < loose
    coefs = np.r_[bn.ar_coefs, bn.ma_coefs]
    theirs = np.r_[
        np.array(r["phi"], float).ravel(), np.array(r["theta"], float).ravel()
    ]
    assert coefs.shape == theirs.shape
    assert np.max(np.abs(coefs - theirs)) < loose
    assert abs(bn.sigma2 / r["sigma2"] - 1) < loose
    assert abs(bn.long_run_multiplier / r["psi1"] - 1) < loose
    trend = _r_trend(r)
    seen = ~np.isnan(trend)
    assert np.max(np.abs(bn.cycle.to_numpy()[seen] - (ya[seen] - trend[seen]))) < loose
