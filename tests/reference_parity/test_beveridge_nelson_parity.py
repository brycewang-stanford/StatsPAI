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
