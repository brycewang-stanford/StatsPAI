"""``sp.xcorr`` against R ``stats::ccf`` / ``stats::ar`` and Stata ``xcorr``.

Committed synthetic file ``_fixtures/xcorr.csv``
(``_generate_xcorr_data.py``): 200 observations, ``y`` leads ``x`` by two
periods. References: ``xcorr_R.json`` (R 4.5.2, ``_generate_xcorr_R.R``)
and ``xcorr_Stata.csv`` (Stata 18 MP, ``_generate_xcorr_Stata.do``).

Sign convention, established numerically here: the lag-``h`` value of
``sp.xcorr(x, y)`` is the lag-``h`` value of R ``ccf(x, y)`` and the
lag-``-h`` value of Stata ``xcorr x y``.

Tolerance. ``EXACT``: 1e-10 absolute on correlations (they are bounded by
one, and some are close to zero, so an absolute error is the meaningful
one); 1e-9 relative on AR coefficients and test statistics. Everything is
a closed-form function of sample moments; observed differences are below
1e-12.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.xcorr import xcorr

FIX = Path(__file__).parent / "_fixtures"
ABS = 1e-10
REL = 1e-9
M = 10


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIX / "xcorr.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((FIX / "xcorr_R.json").read_text(encoding="utf-8"))


def err(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.max(np.abs(a - b) / np.abs(b)))


def test_raw_matches_r_ccf(data, ref):
    cc = xcorr("x", "y", data=data, lags=M)
    assert list(cc.table.index) == ref["lags"]
    assert err(cc.table["xcorr"], ref["raw"]) < ABS


def test_stata_xcorr_is_the_mirror_image(data):
    stata = pd.read_csv(FIX / "xcorr_Stata.csv").sort_values("lag")
    ours = xcorr("x", "y", data=data, lags=M).table["xcorr"].to_numpy()
    assert err(ours[::-1], stata["xc"]) < ABS
    # the series is not symmetric, so the two conventions are told apart
    assert err(ours, stata["xc"]) > 0.1
    swapped = xcorr("y", "x", data=data, lags=M).table["xcorr"]
    assert err(swapped, stata["xc"]) < ABS


def test_yule_walker_order_by_aic_matches_r_ar(data, ref):
    r = ref["yw_aic"]
    cc = xcorr("x", "y", data=data, lags=M, prewhiten="ar", ar_method="yw")
    assert cc.ar_orders == {"x": r["order_x"], "y": r["order_y"]}
    assert rel(cc.ar_coefs["x"], r["phi_x"]) < REL
    assert rel(cc.ar_coefs["y"], r["phi_y"]) < REL
    assert err(cc.table["xcorr"], r["ccf"]) < ABS


def test_yule_walker_fixed_orders(data, ref):
    r = ref["yw_fixed"]
    cc = xcorr(
        "x", "y", data=data, lags=M, prewhiten="ar", ar_method="yw", ar_order=(3, 2)
    )
    assert rel(cc.ar_coefs["x"], r["phi_x"]) < REL
    assert rel(cc.ar_coefs["y"], r["phi_y"]) < REL
    assert err(cc.table["xcorr"], r["ccf"]) < ABS


def test_least_squares_fixed_orders_and_haugh(data, ref):
    r = ref["ols_fixed"]
    cc = xcorr("x", "y", data=data, lags=M, prewhiten="ar", ar_order=(3, 2))
    assert cc.n_obs == r["n"]
    assert rel(cc.ar_coefs["x"], r["phi_x"]) < REL
    assert rel(cc.ar_coefs["y"], r["phi_y"]) < REL
    assert err(cc.table["xcorr"], r["ccf"]) < ABS
    assert rel(cc.haugh["statistic"], r["S"]) < REL
    assert rel(cc.haugh["statistic_adj"], r["S_adj"]) < REL
    assert rel(cc.haugh["pvalue"], r["p"]) < REL
    assert cc.haugh["df"] == 2 * M + 1


def test_box_jenkins_filter_of_x(data, ref):
    cc = xcorr("x", "y", data=data, lags=M, prewhiten="x", ar_order=3)
    assert err(cc.table["xcorr"], ref["bj"]["ccf"]) < ABS
    assert cc.haugh == {}


def test_least_squares_order_matches_statsmodels(data):
    """Order choice on the common sample is statsmodels' ``ar_select_order``."""
    sm = pytest.importorskip("statsmodels.tsa.ar_model")
    for ic in ("aic", "bic"):
        cc = xcorr("x", "y", data=data, prewhiten="ar", ic=ic, max_order=12)
        for key, col in (("x", "x"), ("y", "y")):
            sel = sm.ar_select_order(data[col].to_numpy(), 12, ic=ic, trend="c")
            lags = sel.ar_lags or []
            assert cc.ar_orders[key] == len(lags)
