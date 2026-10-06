"""``sp.lrvar`` against R ``sandwich``.

The long-run variance of a series is ``T`` times the HAC variance of the
coefficient of an intercept-only regression, so every estimate is compared
with ``T * vcov`` from ``sandwich`` 3.1-1 (R 4.5.2) on the committed
synthetic file ``_fixtures/lrvar.csv`` (``_generate_lrvar_data.py``;
reference ``lrvar_R.json`` from ``_generate_lrvar_R.R``): an ARMA(1,1)
series ``x`` and a bivariate VAR(1) ``(y1, y2)``.

Covered: ``kernHAC`` for the five kernels x {fixed bandwidth, ``bwAndrews``,
``bwNeweyWest`` (Bartlett, Parzen, QS)} x prewhitening {0, 1} x ``adjust``;
``NeweyWest`` with a fixed and with the automatic lag; ``sandwich::lrvar``.

Tolerance. ``EXACT`` (1e-9 relative): the estimator is a finite sum and
both sides evaluate the same formula, so only rounding separates them
(observed: below 1e-14).

One documented difference: with ``K`` series ``sandwich`` scales
``adjust=TRUE`` by ``T / (T - K)`` (the regression has ``K`` coefficients),
``sp.lrvar`` by ``T / (T - 1)`` (one mean per series). The test converts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.lrvar import lrvar

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9
KERNEL = {
    "Bartlett": "bartlett",
    "Parzen": "parzen",
    "Quadratic Spectral": "qs",
    "Tukey-Hanning": "tukey-hanning",
    "Truncated": "truncated",
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIX / "lrvar.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((FIX / "lrvar_R.json").read_text(encoding="utf-8"))


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.max(np.abs(a - b) / np.abs(b)))


def _series(data: pd.DataFrame, tag: str):
    return data["x"] if tag == "x" else data[["y1", "y2"]]


def _to_sandwich(value, tag: str, adjust: bool, n: int):
    """Our T/(T-1) adjustment expressed in sandwich's T/(T-K)."""
    if tag == "Y" and adjust:
        return np.asarray(value) * (n - 1) / (n - 2)
    return value


def test_kernhac_every_kernel_bandwidth_prewhitening(data, ref):
    n = ref["n"]
    assert len(ref["kernhac"]) == 104
    for case in ref["kernhac"]:
        rule = case["rule"]
        bandwidth = rule if rule in ("andrews", "newey-west") else float(rule)
        fit = lrvar(
            _series(data, case["series"]),
            kernel=KERNEL[case["kernel"]],
            bandwidth=bandwidth,
            prewhite=case["prewhite"],
            adjust=case["adjust"],
        )
        label = (case["series"], case["kernel"], rule, case["prewhite"])
        # same closed-form bandwidth on both sides
        assert rel(fit.bandwidth, case["bw"]) < EXACT, label
        ours = _to_sandwich(fit.lrvar, case["series"], case["adjust"], n)
        # same finite weighted sum of autocovariances
        assert rel(ours, case["lrvar"]) < EXACT, label


def test_neweywest_fixed_and_automatic_lag(data, ref):
    n = ref["n"]
    for case in ref["neweywest"]:
        kwargs = dict(prewhite=case["prewhite"], adjust=case["adjust"])
        if case["lag"] == "auto":
            fit = lrvar(
                _series(data, case["series"]),
                bandwidth="newey-west",
                integer_lag=True,
                **kwargs,
            )
            # NeweyWest() truncates its bandwidth to a lag L, weights
            # 1 - j/(L+1)
            assert fit.bandwidth == np.floor(case["bw"]) + 1
        else:
            fit = lrvar(
                _series(data, case["series"]), bandwidth=case["lag"] + 1, **kwargs
            )
        ours = _to_sandwich(fit.lrvar, case["series"], case["adjust"], n)
        assert rel(ours, case["lrvar"]) < EXACT, case


def test_sandwich_lrvar_is_var_mean(data, ref):
    r = ref["lrvar"]
    n = ref["n"]
    andrews = dict(kernel="qs", bandwidth="andrews")
    nw = dict(kernel="bartlett", bandwidth="newey-west", integer_lag=True)
    pairs = [
        (lrvar(data["x"], prewhite=1, adjust=True, **andrews), r["andrews"]),
        (lrvar(data["x"], **andrews), r["andrews_raw"]),
        (lrvar(data["x"], prewhite=1, adjust=True, **nw), r["neweywest"]),
        (lrvar(data["x"], **nw), r["neweywest_raw"]),
    ]
    for fit, target in pairs:
        # sandwich::lrvar returns the variance of the mean, J / T
        assert rel(fit.var_mean, target) < EXACT
    both = data[["y1", "y2"]]
    for kwargs, key in ((andrews, "andrews_Y"), (nw, "neweywest_Y")):
        fit = lrvar(both, prewhite=1, adjust=True, **kwargs)
        assert rel(fit.var_mean * (n - 1) / (n - 2), r[key]) < EXACT
