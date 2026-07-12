"""Analytical parity: panel estimators recover known coefficients.

* ``sp.panel_fgls`` recovers the slope vector under panel heteroskedasticity.
* ``sp.interactive_fe`` (Bai 2009) recovers the structural slope even when the
  regressor is correlated with the interactive factor structure it purges.
* ``sp.panel_logit`` (conditional fixed-effects logit) recovers the slope with
  no incidental-parameters bias.
* ``sp.panel_probit`` (random-effects probit) recovers the structural slope.

Analytical evidence tier (known-truth recovery on deterministic DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# --------------------------------------------------------------------------
# panel FGLS
# --------------------------------------------------------------------------
def test_panel_fgls_recovers_slopes_under_heteroskedasticity():
    rng = np.random.default_rng(0)
    N, T = 40, 10
    rows = []
    for i in range(N):
        for t in range(T):
            x1, x2 = rng.standard_normal(), rng.standard_normal()
            # panel-specific error scale -> heteroskedastic panels
            y = 1.0 + 2.0 * x1 - 1.0 * x2 + rng.standard_normal() * (0.5 + 0.5 * i / N)
            rows.append((i, t, y, x1, x2))
    df = pd.DataFrame(rows, columns=["id", "time", "y", "x1", "x2"])
    res = sp.panel_fgls(df, y="y", x=["x1", "x2"], id="id", time="time",
                        panels="heteroskedastic")
    p = res.params
    assert float(p["x1"]) == pytest.approx(2.0, abs=0.1)
    assert float(p["x2"]) == pytest.approx(-1.0, abs=0.1)


# --------------------------------------------------------------------------
# interactive fixed effects (Bai 2009)
# --------------------------------------------------------------------------
def test_interactive_fe_purges_factor_structure():
    rng = np.random.default_rng(1)
    N, T = 60, 20
    lam = rng.standard_normal(N)
    f = rng.standard_normal(T)
    rows = []
    for i in range(N):
        for t in range(T):
            # regressor deliberately correlated with the factor structure
            x = rng.standard_normal() + 0.5 * lam[i] * f[t]
            y = 2.0 * x + lam[i] * f[t] + rng.standard_normal() * 0.3
            rows.append((i, t, y, x))
    df = pd.DataFrame(rows, columns=["id", "time", "y", "x"])
    res = sp.interactive_fe(df, y="y", x=["x"], id="id", time="time", n_factors=1)
    assert float(res.params["x"]) == pytest.approx(2.0, abs=0.1)


# --------------------------------------------------------------------------
# conditional fixed-effects logit
# --------------------------------------------------------------------------
def test_panel_logit_fe_recovers_slope_without_incidental_bias():
    rng = np.random.default_rng(2)
    N, T, beta = 300, 8, 1.0
    rows = []
    for i in range(N):
        ai = rng.standard_normal()  # unit fixed effect
        for t in range(T):
            x = rng.standard_normal()
            p = 1.0 / (1.0 + np.exp(-(ai + beta * x)))
            y = int(rng.random() < p)
            rows.append((i, t, y, x))
    df = pd.DataFrame(rows, columns=["id", "time", "y", "x"])
    res = sp.panel_logit(df, y="y", x=["x"], id="id", time="time", method="fe")
    assert float(res.params["x"]) == pytest.approx(1.0, abs=0.15)


# --------------------------------------------------------------------------
# random-effects probit
# --------------------------------------------------------------------------
def test_panel_probit_re_recovers_structural_slope():
    rng = np.random.default_rng(7)
    N, T, beta, b0, sigma_u = 500, 6, 0.8, 0.2, 0.7
    rows = []
    for i in range(N):
        ui = rng.standard_normal() * sigma_u  # random intercept
        for t in range(T):
            x = rng.standard_normal()
            p = norm.cdf(b0 + beta * x + ui)
            y = int(rng.random() < p)
            rows.append((i, t, y, x))
    df = pd.DataFrame(rows, columns=["id", "time", "y", "x"])
    res = sp.panel_probit(df, y="y", x=["x"], id="id", time="time", method="re")
    assert float(res.params["x"]) == pytest.approx(0.8, abs=0.1)
