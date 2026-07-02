"""Analytical parity: sp.etregress endogenous-treatment DGP recovery.

Endogenous treatment-effects model (Stata etregress): a binary treatment D with
error correlated to the outcome error biases naive OLS, and the control-function
estimator recovers the structural treatment effect. On a known DGP

    D* = g0 + g1 z + u,  D = 1(D* > 0)
    y  = b0 + b1 x + delta D + eps,   corr(eps, u) = rho > 0

the estimated treatment effect recovers delta, the exogenous slope recovers b1,
and the correction pulls the estimate below the (upward-biased) naive
difference. Analytical evidence tier (known-truth recovery on a deterministic
DGP; the implementation is a two-step control function, so it is not pinned to
an R MLE bit-for-bit).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

DELTA = 2.0
B1 = 0.5
RHO = 0.5


def _simulate(seed, n=4000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    e = rng.normal(0, 1, (n, 2))
    u = e[:, 0]
    eps = RHO * e[:, 0] + np.sqrt(1 - RHO**2) * e[:, 1]
    D = (0.3 + 0.8 * z + u > 0).astype(int)
    y = 1.0 + B1 * x + DELTA * D + eps
    return pd.DataFrame({"y": y, "x": x, "z": z, "D": D})


def test_recovers_treatment_effect_and_slope():
    df = _simulate(0)
    r = sp.etregress(df, y="y", x=["x"], treatment="D", z=["z"], method="mle")
    p = dict(r.params)
    assert float(p["D"]) == pytest.approx(DELTA, abs=0.15)
    assert float(p["x"]) == pytest.approx(B1, abs=0.1)


def test_corrects_upward_endogeneity_bias():
    df = _simulate(0)
    r = sp.etregress(df, y="y", x=["x"], treatment="D", z=["z"], method="mle")
    corrected = float(dict(r.params)["D"])
    # naive difference-in-means is biased UP by the positive error correlation
    naive = df.loc[df.D == 1, "y"].mean() - df.loc[df.D == 0, "y"].mean()
    assert naive > DELTA + 0.1
    assert abs(corrected - DELTA) < abs(naive - DELTA)
