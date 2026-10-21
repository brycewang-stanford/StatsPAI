"""Spatial maximum-likelihood fits report z statistics, as Stata does.

2026-10-20 probe: ``sp.sdem`` / ``sp.gns`` (and ``sar`` / ``sem`` /
``sdm`` / ``sac``) matched Stata 18 ``spregress, ml`` in estimates and
standard errors but built p-values and intervals from t(n - k). The
numbers below are Stata 18 MP's on the same data (a 40-unit ring,
``spmatrix fromdata W = w1-w40, normalize(row)``; ``spregress y x, ml
errorlag(W) ivarlag(W: x)`` and ``... dvarlag(W) errorlag(W) ivarlag(W:
x)``), read from the printed coefficient table (7 significant digits).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def ring():
    n = 40
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "x": x})
    return W, df


# (term, lower, upper) from Stata's table
STATA = {
    "sdem": {
        "x": (0.3908608, 1.279745),
        "const": (0.9904514, 1.63412),
        "W_x": (-0.2455561, 0.9441259),
        "lambda": (-0.3388963, 0.2790987),
    },
    "gns": {
        "x": (0.2832086, 1.303562),
        "const": (-0.4482438, 2.502106),
        "W_x": (-1.040584, 1.354375),
        "rho": (-0.8921094, 1.329275),
        "lambda": (-1.331514, 0.8426685),
    },
}


@pytest.mark.parametrize("fn", ["sdem", "gns"])
def test_intervals_are_normal_like_stata(ring, fn):
    W, df = ring
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = getattr(sp, fn)(W, df, "y ~ x")
    # 7 significant digits printed; the optimisers agree to ~1e-6. The
    # standard errors come from a central-difference Hessian (step 5e-5)
    # of a log-likelihood of size ~60, whose rounding error is about
    # eps * 60 / (5e-5)^2 ~ 5e-6 relative per entry; for GNS (rho and
    # lambda nearly confounded on a 40-unit ring) that moves the interval
    # ends by up to 6.3e-6 between platforms (const lower end: -0.44824436
    # on macOS, -0.44823810 on Windows / py3.10, Stata -0.4482438).
    tol = 2e-5 if fn == "gns" else 5e-6
    for term, (lo, hi) in STATA[fn].items():
        assert r.conf_int_lower[term] == pytest.approx(lo, abs=tol), term
        assert r.conf_int_upper[term] == pytest.approx(hi, abs=tol), term
    # p = 2 * (1 - Phi(|b / se|))
    z = r.params / r.std_errors
    from scipy import stats

    np.testing.assert_allclose(r.pvalues, 2 * stats.norm.sf(np.abs(z)), rtol=1e-12)


def test_slx_is_least_squares_and_keeps_t(ring):
    W, df = ring
    r = sp.slx(W, df, "y ~ x")
    from scipy import stats

    t = r.params / r.std_errors
    dof = len(df) - len(r.params)
    np.testing.assert_allclose(r.pvalues, 2 * stats.t.sf(np.abs(t), dof), rtol=1e-12)
