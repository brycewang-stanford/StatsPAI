"""Analytical parity: spatial regression estimators recover known-truth.

* ``sp.slx`` (spatially-lagged X) is OLS on the augmented design ``[1, X, WX]``
  and must reproduce ``numpy.linalg.lstsq`` on that design bit-for-bit.
* ``sp.sac`` / ``sp.sarar_gmm`` recover the spatial-autoregressive coefficient
  ``rho`` of data generated as ``y = (I - rho W)^{-1}(X beta + eps)``.
* ``sp.spatial_did`` recovers a known treatment effect and reports no
  spillover when the DGP has none.
* ``sp.spatial_iv`` recovers a structural coefficient through an excluded
  instrument, and accepts a native StatsPAI ``W`` object (regression guard for
  the ``_coerce_W`` first-row bug).
* ``sp.spatial_panel`` (SAR + fixed effects) recovers ``rho`` and ``beta``.

Analytical evidence tier (known-truth recovery on deterministic DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _knn_W(n, seed, k=6, span=10.0):
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(0, span, n), rng.uniform(0, span, n)])
    return sp.knn_weights(coords, k=k)


def _row_std(Wobj):
    Wd = np.asarray(Wobj.full(), dtype=float)
    return Wd / Wd.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# SLX == OLS on [1, X, WX]
# --------------------------------------------------------------------------
def test_slx_equals_augmented_ols():
    n = 200
    W = _knn_W(n, seed=0)
    Wr = _row_std(W)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    Wx = Wr @ x
    y = 1.0 + 2.0 * x + 1.5 * Wx + rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"y": y, "x": x})
    res = sp.slx(W, df, "y ~ x", row_normalize=True)
    design = np.column_stack([np.ones(n), x, Wx])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    got = res.params
    assert float(got["const"]) == pytest.approx(beta[0], abs=1e-8)
    assert float(got["x"]) == pytest.approx(beta[1], abs=1e-8)
    assert float(got["W_x"]) == pytest.approx(beta[2], abs=1e-8)


# --------------------------------------------------------------------------
# SAC / SARAR recover the spatial-lag rho
# --------------------------------------------------------------------------
def _simulate_sar(n, seed, rho=0.5, beta=2.0, const=1.0, noise=0.5):
    W = _knn_W(n, seed=seed, k=8, span=20.0)
    Wr = _row_std(W)
    rng = np.random.default_rng(seed + 100)
    x = rng.standard_normal(n)
    inv = np.linalg.inv(np.eye(n) - rho * Wr)
    y = inv @ (const + beta * x + rng.standard_normal(n) * noise)
    return W, pd.DataFrame({"y": y, "x": x})


def test_sac_recovers_spatial_rho():
    W, df = _simulate_sar(400, seed=11, rho=0.5)
    res = sp.sac(W, df, "y ~ x", row_normalize=True)
    p = dict(res.params)
    assert float(p["rho"]) == pytest.approx(0.5, abs=0.12)
    assert float(p["x"]) == pytest.approx(2.0, abs=0.15)


def test_sarar_gmm_recovers_spatial_rho():
    W, df = _simulate_sar(400, seed=11, rho=0.5)
    res = sp.sarar_gmm(W, df, "y ~ x", row_normalize=True)
    p = dict(res.params)
    assert float(p["rho"]) == pytest.approx(0.5, abs=0.12)
    assert float(p["x"]) == pytest.approx(2.0, abs=0.15)


# --------------------------------------------------------------------------
# spatial DiD
# --------------------------------------------------------------------------
def _simulate_spatial_did(seed, N=120, T=6, t0=3, tau=2.0, noise=0.4):
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(0, 10, N), rng.uniform(0, 10, N)])
    W = sp.knn_weights(coords, k=5)
    treated = rng.random(N) < 0.5
    ui = rng.standard_normal(N)
    rows = []
    for i in range(N):
        for t in range(T):
            d = 1 if (treated[i] and t >= t0) else 0
            y = ui[i] + 0.3 * t + tau * d + rng.standard_normal() * noise
            rows.append((i, t, y, d))
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "d"])
    return W, df


def test_spatial_did_recovers_direct_effect():
    W, df = _simulate_spatial_did(seed=0, tau=2.0)
    res = sp.spatial_did(df, y="y", treat="d", unit="unit", time="time", W=W)
    assert float(res.direct_effect) == pytest.approx(2.0, abs=0.25)
    assert float(res.pvalue_direct) < 0.01


def test_spatial_did_no_spurious_spillover():
    W, df = _simulate_spatial_did(seed=0, tau=2.0)
    res = sp.spatial_did(df, y="y", treat="d", unit="unit", time="time", W=W)
    # DGP has no spatial spillover of treatment.
    assert float(res.pvalue_spillover) > 0.05


# --------------------------------------------------------------------------
# spatial IV (+ native-W regression guard)
# --------------------------------------------------------------------------
def _simulate_spatial_iv(seed=41, n=600):
    W = _knn_W(n, seed=seed, k=8, span=20.0)
    rng = np.random.default_rng(seed + 1)
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    xend = 0.8 * z + 0.6 * u + rng.standard_normal(n) * 0.3
    exg = rng.standard_normal(n)
    y = 1.0 + 1.5 * xend + 0.5 * exg + u  # endogeneity through u
    df = pd.DataFrame({"y": y, "xend": xend, "exg": exg, "z": z})
    return W, df


def _coef(res, name):
    tab = res.coefficients
    return float(tab.loc[tab["variable"] == name, "coef"].iloc[0])


def test_spatial_iv_recovers_structural_coefficient():
    W, df = _simulate_spatial_iv()
    res = sp.spatial_iv(
        df, y="y", endog=["xend"], exog=["exg"], W=W,
        instruments=["z"], include_WY=False,
    )
    assert _coef(res, "xend") == pytest.approx(1.5, abs=0.15)
    assert _coef(res, "exg") == pytest.approx(0.5, abs=0.15)


def test_spatial_iv_accepts_native_W_object():
    """Regression guard: a native StatsPAI ``W`` must give the same answer as
    its dense array (``_coerce_W`` previously grabbed only the first row)."""
    W, df = _simulate_spatial_iv()
    kw = dict(y="y", endog=["xend"], exog=["exg"], instruments=["z"], include_WY=False)
    r_native = sp.spatial_iv(df, W=W, **kw)
    r_array = sp.spatial_iv(df, W=np.asarray(W.full()), **kw)
    np.testing.assert_allclose(
        r_native.coefficients["coef"].to_numpy(),
        r_array.coefficients["coef"].to_numpy(),
        rtol=1e-10,
    )


# --------------------------------------------------------------------------
# spatial panel (SAR + FE)
# --------------------------------------------------------------------------
def test_spatial_panel_sar_fe_recovers_rho_and_beta():
    N, T, rho = 50, 8, 0.4
    W = _knn_W(N, seed=31, k=5)
    Wr = _row_std(W)
    inv = np.linalg.inv(np.eye(N) - rho * Wr)
    rng = np.random.default_rng(31)
    alpha = rng.standard_normal(N)
    rows = []
    for t in range(T):
        x = rng.standard_normal(N)
        y = inv @ (alpha + 2.0 * x + rng.standard_normal(N) * 0.4)
        for i in range(N):
            rows.append((i, t, y[i], x[i]))
    df = pd.DataFrame(rows, columns=["entity", "time", "y", "x"])
    res = sp.spatial_panel(
        df, "y ~ x", entity="entity", time="time", W=W, model="sar", effects="fe"
    )
    p = dict(res.params)
    assert float(p["rho"]) == pytest.approx(0.4, abs=0.1)
    assert float(p["x"]) == pytest.approx(2.0, abs=0.1)
