"""Analytical parity: time-series estimators recover known-truth on simulated
processes.

Each case has an unambiguous population target:

* ``sp.garch`` recovers the persistence ``alpha + beta`` of a simulated
  GARCH(1,1) process.
* ``sp.granger_causality`` detects a one-directional lag dependence and rejects
  the reverse.
* ``sp.irf`` (non-orthogonal) equals the closed-form VAR(1) power matrix
  ``A**h``.
* ``sp.johansen`` recovers the cointegration rank of a system with one common
  stochastic trend.
* ``sp.panel_unitroot`` rejects on a stationary panel and fails to reject on a
  random-walk panel.
* ``sp.bvar`` recovers the DGP coefficients under a loose Minnesota prior and
  shrinks to the prior mean (own first lag = 1, cross terms = 0) under a tight
  prior.
* ``sp.its`` recovers a known level shift at the intervention and reports no
  spurious slope change.

Analytical evidence tier (known-truth recovery on deterministic DGPs; no
cross-package target).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


# --------------------------------------------------------------------------
# GARCH(1,1) persistence recovery
# --------------------------------------------------------------------------
def _simulate_garch(seed=0, n=6000, omega=0.1, alpha=0.08, beta=0.9):
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    sig2 = np.zeros(n)
    sig2[0] = omega / (1.0 - alpha - beta)
    for t in range(1, n):
        sig2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sig2[t - 1]
        eps[t] = np.sqrt(sig2[t]) * rng.standard_normal()
    return eps


def test_garch_recovers_persistence():
    eps = _simulate_garch()
    r = sp.garch(eps, p=1, q=1)
    # true persistence = alpha + beta = 0.98
    assert float(r.persistence) == pytest.approx(0.98, abs=0.03)


def test_garch_persistence_below_one_for_stationary_process():
    eps = _simulate_garch()
    r = sp.garch(eps, p=1, q=1)
    assert 0.0 < float(r.persistence) < 1.0


def test_garch_omega_positive():
    eps = _simulate_garch()
    r = sp.garch(eps, p=1, q=1)
    assert float(r.omega) > 0


# --------------------------------------------------------------------------
# Granger causality direction
# --------------------------------------------------------------------------
def _simulate_causal_var(seed=1, n=2000):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
        # y depends on lagged x; x does not depend on lagged y
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + rng.standard_normal()
    return pd.DataFrame({"x": x, "y": y})


def test_granger_detects_true_direction():
    vr = sp.var(_simulate_causal_var(), variables=["x", "y"], lags=1)
    gc = sp.granger_causality(vr, caused="y", causing="x")
    assert bool(gc["reject"]) is True
    assert float(gc["p_value"]) < 0.01


def test_granger_rejects_spurious_reverse_direction():
    vr = sp.var(_simulate_causal_var(), variables=["x", "y"], lags=1)
    gc = sp.granger_causality(vr, caused="x", causing="y")
    assert bool(gc["reject"]) is False
    assert float(gc["p_value"]) > 0.05


# --------------------------------------------------------------------------
# VAR(1) impulse responses == closed-form A**h
# --------------------------------------------------------------------------
def test_irf_matches_closed_form_power_matrix():
    df = _simulate_causal_var(seed=1, n=4000)
    names = ["x", "y"]
    vr = sp.var(df, variables=names, lags=1)
    # Build companion matrix A: A[i, j] = effect of variable j at lag 1
    # on variable i (row = response equation, column = shock variable).
    A = np.zeros((2, 2))
    for i, resp in enumerate(names):
        eq = vr.coefs[resp]
        for j, shock in enumerate(names):
            A[i, j] = float(eq.loc[f"L1.{shock}", "coef"])
    ir = sp.irf(vr, periods=6, orthogonal=False)["irf"]
    Ah = np.eye(2)
    for h in range(6):
        for j, shock in enumerate(names):
            for i, resp in enumerate(names):
                got = float(ir[f"{shock} -> {resp}"][h])
                assert got == pytest.approx(Ah[i, j], abs=1e-9)
        Ah = Ah @ A


# --------------------------------------------------------------------------
# Johansen cointegration rank recovery
# --------------------------------------------------------------------------
def test_johansen_recovers_rank_one():
    rng = np.random.default_rng(2)
    n = 1000
    trend = np.cumsum(rng.standard_normal(n))  # single common stochastic trend
    y1 = trend + rng.standard_normal(n) * 0.5
    y2 = 2.0 * trend + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"y1": y1, "y2": y2})
    jo = sp.johansen(df, variables=["y1", "y2"], lags=1, test="trace")
    assert int(jo.rank) == 1


def test_johansen_full_rank_on_independent_stationary_series():
    rng = np.random.default_rng(5)
    n = 1000
    # Two independent stationary series -> both are already I(0), rank 2.
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    for t in range(1, n):
        y1[t] = 0.3 * y1[t - 1] + rng.standard_normal()
        y2[t] = 0.3 * y2[t - 1] + rng.standard_normal()
    df = pd.DataFrame({"y1": y1, "y2": y2})
    jo = sp.johansen(df, variables=["y1", "y2"], lags=1, test="trace")
    assert int(jo.rank) == 2


# --------------------------------------------------------------------------
# Panel unit-root reject / fail-to-reject
# --------------------------------------------------------------------------
def _panel(seed, rho, N=30, T=50):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        v = 0.0
        for t in range(T):
            v = rho * v + rng.standard_normal()
            rows.append((i, t, v))
    return pd.DataFrame(rows, columns=["id", "time", "val"])


def test_panel_unitroot_rejects_on_stationary_panel():
    r = sp.panel_unitroot(_panel(3, rho=0.3), "val", id="id", time="time", test="ips")
    assert float(r.p_value) < 0.05


def test_panel_unitroot_fails_to_reject_on_random_walk():
    r = sp.panel_unitroot(_panel(4, rho=1.0), "val", id="id", time="time", test="ips")
    assert float(r.p_value) > 0.10


# --------------------------------------------------------------------------
# Bayesian VAR (Minnesota prior)
# --------------------------------------------------------------------------
def _simulate_bvar_data(seed=1, n=800):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.standard_normal()
        y[t] = 0.3 * y[t - 1] + 0.5 * x[t - 1] + rng.standard_normal()
    return pd.DataFrame({"x": x, "y": y})


def test_bvar_loose_prior_recovers_dgp_coefficients():
    df = _simulate_bvar_data()
    # Loose prior -> posterior mean tracks the (near-OLS) DGP coefficients.
    # coef rows = [lag1.x, lag1.y, const]; columns = [eq x, eq y].
    coef = sp.bvar(df, lags=1, lambda1=10.0).coef
    assert float(coef[0, 0]) == pytest.approx(0.6, abs=0.1)  # x on lag1.x
    assert float(coef[0, 1]) == pytest.approx(0.5, abs=0.1)  # y on lag1.x
    assert float(coef[1, 1]) == pytest.approx(0.3, abs=0.1)  # y on lag1.y


def test_bvar_tight_prior_shrinks_to_minnesota_mean():
    df = _simulate_bvar_data()
    # Tight prior -> Minnesota mean: own first lag = 1, cross terms = 0.
    coef = sp.bvar(df, lags=1, lambda1=0.001).coef
    assert float(coef[0, 0]) == pytest.approx(1.0, abs=0.05)  # own lag -> 1
    assert float(coef[1, 1]) == pytest.approx(1.0, abs=0.05)  # own lag -> 1
    assert float(coef[0, 1]) == pytest.approx(0.0, abs=0.05)  # cross -> 0


# --------------------------------------------------------------------------
# interrupted time series
# --------------------------------------------------------------------------
def test_its_recovers_level_shift():
    rng = np.random.default_rng(0)
    T, t0, level, slope = 120, 60, 5.0, 0.05
    t = np.arange(T)
    y = 2.0 + slope * t + level * (t >= t0) + rng.standard_normal(T) * 0.5
    df = pd.DataFrame({"y": y, "t": t})
    res = sp.its(df, y="y", time="t", intervention=t0)
    assert float(res.level_change) == pytest.approx(5.0, abs=0.4)
    assert float(res.pvalue_level) < 0.01


def test_its_no_spurious_slope_change():
    rng = np.random.default_rng(0)
    T, t0, level = 120, 60, 5.0
    t = np.arange(T)
    # Same slope before and after -> slope_change should be ~0.
    y = 2.0 + 0.05 * t + level * (t >= t0) + rng.standard_normal(T) * 0.5
    df = pd.DataFrame({"y": y, "t": t})
    res = sp.its(df, y="y", time="t", intervention=t0)
    assert float(res.pvalue_slope) > 0.05
