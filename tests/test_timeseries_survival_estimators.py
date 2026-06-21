"""Property tests for time-series / survival estimators whose result classes
were previously only covered indirectly: ``sp.garch``, ``sp.var``, ``sp.cox``,
``sp.kaplan_meier`` and ``sp.local_projections``.

Each test simulates from a known data-generating process and asserts the
structural law the estimator must obey — GARCH persistence below one, a
symmetric PSD VAR innovation covariance, a recovered Cox coefficient, a
monotone survival curve, and a local-projection impulse response that matches
the contemporaneous shock loading.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------------
# GARCH(1,1)
# --------------------------------------------------------------------------
def test_garch_persistence_and_positive_variance():
    rng = np.random.RandomState(0)
    n = 1500
    e = rng.randn(n)
    h = np.ones(n)
    for t in range(1, n):  # true alpha=0.10, beta=0.85 -> persistence 0.95
        h[t] = 0.05 + 0.10 * e[t - 1] ** 2 * h[t - 1] + 0.85 * h[t - 1]
    g = sp.garch(e * np.sqrt(h), p=1, q=1)

    assert g.omega > 0
    assert np.all(np.asarray(g.alpha) >= 0) and np.all(np.asarray(g.beta) >= 0)
    persistence = float(np.sum(g.alpha) + np.sum(g.beta))
    assert g.persistence == pytest.approx(persistence, abs=1e-8)
    # A covariance-stationary GARCH has persistence < 1.
    assert g.persistence < 1.0
    # Conditional variances are strictly positive.
    assert np.all(np.asarray(g.sigma2) > 0)


# --------------------------------------------------------------------------
# Vector autoregression
# --------------------------------------------------------------------------
def test_var_innovation_covariance_is_symmetric_psd():
    rng = np.random.RandomState(0)
    T = 400
    y1 = np.zeros(T)
    y2 = np.zeros(T)
    for t in range(1, T):
        y1[t] = 0.5 * y1[t - 1] + 0.1 * y2[t - 1] + rng.randn()
        y2[t] = 0.2 * y1[t - 1] + 0.3 * y2[t - 1] + rng.randn()
    v = sp.var(pd.DataFrame({"y1": y1, "y2": y2}), variables=["y1", "y2"], lags=1)

    sigma = np.asarray(v.sigma_u)
    assert sigma.shape == (2, 2)
    assert np.allclose(sigma, sigma.T)  # symmetric
    assert np.all(np.linalg.eigvalsh(sigma) > 0)  # positive definite
    assert v.n_obs == T - 1  # one observation lost to the single lag
    irf = v.irf(10, orthogonal=False)
    np.testing.assert_allclose(irf["irf"]["y1 -> y1"][0], 1.0)
    np.testing.assert_allclose(irf["irf"]["y1 -> y2"][0], 0.0)
    assert irf is not None and len(irf) > 0


# --------------------------------------------------------------------------
# Cox proportional hazards
# --------------------------------------------------------------------------
def test_cox_recovers_coefficient_and_concordance():
    rng = np.random.RandomState(0)
    n = 600
    x = rng.randn(n)
    # Hazard rises with x (beta = 0.7) -> shorter durations.
    dur = rng.exponential(np.exp(-0.7 * x))
    df = pd.DataFrame({"dur": dur, "event": np.ones(n, dtype=int), "x": x})
    c = sp.cox(duration="dur", event="event", data=df, x=["x"])

    assert float(c.params["x"]) == pytest.approx(0.7, abs=0.2)
    # Concordance is a probability strictly inside (0, 1) for an informative fit.
    assert 0.5 < c.concordance < 1.0
    assert (
        float(c.conf_int_lower["x"])
        < float(c.params["x"])
        < float(c.conf_int_upper["x"])
    )


# --------------------------------------------------------------------------
# Kaplan-Meier
# --------------------------------------------------------------------------
def test_kaplan_meier_survival_is_monotone():
    rng = np.random.RandomState(1)
    df = pd.DataFrame(
        {"dur": rng.exponential(5, 500), "event": rng.binomial(1, 0.8, 500)}
    )
    km = sp.kaplan_meier(df, duration="dur", event="event")
    table = km.survival_table
    surv = table["survival"].to_numpy()
    # A survival function starts at (or below) 1 and never increases.
    assert surv[0] <= 1.0 + 1e-9
    assert np.all(np.diff(surv) <= 1e-9)
    assert km.median_survival > 0


def test_kaplan_meier_by_group():
    rng = np.random.RandomState(2)
    n = 400
    g = rng.binomial(1, 0.5, n)
    df = pd.DataFrame(
        {
            "dur": rng.exponential(3 + 2 * g, n),
            "event": rng.binomial(1, 0.85, n),
            "grp": g,
        }
    )
    km = sp.kaplan_meier(df, duration="dur", event="event", group="grp")
    # Grouped estimation still produces a usable survival table.
    assert len(km.survival_table) > 0


# --------------------------------------------------------------------------
# Local projections
# --------------------------------------------------------------------------
def test_local_projections_impulse_matches_shock_loading():
    rng = np.random.RandomState(0)
    T = 600
    shock = rng.randn(T)
    y = np.zeros(T)
    for t in range(1, T):  # contemporaneous loading on the shock is 1.0
        y[t] = 0.6 * y[t - 1] + 1.0 * shock[t] + rng.randn() * 0.5
    lp = sp.local_projections(
        pd.DataFrame({"y": y, "shock": shock}),
        outcome="y",
        shock="shock",
        horizons=8,
    )
    irf = np.asarray(lp.irf)
    assert len(irf) == 9  # horizons 0..8
    assert irf[0] == pytest.approx(1.0, abs=0.25)
    lo = np.asarray(lp.ci_lower)
    hi = np.asarray(lp.ci_upper)
    assert np.all(lo <= irf + 1e-9) and np.all(irf <= hi + 1e-9)
    assert len(lp.to_frame()) == 9
