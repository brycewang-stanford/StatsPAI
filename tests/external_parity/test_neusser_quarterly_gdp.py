"""Neusser (2016), section 17.4: quarterly GDP growth from annual data.

Opt-in: the book's data are not redistributed. Export sheet
``quartalsweise`` of ``BeispielQuartalschaetzung/.../data.xls`` to
``$STATSPAI_NEUSSER_DIR/_statspai/quarterly_gdp.csv`` with columns ``bip``
(annual GDP growth, in the fourth quarter only), ``ip`` (industrial
production growth), ``sent`` (consumer sentiment) and ``bip_seco`` (the
official quarterly estimate), 70 rows, and set ``STATSPAI_NEUSSER_DIR``.

The state is ``(q_t, q_{t-1}, q_{t-2}, q_{t-3})`` with ``q_t = phi q_{t-1}
+ v_t``; annual growth is ``a1 + (q_t + ... + q_{t-3}) / 4`` without error
in the fourth quarter, and each indicator is ``a_i + g_i q_t + w_i``. Nine
parameters and starting values as in the book's ``objfct_quarterly.m`` and
``main.m``.

Three references:

* the book's ``KalmanFilterTVP.m``, ported below. It scores a missing
  annual value as a ``N(0, 1)`` draw equal to zero, so its log-likelihood
  is ours minus ``0.5 log(2 pi)`` per missing value, exactly;
* statsmodels ``MLEModel`` on the same model: the likelihood at a common
  parameter vector to 1e-9, the optimum to 1e-5 (two different optimisers;
  statsmodels is started near the optimum, see the test);
* the numbers of our own first run, pinned at 1e-6, so that a change in
  the estimates is noticed. The book prints no parameter table to compare
  with, only the figure.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries.statespace import kalman_filter, statespace

ROOT = os.environ.get("STATSPAI_NEUSSER_DIR")
CSV = Path(ROOT) / "_statspai" / "quarterly_gdp.csv" if ROOT else None
pytestmark = pytest.mark.skipif(
    CSV is None or not CSV.is_file(),
    reason="set STATSPAI_NEUSSER_DIR to the book folder holding "
    "_statspai/quarterly_gdp.csv",
)

NAMES = ["a1", "a2", "a3", "g2", "g3", "log_r2", "log_r3", "phi", "log_q"]
SHIFT = np.eye(4, k=-1)


def build(p: np.ndarray) -> dict:
    F = SHIFT.copy()
    F[0, 0] = p[7]
    G = np.zeros((3, 4))
    G[0] = 0.25
    G[1, 0], G[2, 0] = p[3], p[4]
    return {
        "F": F,
        "G": G,
        "Q": np.diag([np.exp(p[8]), 0.0, 0.0, 0.0]),
        "R": np.diag([0.0, np.exp(p[5]), np.exp(p[6])]),
        "A": p[:3],
    }


def book_loglik(p: np.ndarray, y: np.ndarray) -> float:
    """objfct_quarterly.m + KalmanFilterTVP.m, line by line."""
    s = build(p)
    T = len(y)
    x = np.zeros(4)
    P = np.linalg.solve(
        np.eye(16) - np.kron(s["F"], s["F"]), s["Q"].reshape(-1, order="F")
    ).reshape(4, 4, order="F")
    P = (P + P.T) / 2
    loglh = 0.0
    for t in range(T):
        data, A, G, R = y[t].copy(), s["A"].copy(), s["G"].copy(), s["R"].copy()
        if (t + 1) % 4 != 0:
            data[0], A[0], R[0, 0] = 0.0, 0.0, 1.0
            G[0, :] = 0.0
        x = s["F"] @ x
        Ft = s["F"] @ P @ s["F"].T + s["Q"]
        err = data - A - G @ x
        ht = G @ Ft @ G.T + R
        ht = (ht + ht.T) / 2
        Kt = Ft @ G.T @ np.linalg.inv(ht)
        x = x + Kt @ err
        P = Ft - Kt @ G @ Ft
        P = (P + P.T) / 2
        loglh += (
            -1.5 * np.log(2 * np.pi)
            - 0.5 * np.log(np.linalg.det(ht))
            - 0.5 * err @ np.linalg.solve(ht, err)
        )
    return float(loglh)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    assert CSV is not None
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def y(frame) -> np.ndarray:
    return frame[["bip", "ip", "sent"]].to_numpy(float)


@pytest.fixture(scope="module")
def start(y) -> np.ndarray:
    """main.m: OLS on the yearly observations."""
    yr = y[3::4]
    n = len(yr)

    def ols(X: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.linalg.solve(X.T @ X, X.T @ v)

    phi = ols(np.column_stack([np.ones(n - 1), yr[:-1, 0]]), yr[1:, 0])
    X = np.column_stack([np.ones(n), yr[:, 0]])
    b2, b3 = ols(X, yr[:, 1]), ols(X, yr[:, 2])
    log01 = np.log(0.1)
    head = [yr[:, 0].mean(), b2[0], b3[0], b2[1], b3[1]]
    return np.array(head + [log01, log01, phi[1] ** 0.25, log01])


@pytest.fixture(scope="module")
def fit(y, start):
    return statespace(y, build, start, param_names=NAMES)


def test_estimates(fit, y):
    assert fit.converged
    assert y.shape == (70, 3) and fit.n_obs == 157
    assert fit.filter.init == "stationary"
    # pinned from the first run of this test (StatsPAI, 2026-10-06)
    assert fit.loglik == pytest.approx(-484.0015176048, abs=1e-6)
    pinned = [0.586501, 2.30996, -15.0558, 5.06748, 24.9381, 2.20758, 5.40211]
    pinned += [0.869832, -2.22175]
    np.testing.assert_allclose(fit.params.to_numpy(), pinned, rtol=2e-5)


def test_likelihood_differs_from_the_book_by_its_missing_data_constant(fit, y, start):
    n_missing = int(np.isnan(y).sum())
    assert n_missing == 53
    constant = 0.5 * np.log(2 * np.pi) * n_missing
    for p in (start, fit.params.to_numpy()):
        mine = kalman_filter(y, **build(p)).loglik
        # relative: at the starting values the log-likelihood is -2.2e5
        gap = mine - book_loglik(p, y)
        assert gap == pytest.approx(constant, abs=1e-11 * abs(mine))


def test_statsmodels_reaches_the_same_optimum(fit, y, start):
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    class Ref(MLEModel):
        def __init__(self, endog):
            super().__init__(endog, k_states=4, k_posdef=1)
            self["design", 0, :] = 0.25
            self["selection", 0, 0] = 1.0
            self["transition"] = SHIFT.copy()
            self.initialize_stationary()

        def update(self, params, **kwargs):
            params = super().update(params, **kwargs)
            self["obs_intercept", :, 0] = params[:3]
            self["design", 1, 0] = params[3]
            self["design", 2, 0] = params[4]
            self["obs_cov", 1, 1] = np.exp(params[5])
            self["obs_cov", 2, 2] = np.exp(params[6])
            self["transition", 0, 0] = params[7]
            self["state_cov", 0, 0] = np.exp(params[8])

    ref_model = Ref(y)
    est = fit.params.to_numpy()
    for p in (start, est):
        assert kalman_filter(y, **build(p)).loglik == pytest.approx(
            ref_model.loglike(p), rel=1e-9
        )
    # statsmodels' optimisers do not get there from the book's starting
    # values (log-likelihood -2.2e5; its BFGS stops at -5977 and its
    # Nelder-Mead leaves the stationary region), so it is started from a
    # point 10 to 30 per cent away from our estimate
    rng = np.random.default_rng(1)
    near = est + 0.09 * rng.normal(size=9) * np.maximum(np.abs(est), 1.0)
    assert ref_model.loglike(near) < fit.loglik - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ref_model.fit(near, method="bfgs", maxiter=5000, gtol=1e-9, disp=False)
        ref = ref_model.fit(
            ref.params, method="nm", maxiter=50000, xtol=1e-10, ftol=1e-13, disp=False
        )
    assert fit.loglik >= ref.llf - 1e-8
    assert fit.loglik == pytest.approx(ref.llf, abs=1e-7)
    np.testing.assert_allclose(est, ref.params, rtol=1e-5, atol=1e-5)
    approx = ref_model.smooth(est, cov_type="approx")
    np.testing.assert_allclose(fit.se.to_numpy(), approx.bse, rtol=1e-4)
    np.testing.assert_allclose(
        fit.filter.smoothed_state, approx.smoothed_state.T, atol=1e-9
    )


@pytest.mark.parametrize("method", ["nelder-mead", "l-bfgs-b"])
def test_other_optimisers_and_starts_agree(fit, y, start, method):
    # the MATLAB script runs csminwel and fminsearch; the surface has one
    # maximum that every route finds
    alt = statespace(y, build, start, method=method)
    assert alt.loglik == pytest.approx(fit.loglik, abs=1e-7)
    np.testing.assert_allclose(alt.params.to_numpy(), fit.params.to_numpy(), atol=1e-4)
    rng = np.random.default_rng(0)
    other = start + rng.normal(size=9) * np.array([0.5, 0.5, 5, 0.5, 5, 1, 1, 0, 1])
    other[7] = 0.2
    far = statespace(y, build, other, method=method)
    assert far.loglik == pytest.approx(fit.loglik, abs=1e-7)


def test_quarterly_path(fit, frame, y):
    q = fit.params["a1"] + fit.filter.smoothed_state[:, 0]
    # the four quarters of a year average to the annual figure, exactly
    for i in np.flatnonzero(np.isfinite(y[:, 0])):
        assert q[i - 3 : i + 1].mean() == pytest.approx(y[i, 0], abs=1e-9)
    seco = frame["bip_seco"].to_numpy(float)
    assert np.corrcoef(q, seco)[0, 1] == pytest.approx(0.46294, abs=1e-4)
    assert np.sqrt(np.mean((q - seco) ** 2)) == pytest.approx(0.98519, abs=1e-4)
