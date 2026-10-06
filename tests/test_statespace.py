"""``sp.kalman_filter`` and ``sp.statespace``: known truths, edges, errors.

The filter and smoother are checked against quantities that do not use a
recursion at all (the multivariate normal density and conditional moments
of the stacked system), against a Rauch-Tung-Striebel pass written out
separately, and against a faithful port of the MATLAB routines that
accompany Neusser (2016, section 17.4). Cross-language parity is in
``tests/reference_parity/test_statespace_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statspai.exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from statspai.timeseries._statespace_core import stationary_cov
from statspai.timeseries.statespace import (
    KalmanResult,
    StateSpaceResult,
    kalman_filter,
    statespace,
)

SHIFT = np.eye(4, k=-1)


def mixed_system(phi=0.8, q=0.6):
    """Quarterly AR(1) in companion form; singular Q and R."""
    F = SHIFT.copy()
    F[0, 0] = phi
    G = np.zeros((3, 4))
    G[0] = 0.25
    G[1, 0], G[2, 0] = 2.0, -1.5
    return {
        "F": F,
        "G": G,
        "Q": np.diag([q, 0.0, 0.0, 0.0]),
        "R": np.diag([0.0, 1.5, 2.5]),
        "A": np.array([0.5, 1.0, -2.0]),
    }


def simulate(sysm, T, seed, x0=None):
    rng = np.random.default_rng(seed)
    m = sysm["F"].shape[0]
    n = sysm["G"].shape[0]
    x = np.zeros(m) if x0 is None else np.asarray(x0, float)
    y = np.zeros((T, n))
    for t in range(T):
        x = sysm["F"] @ x + rng.multivariate_normal(np.zeros(m), sysm["Q"])
        y[t] = (
            sysm["A"] + sysm["G"] @ x + rng.multivariate_normal(np.zeros(n), sysm["R"])
        )
    return y


def mixed_data(T=24, seed=3):
    y = simulate(mixed_system(), T, seed)
    y[np.arange(T) % 4 != 3, 0] = np.nan
    return y


def stacked_moments(y, F, G, Q, R, A, x0, P0):
    """Likelihood and E[X | Y], Var[X | Y] from the joint normal law.

    No recursion over the data: the covariance of the stacked states is
    written out, the observed elements of Y are selected, and the usual
    conditioning formulas are applied.
    """
    T, n = y.shape
    m = F.shape[0]
    var = []
    mean = []
    P, mu = P0, x0
    for _ in range(T):
        P = F @ P @ F.T + Q
        mu = F @ mu
        var.append(P)
        mean.append(mu)
    Sxx = np.zeros((T * m, T * m))
    for s in range(T):
        block = var[s]
        for t in range(s, T):
            Sxx[s * m : (s + 1) * m, t * m : (t + 1) * m] = block
            Sxx[t * m : (t + 1) * m, s * m : (s + 1) * m] = block.T
            block = block @ F.T
    Gb = np.kron(np.eye(T), G)
    keep = np.isfinite(y).reshape(-1)
    Gb = Gb[keep]
    Syy = Gb @ Sxx @ Gb.T + np.kron(np.eye(T), R)[np.ix_(keep, keep)]
    mx = np.concatenate(mean)
    resid = y.reshape(-1)[keep] - (np.tile(A, T)[keep] + Gb @ mx)
    loglik = stats.multivariate_normal(np.zeros(keep.sum()), Syy).logpdf(resid)
    gain = Sxx @ Gb.T @ np.linalg.inv(Syy)
    cond_mean = (mx + gain @ resid).reshape(T, m)
    cond_var = Sxx - gain @ Gb @ Sxx
    blocks = np.array(
        [cond_var[t * m : (t + 1) * m, t * m : (t + 1) * m] for t in range(T)]
    )
    return float(loglik), cond_mean, blocks


def rts(out, F):
    """Rauch-Tung-Striebel smoother with a pseudo-inverse (Neusser 17.2)."""
    xf, Pf, xp, Pp = (
        out.filtered_state,
        out.filtered_cov,
        out.predicted_state,
        out.predicted_cov,
    )
    xs, Ps = xf.copy(), Pf.copy()
    for t in range(len(xf) - 2, -1, -1):
        J = Pf[t] @ F.T @ np.linalg.pinv(Pp[t + 1], hermitian=True)
        xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
    return xs, Ps


# --- the book's MATLAB routines, ported line by line ----------------------


def book_filter(data, A, G, F, R, Q):
    """KalmanFilterTVP.m: zero initial state, stationary variance."""
    T = data.shape[0]
    m = Q.shape[1]
    x = np.zeros(m)
    P = np.linalg.solve(
        np.eye(m * m) - np.kron(F[0], F[0]), Q[0].reshape(-1, order="F")
    ).reshape(m, m, order="F")
    P = (P + P.T) / 2
    states, variances, loglh = [], [], 0.0
    for t in range(T):
        x = F[t] @ x
        Ft = F[t] @ P @ F[t].T + Q[t]
        yhat = A[t] + G[t] @ x
        ht = G[t] @ Ft @ G[t].T + R[t]
        ht = (ht + ht.T) / 2
        Kt = Ft @ G[t].T @ np.linalg.inv(ht)
        x = x + Kt @ (data[t] - yhat)
        P = Ft - Kt @ G[t] @ Ft
        P = (P + P.T) / 2
        loglh += (
            -0.5 * data.shape[1] * np.log(2 * np.pi)
            - 0.5 * np.log(np.linalg.det(ht))
            - 0.5 * (data[t] - yhat) @ np.linalg.solve(ht, data[t] - yhat)
        )
        states.append(x)
        variances.append(P)
    return np.array(states), np.array(variances), float(loglh)


def book_smoother(Xt, Pt, F, Q, as_printed=True):
    """KalmanSmootherTVP.m; ``as_printed=False`` puts the transposes right."""
    T = len(Xt)
    XT, PT = Xt.copy(), Pt.copy()
    for t in range(T - 1, 0, -1):
        Pn = F[t] @ Pt[t - 1] @ F[t].T + Q[t]
        inv = np.linalg.inv(Pn)
        XT[t - 1] = Xt[t - 1] + Pt[t - 1] @ F[t].T @ inv @ (XT[t] - F[t] @ Xt[t - 1])
        if as_printed:
            PT[t - 1] = (
                Pt[t - 1] + Pt[t - 1] @ F[t] @ inv @ (PT[t] - Pn) @ inv @ Pt[t - 1]
            )
        else:
            J = Pt[t - 1] @ F[t].T @ inv
            PT[t - 1] = Pt[t - 1] + J @ (PT[t] - Pn) @ J.T
    return XT, PT


def book_arrays(y, sysm):
    """objfct_quarterly.m: a missing annual value becomes a N(0, 1) zero."""
    T = len(y)
    data = y.copy()
    A = np.tile(sysm["A"], (T, 1))
    G = np.tile(sysm["G"], (T, 1, 1))
    R = np.tile(sysm["R"], (T, 1, 1))
    for t in range(T):
        if (t + 1) % 4 != 0:
            data[t, 0] = 0.0
            G[t, 0, :] = 0.0
            A[t, 0] = 0.0
            R[t, 0, 0] = 1.0
    F = np.tile(sysm["F"], (T, 1, 1))
    Q = np.tile(sysm["Q"], (T, 1, 1))
    return data, A, G, F, R, Q


# --- known truths ----------------------------------------------------------


def test_ar2_likelihood_is_the_exact_gaussian_density():
    rng = np.random.default_rng(0)
    phi1, phi2, s2 = 0.5, 0.3, 1.3
    F = np.array([[phi1, phi2], [1.0, 0.0]])
    Q = np.diag([s2, 0.0])
    y = simulate(
        {
            "F": F,
            "G": np.array([[1.0, 0.0]]),
            "Q": Q,
            "R": np.zeros((1, 1)),
            "A": np.zeros(1),
        },
        40,
        1,
    )[:, 0]
    y[[7, 8, 20]] = np.nan
    out = kalman_filter(y, F=F, G=[1.0, 0.0], Q=Q, R=0.0)
    assert out.init == "stationary"
    # autocovariances from the stationary state covariance
    P = stationary_cov(F, Q)
    gam = [P[0, 0], P[0, 1]]
    for _ in range(40):
        gam.append(phi1 * gam[-1] + phi2 * gam[-2])
    idx = np.flatnonzero(np.isfinite(y))
    cov = np.array([[gam[abs(i - j)] for j in idx] for i in idx])
    truth = stats.multivariate_normal(np.zeros(len(idx)), cov).logpdf(y[idx])
    assert out.loglik == pytest.approx(truth, rel=1e-11)
    assert out.n_obs == 37 and out.n_dates == 37
    assert rng is not None


def test_filter_and_smoother_equal_the_joint_normal_conditioning():
    # singular Q and R, one observable seen every fourth date
    y = mixed_data()
    s = mixed_system()
    out = kalman_filter(y, **s)
    P0 = stationary_cov(s["F"], s["Q"])
    loglik, mean, var = stacked_moments(
        y, s["F"], s["G"], s["Q"], s["R"], s["A"], np.zeros(4), P0
    )
    assert out.loglik == pytest.approx(loglik, rel=1e-10)
    np.testing.assert_allclose(out.smoothed_state, mean, atol=1e-9)
    np.testing.assert_allclose(out.smoothed_cov, var, atol=1e-9)
    # the last filtered moments are the last smoothed ones
    np.testing.assert_allclose(out.filtered_state[-1], mean[-1], atol=1e-9)
    # an exactly observed average has no smoothing error
    w = np.full(4, 0.25)
    assert abs(w @ out.smoothed_cov[3] @ w) < 1e-12
    assert w @ out.smoothed_state[3] + 0.5 == pytest.approx(y[3, 0], abs=1e-10)


def test_user_initial_state_and_scattered_missing_values():
    rng = np.random.default_rng(5)
    F = np.array([[0.7, 0.2], [-0.1, 0.5]])
    s = {
        "F": F,
        "G": np.array([[1.0, 0.5], [0.3, 1.0]]),
        "Q": np.array([[1.0, 0.3], [0.3, 0.6]]),
        "R": np.array([[0.5, 0.2], [0.2, 0.8]]),
        "A": np.array([1.0, -1.0]),
    }
    y = simulate(s, 20, 6)
    y[rng.random(y.shape) < 0.25] = np.nan
    y[[0, 9]] = np.nan
    x0, P0 = np.array([0.4, -1.0]), np.array([[2.0, 0.5], [0.5, 1.0]])
    out = kalman_filter(y, x0=x0, P0=P0, **s)
    assert out.init == "user"
    loglik, mean, var = stacked_moments(y, F, s["G"], s["Q"], s["R"], s["A"], x0, P0)
    assert out.loglik == pytest.approx(loglik, rel=1e-11)
    np.testing.assert_allclose(out.smoothed_state, mean, atol=1e-10)
    np.testing.assert_allclose(out.smoothed_cov, var, atol=1e-10)
    # a date with nothing observed is a pure prediction step
    np.testing.assert_array_equal(out.filtered_state[9], out.predicted_state[9])
    np.testing.assert_array_equal(out.filtered_cov[0], F @ P0 @ F.T + s["Q"])
    assert out.loglik_obs[9] == 0.0
    assert np.all(np.isnan(out.innovations[9]))
    assert out.n_obs == int(np.isfinite(y).sum())


def test_smoother_agrees_with_rauch_tung_striebel():
    y = mixed_data(T=40, seed=8)
    s = mixed_system()
    out = kalman_filter(y, **s)
    xs, Ps = rts(out, s["F"])
    np.testing.assert_allclose(out.smoothed_state, xs, atol=1e-10)
    np.testing.assert_allclose(out.smoothed_cov, Ps, atol=1e-10)


def test_standardised_innovations_are_cholesky_whitened():
    y = mixed_data(T=16)
    out = kalman_filter(y, **mixed_system())
    for t in (2, 3):
        seen = np.isfinite(y[t])
        S = out.innovations_cov[t][np.ix_(seen, seen)]
        e = np.linalg.solve(np.linalg.cholesky(S), out.innovations[t][seen])
        np.testing.assert_allclose(out.std_innovations[t][seen], e, atol=1e-12)
        quad = out.innovations[t][seen] @ np.linalg.solve(S, out.innovations[t][seen])
        ll = -0.5 * (seen.sum() * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + quad)
        assert out.loglik_obs[t] == pytest.approx(ll, rel=1e-12)
    assert np.isnan(out.std_innovations[2, 0]) and np.isnan(
        out.innovations_cov[2, 0, 1]
    )


# --- the book's code -------------------------------------------------------


def test_book_filter_differs_by_the_constant_of_its_missing_data_device():
    y = mixed_data(T=48, seed=11)
    s = mixed_system()
    out = kalman_filter(y, **s)
    data, A, G, F, R, Q = book_arrays(y, s)
    Xt, Pt, loglh = book_filter(data, A, G, F, R, Q)
    np.testing.assert_allclose(out.filtered_state, Xt, atol=1e-12)
    np.testing.assert_allclose(out.filtered_cov, Pt, atol=1e-12)
    # each missing value was scored as a N(0, 1) draw equal to zero
    n_missing = int(np.isnan(y).sum())
    assert n_missing == 36
    assert out.loglik - loglh == pytest.approx(
        0.5 * np.log(2 * np.pi) * n_missing, abs=1e-10
    )


def test_book_smoother_means_are_right_and_its_variance_line_is_not():
    y = mixed_data(T=48, seed=11)
    s = mixed_system()
    out = kalman_filter(y, **s)
    data, A, G, F, R, Q = book_arrays(y, s)
    Xt, Pt, _ = book_filter(data, A, G, F, R, Q)
    XT, PT = book_smoother(Xt, Pt, F, Q, as_printed=True)
    np.testing.assert_allclose(out.smoothed_state, XT, atol=1e-10)
    # KalmanSmootherTVP.m has Pt*F*inv(Ptp1)*(...)*inv(Ptp1)*Pt where the
    # recursion is Pt*F'*inv(Ptp1)*(...)*inv(Ptp1)*F*Pt
    assert np.max(np.abs(out.smoothed_cov - PT)) > 1e-2
    _, PT_fixed = book_smoother(Xt, Pt, F, Q, as_printed=False)
    np.testing.assert_allclose(out.smoothed_cov, PT_fixed, atol=1e-10)


# --- initial state ---------------------------------------------------------


def test_auto_init_is_diffuse_for_a_random_walk_and_says_so():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(size=60)) + rng.normal(size=60)
    out = kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=1.0, burn=1)
    assert out.init == "diffuse"
    assert "diffuse" in out.summary()
    assert out.n_obs == 59
    # the approximation does not matter after the first date
    big = kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=1.0, burn=1, kappa=1e9)
    np.testing.assert_allclose(out.smoothed_state, big.smoothed_state, atol=1e-5)
    assert out.loglik == pytest.approx(big.loglik, abs=1e-5)
    # exact diffuse likelihood of the local level: condition on y_1
    exact = kalman_filter(y[1:], F=1.0, G=1.0, Q=1.0, R=1.0, x0=y[0], P0=1.0)
    assert out.loglik == pytest.approx(exact.loglik, abs=1e-5)
    with pytest.raises(MethodIncompatibility, match="unit circle"):
        kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=1.0, init="stationary")


def test_stationary_init_needs_constant_matrices():
    y = np.zeros(5)
    F = np.full((5, 1, 1), 0.5)
    with pytest.raises(MethodIncompatibility, match="constant"):
        kalman_filter(y, F=F, G=1.0, Q=1.0, R=1.0)
    out = kalman_filter(y, F=F, G=1.0, Q=1.0, R=1.0, P0=[[1.0]])
    ref = kalman_filter(y, F=0.5, G=1.0, Q=1.0, R=1.0, P0=[[1.0]])
    assert out.loglik == ref.loglik


# --- inputs, forecasts, errors ----------------------------------------------


def test_inputs_names_and_frames():
    y = mixed_data(T=12)
    df = pd.DataFrame(
        y,
        columns=["gdp", "ip", "sent"],
        index=pd.period_range("2000Q1", periods=12, freq="Q"),
    )
    names = ["q", "q1", "q2", "q3"]
    out = kalman_filter(
        ["gdp", "ip", "sent"], data=df, state_names=names, **mixed_system()
    )
    ref = kalman_filter(y, **mixed_system())
    assert out.loglik == ref.loglik
    assert isinstance(out, KalmanResult)
    frame = out.states("filtered")
    assert list(frame.columns) == names + [f"{n}_se" for n in names]
    assert frame.index.equals(df.index)
    assert out.to_dict()["n_obs"] == out.n_obs
    assert "Log likelihood" in out.summary()
    with pytest.raises(MethodIncompatibility):
        out.states("nowcast")
    with pytest.raises(MethodIncompatibility, match="not in data"):
        kalman_filter("nope", data=df, **mixed_system())
    no_smooth = kalman_filter(y, smooth=False, **mixed_system())
    assert no_smooth.smoothed_state is None
    with pytest.raises(MethodIncompatibility, match="smoother"):
        no_smooth.states()


def test_forecast_of_an_ar1_plus_noise():
    rng = np.random.default_rng(4)
    y = rng.normal(size=30)
    out = kalman_filter(y, F=0.6, G=2.0, Q=0.5, R=0.3, A=1.0)
    fc = out.forecast(5)
    x, P = out.filtered_state[-1, 0], out.filtered_cov[-1, 0, 0]
    for h in range(1, 6):
        mean = 0.6**h * x
        mse = 0.6 ** (2 * h) * P + 0.5 * sum(0.6 ** (2 * j) for j in range(h))
        assert fc["state"].iloc[h - 1, 0] == pytest.approx(mean, rel=1e-12)
        assert fc["state_se"].iloc[h - 1, 0] == pytest.approx(np.sqrt(mse), rel=1e-12)
        assert fc["obs"].iloc[h - 1, 0] == pytest.approx(1.0 + 2.0 * mean, rel=1e-12)
        assert fc["obs_cov"][h - 1, 0, 0] == pytest.approx(4.0 * mse + 0.3, rel=1e-12)
    with pytest.raises(MethodIncompatibility):
        out.forecast(0)
    with pytest.raises(MethodIncompatibility, match="Unknown"):
        out.forecast(1, Z=1.0)


def test_forecast_with_time_varying_loadings_needs_their_future_values():
    rng = np.random.default_rng(6)
    x = rng.normal(size=25)
    y = 2.0 * x + rng.normal(size=25)
    out = kalman_filter(y, F=1.0, G=x[:, None], Q=0.01, R=1.0, P0=[[10.0]])
    with pytest.raises(MethodIncompatibility, match="forecast dates"):
        out.forecast(2)
    fc = out.forecast(2, G=np.array([[1.0], [3.0]]))
    b = out.filtered_state[-1, 0]
    np.testing.assert_allclose(fc["obs"].to_numpy()[:, 0], [b, 3.0 * b], rtol=1e-12)


def test_bad_inputs_fail_loudly():
    y = np.zeros((6, 2))
    ok = {"F": np.eye(2) * 0.5, "G": np.eye(2), "Q": np.eye(2), "R": np.eye(2)}
    with pytest.raises(MethodIncompatibility, match="G has shape"):
        kalman_filter(y, **{**ok, "G": np.eye(3)})
    with pytest.raises(MethodIncompatibility, match="symmetric"):
        kalman_filter(y, **{**ok, "Q": np.array([[1.0, 0.5], [0.0, 1.0]])})
    with pytest.raises(MethodIncompatibility, match="time slices"):
        kalman_filter(y, **{**ok, "R": np.tile(np.eye(2), (4, 1, 1))})
    with pytest.raises(MethodIncompatibility, match="NaN"):
        kalman_filter(y, **{**ok, "F": np.full((2, 2), np.nan)})
    with pytest.raises(MethodIncompatibility, match="init="):
        kalman_filter(y, init="nonsense", **ok)
    with pytest.raises(MethodIncompatibility, match="P0"):
        kalman_filter(y, P0=np.eye(3), **ok)
    with pytest.raises(MethodIncompatibility, match="infinite"):
        kalman_filter(np.array([1.0, np.inf]), F=0.5, G=1.0, Q=1.0, R=1.0)
    with pytest.raises(MethodIncompatibility, match="burn"):
        kalman_filter(y, burn=6, **ok)
    with pytest.raises(DataInsufficient):
        kalman_filter(np.zeros((0, 1)), F=0.5, G=1.0, Q=1.0, R=1.0)
    # two copies of one noiseless observable: G P G' + R is singular
    with pytest.raises(MethodIncompatibility, match="not positive definite"):
        kalman_filter(y, F=0.5, G=[[1.0], [1.0]], Q=1.0, R=np.zeros((2, 2)))


# --- maximum likelihood -----------------------------------------------------


def local_level(T=300, seed=1):
    rng = np.random.default_rng(seed)
    return np.cumsum(np.sqrt(0.5) * rng.normal(size=T)) + np.sqrt(2.0) * rng.normal(
        size=T
    )


def build_level(th):
    return {"F": 1.0, "G": 1.0, "Q": np.exp(th[0]), "R": np.exp(th[1])}


def test_statespace_local_level_and_delta_method():
    y = local_level()
    fit = statespace(
        y,
        build_level,
        [0.0, 0.0],
        init="diffuse",
        burn=1,
        param_names=["log_q", "log_r"],
        transform=lambda th: {"q": np.exp(th[0]), "r": np.exp(th[1])},
    )
    assert isinstance(fit, StateSpaceResult)
    assert fit.converged and fit.model_info["hessian_pd"]
    assert max(abs(g) for g in fit.model_info["gradient"]) < 1e-4
    # truth (0.5, 2.0) within three standard errors
    tr = fit.transformed
    assert abs(tr.loc["q", "estimate"] - 0.5) < 3 * tr.loc["q", "se"]
    assert abs(tr.loc["r", "estimate"] - 2.0) < 3 * tr.loc["r", "se"]
    # delta method for exp(): se = exp(theta) * se(theta)
    np.testing.assert_allclose(
        tr["se"].to_numpy(),
        np.exp(fit.params.to_numpy()) * fit.se.to_numpy(),
        rtol=1e-6,
    )
    assert fit.aic == pytest.approx(-2 * fit.loglik + 4)
    assert fit.bic == pytest.approx(-2 * fit.loglik + 2 * np.log(299))
    assert fit.loglik == pytest.approx(fit.filter.loglik, rel=1e-12)
    assert fit.loglik >= fit.model_info["start_loglik"]
    assert "Transformed" in fit.summary()
    assert fit.forecast(3)["obs"].shape == (3, 1)
    assert fit.to_dict()["converged"] is True
    assert fit.coef.equals(fit.params)


@pytest.mark.parametrize("method", ["l-bfgs-b", "nelder-mead"])
def test_optimisers_reach_the_same_maximum(method):
    y = local_level(T=150)
    ref = statespace(y, build_level, [0.0, 0.0], init="diffuse", burn=1)
    alt = statespace(y, build_level, [1.0, -1.0], init="diffuse", burn=1, method=method)
    assert alt.converged
    assert alt.loglik == pytest.approx(ref.loglik, abs=1e-8)
    np.testing.assert_allclose(alt.params.to_numpy(), ref.params.to_numpy(), atol=1e-4)


def test_engines_agree():
    y = mixed_data(T=40, seed=2)

    def build(th):
        return mixed_system(phi=th[0], q=np.exp(th[1]))

    a = statespace(y, build, [0.5, 0.0], engine="numpy")
    b = statespace(y, build, [0.5, 0.0], engine="numba")
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), atol=1e-7)
    np.testing.assert_allclose(a.se.to_numpy(), b.se.to_numpy(), rtol=1e-5)


def test_alternative_covariances_are_close_under_a_correct_model():
    y = local_level(T=600, seed=3)
    se = {
        vce: statespace(y, build_level, [0.0, 0.0], init="diffuse", burn=1, vce=vce).se
        for vce in ("hessian", "opg", "robust")
    }
    for vce in ("opg", "robust"):
        np.testing.assert_allclose(
            se[vce].to_numpy(), se["hessian"].to_numpy(), rtol=0.3
        )


def test_unidentified_parameter_warns_and_reports_no_standard_errors():
    y = local_level(T=120)

    def build(th):
        # only the sum th[1] + th[2] enters
        return {"F": 1.0, "G": 1.0, "Q": np.exp(th[0]), "R": np.exp(th[1] + th[2])}

    with pytest.warns(ConvergenceWarning, match="not positive definite"):
        fit = statespace(y, build, [0.0, 0.0, 0.0], init="diffuse", burn=1)
    assert not fit.converged
    assert not fit.model_info["hessian_pd"]
    assert np.all(np.isnan(fit.se.to_numpy()))
    assert np.isfinite(fit.loglik)


def test_statespace_bad_inputs_fail_loudly():
    y = local_level(T=40)
    with pytest.raises(MethodIncompatibility, match="dict"):
        statespace(y, lambda th: 1.0, [0.0, 0.0])
    with pytest.raises(MethodIncompatibility, match="unknown keys"):
        statespace(y, lambda th: {**build_level(th), "H": 1.0}, [0.0, 0.0])
    with pytest.raises(MethodIncompatibility, match="missing"):
        statespace(y, lambda th: {"F": 1.0, "G": 1.0, "Q": 1.0}, [0.0])
    with pytest.raises(MethodIncompatibility, match="not finite"):
        statespace(
            y,
            lambda th: {"F": 1.0, "G": 1.0, "Q": -1.0, "R": -5.0},
            [0.0, 0.0],
            init="diffuse",
            kappa=1.0,
        )
    with pytest.raises(MethodIncompatibility, match="param_names"):
        statespace(y, build_level, [0.0, 0.0], param_names=["a"])
    for bad in ({"method": "newton"}, {"vce": "hac"}, {"engine": "c"}, {"burn": 40}):
        with pytest.raises(MethodIncompatibility):
            statespace(y, build_level, [0.0, 0.0], init="diffuse", **bad)
    with pytest.raises(MethodIncompatibility, match="finite"):
        statespace(y, build_level, [np.nan, 0.0])
    with pytest.raises(DataInsufficient):
        statespace(y[:2], build_level, [0.0, 0.0], init="diffuse")


def test_plots_return_figures():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    y = local_level(T=60)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = statespace(y, build_level, [0.0, 0.0], init="diffuse", burn=1)
    fig = fit.plot()
    assert len(fig.axes) == 1
    assert len(fit.filter.plot("filtered").axes) == 1


def test_auto_init_is_settled_at_the_start_and_keeps_the_stationary_region():
    rng = np.random.default_rng(9)
    y = np.zeros(200)
    for t in range(1, 200):
        y[t] = 0.97 * y[t - 1] + rng.normal()

    def build(th):
        return {"F": th[0], "G": 1.0, "Q": np.exp(th[1]), "R": 0.0}

    fit = statespace(y, build, [0.5, 0.0])
    assert fit.filter.init == "stationary"
    assert abs(fit.params.iloc[0]) < 1.0
    # an explosive root is outside the parameter space, not a diffuse model
    with pytest.raises(MethodIncompatibility, match="unit circle"):
        kalman_filter(y, init="stationary", **build([1.02, 0.0]))
    # started outside, the rule is diffuse and stays diffuse
    out = statespace(y, build, [1.0, 0.0], burn=1)
    assert out.filter.init == "diffuse"


# --- exact diffuse initial state ---------------------------------------------


def diffuse_gls(y, F, G, Q, R, A, x0, Pstar, Pinf):
    """Smoothed moments and diffuse likelihood without any recursion.

    ``X_0 = x0 + B delta + e`` with ``B B' = Pinf``, ``Var(e) = Pstar`` and
    ``delta`` an unknown constant. Stack the sample, estimate ``delta`` by
    generalised least squares and add its variance to the conditional one:
    that is the posterior under a flat prior on ``delta``. The diffuse
    log-likelihood is the density of the data at the estimate less
    ``0.5 log det(D' S^-1 D)``, with ``D`` the loadings of the data on
    ``delta``. System matrices carry a leading time axis of length T.
    """
    T, n = y.shape
    m = F.shape[-1]
    w, U = np.linalg.eigh(Pinf)
    B0 = U[:, w > 1e-10] * np.sqrt(w[w > 1e-10])
    k = B0.shape[1]
    mu = np.zeros((T, m))
    B = np.zeros((T, m, k))
    W = np.zeros((T, m, T + 1, m))  # loading of X_t on (e, V_1, ..., V_T)
    cmu, cB = x0, B0
    cW = np.zeros((m, T + 1, m))
    cW[:, 0, :] = np.eye(m)
    for t in range(T):
        cmu, cB = F[t] @ cmu, F[t] @ cB
        cW = np.einsum("ab,bjc->ajc", F[t], cW)
        cW[:, t + 1, :] += np.eye(m)
        mu[t], B[t], W[t] = cmu, cB, cW
    shocks = np.zeros(((T + 1) * m, (T + 1) * m))
    shocks[:m, :m] = Pstar
    Gb = np.zeros((T * n, T * m))
    Rb = np.zeros((T * n, T * n))
    for t in range(T):
        lo = (t + 1) * m
        shocks[lo : lo + m, lo : lo + m] = Q[t]
        Gb[t * n : (t + 1) * n, t * m : (t + 1) * m] = G[t]
        Rb[t * n : (t + 1) * n, t * n : (t + 1) * n] = R[t]
    Wf = W.reshape(T * m, (T + 1) * m)
    Sxx = Wf @ shocks @ Wf.T
    seen = np.isfinite(y).reshape(-1)
    Gb, Rb = Gb[seen], Rb[np.ix_(seen, seen)]
    yy = (y - A).reshape(-1)[seen]
    Bx = B.reshape(T * m, k)
    c, D = Gb @ mu.reshape(-1), Gb @ Bx
    Syy = Gb @ Sxx @ Gb.T + Rb
    Sxy = Sxx @ Gb.T
    Si = np.linalg.inv(Syy)
    info = D.T @ Si @ D
    Vd = np.linalg.inv(info)
    delta = Vd @ D.T @ Si @ (yy - c)
    e = yy - c - D @ delta
    xs = mu.reshape(-1) + Bx @ delta + Sxy @ Si @ e
    H = Bx - Sxy @ Si @ D
    V = (Sxx - Sxy @ Si @ Sxy.T + H @ Vd @ H.T).reshape(T, m, T, m)
    ll = -0.5 * (
        seen.sum() * np.log(2 * np.pi)
        + np.linalg.slogdet(Syy)[1]
        + np.linalg.slogdet(info)[1]
        + e @ Si @ e
    )
    return xs.reshape(T, m), np.array([V[t, :, t, :] for t in range(T)]), float(ll)


def diffuse_case(name, T=14, seed=7):
    """Awkward models for the exact diffuse filter, with a time axis."""
    rng = np.random.default_rng(seed)
    if name == "coupled":
        # diffuse states feed each other and the stationary one is separate;
        # correlated measurement errors; whole and partial rows missing
        F = np.array([[1.0, 0.7, 0.0], [0.3, 1.1, 0.2], [0.0, 0.0, 0.5]])
        G = rng.normal(size=(2, 3))
        Q = np.diag([0.3, 0.2, 0.4])
        R = np.array([[0.5, 0.2], [0.2, 0.6]])
        sysm = {"F": F, "G": G, "Q": Q, "R": R, "A": np.array([0.3, -0.2])}
        y = rng.normal(size=(T, 2)).cumsum(axis=0)
        y[0, 1] = y[2, 0] = y[6, 0] = np.nan
        y[1, :] = np.nan
        return y, sysm, [True, True, False]
    if name == "varying":
        # every matrix time-varying, first date missing, det F_1 != 1
        ang = 0.3 + 0.2 * np.arange(T)
        F = np.array([[[0.9 + 0.2 * np.sin(a), 0.1], [0.2, 1.1]] for a in ang])
        Q = np.array([[[0.5 + 0.3 * np.cos(a) ** 2, 0.1], [0.1, 0.4]] for a in ang])
        R = np.array([[[0.3 + 0.2 * np.sin(a) ** 2]] for a in ang])
        G = np.array([[[1.0, np.cos(2 * a)]] for a in ang])
        A = np.array([[0.05 * t] for t in range(T)])
        y = rng.normal(size=(T, 1)).cumsum(axis=0)
        y[[0, 4], 0] = np.nan
        return y, {"F": F, "G": G, "Q": Q, "R": R, "A": A}, [True, True]
    if name == "slow":
        # a loading that is zero at first: observations that carry no
        # information on the diffuse state arrive inside the diffuse period
        F = np.diag([1.0, 0.6])
        G = np.zeros((T, 1, 2))
        G[:, 0, 1] = 1.0
        G[3:, 0, 0] = np.linspace(0.5, 2.0, T - 3)
        sysm = {"F": F, "G": G, "Q": np.diag([0.2, 0.5]), "R": 0.3, "A": [0.0]}
        return rng.normal(size=(T, 1)), sysm, [True, False]
    # three observables on two diffuse trends: more data than diffuse
    # states at the first date, and a singular measurement covariance
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    G = np.array([[1.0, 0.0], [1.0, 0.5], [0.3, -1.0]])
    R = np.diag([0.4, 0.0, 0.7])
    sysm = {"F": F, "G": G, "Q": np.diag([0.3, 0.05]), "R": R, "A": np.zeros(3)}
    y = rng.normal(size=(T, 3)).cumsum(axis=0)
    y[3, 1] = np.nan
    return y, sysm, [True, True]


def with_time_axis(sysm, T, n):
    out = {}
    for key, ndim in (("F", 2), ("G", 2), ("Q", 2), ("R", 2), ("A", 1)):
        arr = np.asarray(sysm[key], dtype=float)
        if key == "R" and arr.ndim == 0:
            arr = arr.reshape(1, 1)
        out[key] = arr if arr.ndim == ndim + 1 else np.tile(arr, (T,) + (1,) * ndim)
    return out


@pytest.mark.parametrize("name", ["coupled", "varying", "slow", "wide"])
def test_exact_diffuse_equals_generalised_least_squares(name):
    y, sysm, flags = diffuse_case(name)
    T, n = y.shape
    m = len(flags)
    x0 = np.linspace(0.1, 0.3, m)
    out = kalman_filter(y, x0=x0, diffuse=flags, **sysm)
    assert out.init == "exact"
    full = with_time_axis(sysm, T, n)
    xs, V, ll = diffuse_gls(y, x0=x0, Pstar=out.P0, Pinf=out.P0_inf, **full)
    # two routes to the same closed form; the inverse of the stacked
    # covariance (order T n, up to 42) is good to about 1e-8
    np.testing.assert_allclose(out.smoothed_state, xs, atol=1e-7)
    np.testing.assert_allclose(out.smoothed_cov, V, atol=1e-7)
    assert out.loglik == pytest.approx(ll, abs=1e-7)
    assert out.n_diffuse == sum(flags)
    # filtered moments at t are the smoothed ones of the sample cut at t,
    # once the diffuse state is identified
    done = int(np.flatnonzero(~np.any(out.filtered_cov_inf, axis=(1, 2)))[0])
    for t in (done, done + 2, T - 1):
        cut = {k: v[: t + 1] for k, v in full.items()}
        xt, Vt, _ = diffuse_gls(y[: t + 1], x0=x0, Pstar=out.P0, Pinf=out.P0_inf, **cut)
        np.testing.assert_allclose(out.filtered_state[t], xt[-1], atol=1e-7)
        np.testing.assert_allclose(out.filtered_cov[t], Vt[-1], atol=1e-7)


@pytest.mark.parametrize("name", ["coupled", "varying", "slow", "wide"])
def test_large_variance_filter_converges_to_the_exact_one(name):
    y, sysm, flags = diffuse_case(name)
    out = kalman_filter(y, diffuse=flags, **sysm)
    done = int(np.flatnonzero(~np.any(out.filtered_cov_inf, axis=(1, 2)))[0])
    gaps = []
    for kappa in (1e3, 1e5):
        P0 = out.P0 + kappa * out.P0_inf
        approx = kalman_filter(y, P0=P0, **sysm)
        err = [
            np.max(np.abs(approx.smoothed_state - out.smoothed_state)),
            np.max(np.abs(approx.filtered_state[done:] - out.filtered_state[done:])),
            np.max(np.abs(approx.filtered_cov[done:] - out.filtered_cov[done:])),
            abs(approx.loglik + 0.5 * out.n_diffuse * np.log(kappa) - out.loglik),
        ]
        gaps.append(max(err))
    # the error is of order 1 / kappa: a hundred times smaller. (The
    # smoothed covariance of the approximation is left out: it is a
    # difference of terms of order kappa and loses digits as kappa grows.)
    assert gaps[0] < 1.0
    assert gaps[0] / 150 < gaps[1] < gaps[0] / 50
    # inside the diffuse period too, the exact predicted and filtered
    # states are the limit for a diffuse X_0
    P0 = out.P0 + 1e8 * out.P0_inf
    approx = kalman_filter(y, P0=P0, **sysm)
    np.testing.assert_allclose(approx.filtered_state, out.filtered_state, atol=1e-4)
    np.testing.assert_allclose(approx.predicted_state, out.predicted_state, atol=1e-4)


def test_exact_local_level_conditions_on_the_first_observation():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.normal(size=60)) + rng.normal(size=60)
    out = kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=0.7, init="exact")
    assert out.init == "exact" and out.n_diffuse == 1
    assert "Exact diffuse" in out.summary()
    # after y_1 the level is N(y_1, R); the absorbed observation leaves
    # -0.5 log(2 pi) - 0.5 log F_inf with F_inf = 1
    rest = kalman_filter(y[1:], F=1.0, G=1.0, Q=1.0, R=0.7, x0=y[0], P0=0.7)
    assert out.loglik == pytest.approx(rest.loglik - 0.5 * np.log(2 * np.pi), abs=1e-11)
    np.testing.assert_allclose(out.filtered_state[1:], rest.filtered_state, atol=1e-12)
    np.testing.assert_allclose(out.smoothed_cov[1:], rest.smoothed_cov, atol=1e-12)
    assert out.filtered_state[0, 0] == pytest.approx(y[0], abs=1e-14)
    assert out.filtered_cov[0, 0, 0] == pytest.approx(0.7, abs=1e-14)
    assert out.predicted_cov_inf[0, 0, 0] == 1.0 and not out.predicted_cov_inf[1:].any()
    # predicted standard error is infinite until the level is identified
    pred = out.states("predicted")
    assert np.isinf(pred["x1_se"].iloc[0]) and np.isfinite(pred["x1_se"].iloc[1:]).all()
    assert np.isnan(out.std_innovations[0, 0])
    assert np.isfinite(out.std_innovations[1:]).all()
    # the default keeps the large-variance approximation
    assert kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=0.7).init == "diffuse"


def test_partly_diffuse_initial_state_and_its_arguments():
    rng = np.random.default_rng(5)
    y = rng.normal(size=30).cumsum()
    sysm = {"F": np.diag([1.0, 0.7]), "G": [1.0, 1.0], "Q": np.diag([0.1, 0.5])}
    out = kalman_filter(y, R=0.2, diffuse=[True, False], **sysm)
    np.testing.assert_allclose(out.P0, np.diag([0.0, 0.5 / (1 - 0.49)]), atol=1e-14)
    np.testing.assert_array_equal(out.P0_inf, np.diag([1.0, 0.0]))
    # a given P0 supplies the block of the other states; its diffuse rows
    # and columns are irrelevant
    P0 = np.array([[9.0, 0.4], [0.4, 0.5 / (1 - 0.49)]])
    same = kalman_filter(y, R=0.2, P0=P0, diffuse=[True, False], **sysm)
    np.testing.assert_allclose(same.smoothed_state, out.smoothed_state, atol=1e-13)
    assert same.loglik == pytest.approx(out.loglik, abs=1e-12)
    # no diffuse state at all: the ordinary filter
    none = kalman_filter(y, R=0.2, P0=np.eye(2), diffuse=[False, False], **sysm)
    plain = kalman_filter(y, R=0.2, P0=np.eye(2), **sysm)
    assert none.n_diffuse == 0
    assert none.loglik == pytest.approx(plain.loglik, abs=1e-12)
    np.testing.assert_allclose(none.smoothed_cov, plain.smoothed_cov, atol=1e-13)
    with pytest.raises(MethodIncompatibility, match="one per state"):
        kalman_filter(y, R=0.2, diffuse=[True], **sysm)
    with pytest.raises(MethodIncompatibility, match="diffuse="):
        kalman_filter(y, R=0.2, diffuse=[True, False], init="stationary", **sysm)
    with pytest.raises(MethodIncompatibility, match="stable"):
        kalman_filter(y, R=0.2, diffuse=[False, True], **sysm)
    coupled = {**sysm, "F": np.array([[1.0, 0.0], [0.2, 0.7]])}
    with pytest.raises(MethodIncompatibility, match="depends on a diffuse"):
        kalman_filter(y, R=0.2, diffuse=[True, False], **coupled)
    varying = {**sysm, "Q": np.tile(np.diag([0.1, 0.5]), (30, 1, 1))}
    with pytest.raises(MethodIncompatibility, match="constant F and Q"):
        kalman_filter(y, R=0.2, diffuse=[True, False], **varying)
    # a noiseless observation that the state cannot explain
    with pytest.raises(MethodIncompatibility, match="not positive definite"):
        kalman_filter(y, F=1.0, G=np.zeros((30, 1)), Q=1.0, R=0.0, init="exact")


def test_unidentified_diffuse_state_is_reported():
    # the slope of a local linear trend needs two observations
    y = np.array([1.0, np.nan, np.nan])
    sysm = {"F": [[1.0, 1.0], [0.0, 1.0]], "G": [1.0, 0.0], "Q": np.eye(2), "R": 1.0}
    out = kalman_filter(y, init="exact", **sysm)
    assert out.n_diffuse == 1
    assert "does not pin down" in out.summary()
    assert np.isinf(out.states("filtered").iloc[-1][["x1_se", "x2_se"]]).all()
    with pytest.raises(MethodIncompatibility, match="does not pin down"):
        out.forecast(2)
    assert np.all(np.isfinite(out.smoothed_state))
    empty = kalman_filter(np.full(4, np.nan), init="exact", **sysm)
    assert empty.loglik == 0.0 and empty.n_diffuse == 0


def test_statespace_with_the_exact_diffuse_likelihood():
    y = local_level()
    fit = statespace(y, build_level, [0.0, 0.0], init="exact")
    assert fit.converged and fit.filter.init == "exact"
    assert fit.n_obs == len(y)
    assert fit.loglik == pytest.approx(fit.filter.loglik, abs=1e-10)
    # the approximation with its first date burnt maximises the same
    # function up to a constant and terms of order 1 / kappa
    approx = statespace(y, build_level, [0.0, 0.0], init="diffuse", burn=1)
    np.testing.assert_allclose(fit.params, approx.params, atol=1e-5)
    np.testing.assert_allclose(fit.se, approx.se, rtol=1e-4)
    assert fit.loglik == pytest.approx(
        approx.loglik - 0.5 * np.log(2 * np.pi), abs=1e-5
    )
    numba = statespace(
        y, build_level, fit.params.to_numpy(), init="exact", engine="numba"
    )
    assert numba.loglik == pytest.approx(fit.loglik, abs=1e-9)
    # diffuse= alone selects the exact filter
    again = statespace(y, build_level, fit.params.to_numpy(), diffuse=[True])
    assert again.loglik == pytest.approx(fit.loglik, abs=1e-9)
