"""Options and degenerate inputs of ``sp.kalman_filter``,
``sp.statespace``, ``sp.tsfilter``, ``sp.xcorr`` and the Kalman version of
``sp.tvp_var``.

A linear Gaussian state space model on a short sample is one multivariate
normal distribution; likelihood and smoothed states are checked against
it directly. The time convention is ``X_t = F X_{t-1} + V_t`` with
``(x0, P0)`` the moments of ``X_0``.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import statspai as sp
from statspai.exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)

F = np.array([[0.7, 0.2], [0.0, 0.9]])
G = np.array([[1.0, 0.0], [0.5, 1.0], [1.0, -1.0]])
Q = np.array([[0.5, 0.1], [0.1, 0.3]])
R = np.diag([0.4, 0.2, 0.6])
A = np.array([0.3, -0.1, 0.0])
X0 = np.array([0.5, -0.5])
P0 = np.array([[1.0, 0.2], [0.2, 2.0]])


def _obs(T: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(X0, P0)
    out = np.empty((T, 3))
    for t in range(T):
        x = F @ x + rng.multivariate_normal(np.zeros(2), Q)
        out[t] = A + G @ x + rng.multivariate_normal(np.zeros(3), R)
    return out


def _joint(T: int):
    """Mean and covariance of the stacked states X_1..X_T and of the
    stacked observations, with their cross covariance."""
    m = 2
    mean_x = np.empty((T, m))
    cov_x = np.zeros((T * m, T * m))
    mu, P = X0.copy(), P0.copy()
    marg = []
    for t in range(T):
        mu, P = F @ mu, F @ P @ F.T + Q
        mean_x[t] = mu
        marg.append(P)
    for t in range(T):
        for s in range(t, T):
            block = marg[t] @ np.linalg.matrix_power(F.T, s - t)
            cov_x[t * m : (t + 1) * m, s * m : (s + 1) * m] = block
            cov_x[s * m : (s + 1) * m, t * m : (t + 1) * m] = block.T
    H = np.kron(np.eye(T), G)
    mean_y = (A + mean_x @ G.T).ravel()
    cov_y = H @ cov_x @ H.T + np.kron(np.eye(T), R)
    return mean_x.ravel(), cov_x, mean_y, cov_y, cov_x @ H.T


def _filter(y, **kw):
    kw = {**dict(F=F, G=G, Q=Q, R=R, A=A, x0=X0, P0=P0), **kw}
    return sp.kalman_filter(y, **kw)


class TestKalmanFilterAgainstTheJointNormal:
    def test_loglik_is_the_density_of_the_stacked_observations(self):
        y = _obs()
        _, _, mean_y, cov_y, _ = _joint(len(y))
        out = _filter(y)
        expected = stats.multivariate_normal(mean_y, cov_y).logpdf(y.ravel())
        assert out.loglik == pytest.approx(expected, abs=1e-9)
        assert out.loglik == pytest.approx(out.loglik_obs.sum())
        assert out.n_obs == 18 and out.n_dates == 6 and out.init == "user"

    def test_missing_elements_drop_out_of_the_density(self):
        y = _obs()
        y[1, 0] = np.nan
        y[3] = np.nan  # a whole date
        y[4, 1:] = np.nan
        keep = np.isfinite(y.ravel())
        _, _, mean_y, cov_y, _ = _joint(len(y))
        expected = stats.multivariate_normal(
            mean_y[keep], cov_y[np.ix_(keep, keep)]
        ).logpdf(y.ravel()[keep])
        out = _filter(y)
        assert out.loglik == pytest.approx(expected, abs=1e-9)
        assert out.n_obs == keep.sum() and out.n_dates == 5
        assert out.loglik_obs[3] == 0.0
        # A date with nothing observed is a pure prediction step.
        np.testing.assert_allclose(out.filtered_state[3], out.predicted_state[3])
        np.testing.assert_allclose(out.filtered_cov[3], out.predicted_cov[3])
        assert np.isnan(out.innovations[1, 0]) and np.isnan(out.innovations[3]).all()

    def test_smoother_is_the_projection_on_all_observations(self):
        y = _obs()
        y[2, 1] = np.nan
        keep = np.isfinite(y.ravel())
        mean_x, cov_x, mean_y, cov_y, cov_xy = _joint(len(y))
        gain = cov_xy[:, keep] @ np.linalg.inv(cov_y[np.ix_(keep, keep)])
        post_mean = mean_x + gain @ (y.ravel()[keep] - mean_y[keep])
        post_cov = cov_x - gain @ cov_xy[:, keep].T
        out = _filter(y)
        np.testing.assert_allclose(out.smoothed_state.ravel(), post_mean, atol=1e-9)
        for t in range(len(y)):
            np.testing.assert_allclose(
                out.smoothed_cov[t],
                post_cov[2 * t : 2 * t + 2, 2 * t : 2 * t + 2],
                atol=1e-9,
            )
        # Nothing later than the last date revises it.
        np.testing.assert_allclose(
            out.smoothed_state[-1], out.filtered_state[-1], atol=1e-12
        )
        np.testing.assert_allclose(
            out.smoothed_cov[-1], out.filtered_cov[-1], atol=1e-12
        )

    def test_filtered_state_is_the_projection_on_the_past(self):
        y = _obs()
        mean_x, cov_x, mean_y, cov_y, cov_xy = _joint(len(y))
        out = _filter(y)
        for t in (0, 2, 5):
            past = slice(0, 3 * (t + 1))
            rows = slice(2 * t, 2 * t + 2)
            gain = cov_xy[rows, past] @ np.linalg.inv(cov_y[past, past])
            np.testing.assert_allclose(
                out.filtered_state[t],
                mean_x[rows] + gain @ (y.ravel()[past] - mean_y[past]),
                atol=1e-9,
            )
        np.testing.assert_allclose(out.predicted_state[0], F @ X0)
        np.testing.assert_allclose(out.predicted_cov[0], F @ P0 @ F.T + Q)

    def test_burn_leaves_the_first_dates_out_of_the_likelihood(self):
        y = _obs()
        full = _filter(y)
        late = _filter(y, burn=2)
        assert late.loglik == pytest.approx(full.loglik_obs[2:].sum())
        assert late.n_obs == 12 and late.n_dates == 4
        np.testing.assert_allclose(late.smoothed_state, full.smoothed_state)

    def test_standardised_innovations(self):
        out = _filter(_obs())
        for t in range(6):
            L = np.linalg.cholesky(out.innovations_cov[t])
            np.testing.assert_allclose(
                out.std_innovations[t],
                np.linalg.solve(L, out.innovations[t]),
                atol=1e-10,
            )
        ll = -0.5 * (
            3 * np.log(2 * np.pi)
            + np.log(np.linalg.det(out.innovations_cov[0]))
            + out.std_innovations[0] @ out.std_innovations[0]
        )
        assert out.loglik_obs[0] == pytest.approx(ll)

    def test_time_varying_system_matrices(self):
        y = _obs()
        stacked = _filter(
            y,
            F=np.repeat(F[None], 6, axis=0),
            G=np.repeat(G[None], 6, axis=0),
            Q=np.repeat(Q[None], 6, axis=0),
            R=np.repeat(R[None], 6, axis=0),
            A=np.repeat(A[None], 6, axis=0),
        )
        const = _filter(y)
        assert stacked.loglik == pytest.approx(const.loglik, abs=1e-12)
        np.testing.assert_allclose(stacked.smoothed_state, const.smoothed_state)

    def test_no_smoother(self):
        out = _filter(_obs(), smooth=False)
        assert out.smoothed_state is None
        with pytest.raises(MethodIncompatibility, match="smoother was not run"):
            out.states("smoothed")
        assert "Kalman filter\n" in out.summary() + "\n"
        assert list(out.states("filtered").columns) == ["x1", "x2", "x1_se", "x2_se"]


class TestKalmanFilterSpecialCases:
    def test_random_walk_regression_is_the_dlm(self):
        rng = np.random.default_rng(3)
        n = 40
        x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "y": 0.5 + 1.5 * x + 0.4 * rng.normal(size=n)})
        X = np.column_stack([np.ones(n), x])
        W = np.array([0.02, 0.05])
        C0 = np.array([[2.0, 0.4], [0.4, 1.5]])
        m0 = np.array([0.3, -0.2])
        ref = sp.dlm("y ~ x", df, obs_var=0.16, state_var=W, m0=m0, C0=C0)
        out = sp.kalman_filter(
            df["y"],
            F=np.eye(2),
            G=X,
            Q=np.diag(W),
            R=0.16,
            x0=m0,
            P0=C0,
            state_names=["Intercept", "x"],
        )
        # Two implementations of one model.
        assert out.loglik == pytest.approx(ref.loglik, abs=1e-9)
        np.testing.assert_allclose(
            out.filtered_state, ref.filtered[["Intercept", "x"]], atol=1e-9
        )
        np.testing.assert_allclose(
            out.states()[["Intercept", "x", "Intercept_se", "x_se"]],
            ref.smoothed[["Intercept", "x", "Intercept_sd", "x_sd"]],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            out.innovations[:, 0], df["y"] - ref.fitted, atol=1e-9
        )

    def test_stationary_start_gives_the_exact_ar1_likelihood(self):
        rng = np.random.default_rng(4)
        phi, s2, n = 0.6, 1.3, 30
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + np.sqrt(s2) * rng.normal()
        out = sp.kalman_filter(y, F=phi, G=1.0, Q=s2, R=0.0)
        assert out.init == "stationary"
        assert out.P0[0, 0] == pytest.approx(s2 / (1 - phi**2))
        lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        cov = s2 / (1 - phi**2) * phi**lags
        assert out.loglik == pytest.approx(
            stats.multivariate_normal(np.zeros(n), cov).logpdf(y), abs=1e-9
        )
        # The state is observed without error.
        np.testing.assert_allclose(out.filtered_state[:, 0], y, atol=1e-12)
        with pytest.raises(MethodIncompatibility, match="unit circle"):
            sp.kalman_filter(y, F=1.0, G=1.0, Q=s2, R=1.0, init="stationary")

    def test_unstable_transition_starts_diffuse(self):
        y = np.cumsum(np.random.default_rng(5).normal(size=30))
        out = sp.kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=0.5)
        assert out.init == "diffuse"
        assert out.P0[0, 0] == 1e7
        other = sp.kalman_filter(y, F=1.0, G=1.0, Q=1.0, R=0.5, kappa=1e5)
        assert other.P0[0, 0] == 1e5
        assert "Approximate diffuse" in out.model_info["notes"][0]

    def test_exact_diffuse_is_the_limit_of_a_large_prior_variance(self):
        y = np.cumsum(np.random.default_rng(5).normal(size=30))
        kw = dict(F=1.0, G=1.0, Q=1.0, R=0.5)
        exact = sp.kalman_filter(y, init="exact", **kw)
        approx = sp.kalman_filter(y, init="diffuse", kappa=1e8, **kw)
        assert exact.n_diffuse == 1 and exact.init == "exact"
        # loglik = lim [loglik(kappa) + 0.5 n_diffuse log(kappa)]; the
        # remainder is of order 1 / kappa.
        assert exact.loglik == pytest.approx(
            approx.loglik + 0.5 * np.log(1e8), abs=1e-6
        )
        np.testing.assert_allclose(
            exact.smoothed_state, approx.smoothed_state, atol=1e-6
        )
        np.testing.assert_allclose(exact.smoothed_cov, approx.smoothed_cov, atol=1e-6)
        marked = sp.kalman_filter(y, diffuse=[True], **kw)
        assert marked.loglik == exact.loglik
        with pytest.raises(MethodIncompatibility, match="diffuse= marks"):
            sp.kalman_filter(y, diffuse=[True], init="stationary", **kw)

    def test_forecast_moments(self):
        y = _obs()
        out = _filter(y)
        fc = out.forecast(3)
        x, P = out.filtered_state[-1], out.filtered_cov[-1]
        for h in range(3):
            x, P = F @ x, F @ P @ F.T + Q
            np.testing.assert_allclose(fc["state"].iloc[h], x)
            np.testing.assert_allclose(fc["state_cov"][h], P)
            np.testing.assert_allclose(fc["obs"].iloc[h], A + G @ x)
            np.testing.assert_allclose(fc["obs_cov"][h], G @ P @ G.T + R)
            np.testing.assert_allclose(
                fc["obs_se"].iloc[h], np.sqrt(np.diag(G @ P @ G.T + R))
            )
        assert fc["obs"].index.tolist() == [1, 2, 3]

    def test_forecast_needs_the_future_of_a_time_varying_matrix(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=20)
        y = 1.5 * x + rng.normal(size=20)
        out = sp.kalman_filter(y, F=1.0, G=x[:, None], Q=0.01, R=1.0, P0=4.0)
        with pytest.raises(MethodIncompatibility, match="G varies over the sample"):
            out.forecast(2)
        fc = out.forecast(2, G=np.array([[0.5], [2.0]]))
        b = out.filtered_state[-1, 0]
        np.testing.assert_allclose(fc["obs"].iloc[:, 0], [0.5 * b, 2.0 * b])
        with pytest.raises(MethodIncompatibility, match="Unknown system matrices"):
            out.forecast(2, H=1.0)
        with pytest.raises(MethodIncompatibility, match="at least 1"):
            out.forecast(0)

    def test_columns_of_a_frame(self):
        y = _obs()
        df = pd.DataFrame(y, columns=["a", "b", "c"], index=pd.RangeIndex(10, 16))
        out = _filter(["a", "b", "c"], data=df)
        ref = _filter(y)
        assert out.loglik == ref.loglik
        assert out.obs_names == ["a", "b", "c"]
        assert out.states().index.tolist() == list(range(10, 16))
        assert _filter(df).loglik == ref.loglik
        d = out.to_dict()
        assert d["n_obs"] == 18 and d["state_names"] == ["x1", "x2"]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(init="zz"), "init='zz'"),
            (dict(burn=6), "burn must be between"),
            (dict(x0=[1.0, 2.0, 3.0]), "x0 must have 2"),
            (dict(state_names=["a"]), "state_names must have 2"),
            (dict(Q=-Q), "not positive definite"),
            (dict(P0=None, init="diffuse", kappa=-1.0), "kappa must be positive"),
            (dict(G=np.ones((3, 3))), "G"),
            (dict(F=None), "missing"),
        ],
    )
    def test_refusals(self, kwargs, match):
        with pytest.raises(MethodIncompatibility, match=match):
            _filter(_obs(), **kwargs)

    def test_observation_refusals(self):
        with pytest.raises(MethodIncompatibility, match="infinite"):
            _filter(np.where(np.eye(6, 3) > 0, np.inf, _obs()))
        with pytest.raises(DataInsufficient, match="empty"):
            _filter(np.empty((0, 3)))
        with pytest.raises(MethodIncompatibility, match="not in data"):
            _filter(["a"], data=pd.DataFrame({"b": [1.0]}))
        with pytest.raises(MethodIncompatibility, match="one- or two-dimensional"):
            _filter(np.zeros((2, 3, 1)))
        with pytest.raises(MethodIncompatibility, match="which='x'"):
            _filter(_obs()).states("x")


class TestStatespaceEstimation:
    def _level(self, n=120, seed=6):
        rng = np.random.default_rng(seed)
        return np.cumsum(0.5 * rng.normal(size=n)) + rng.normal(size=n)

    @staticmethod
    def _build(th):
        return {"F": 1.0, "G": 1.0, "Q": np.exp(th[0]), "R": np.exp(th[1])}

    def test_local_level_estimates_are_a_maximum_and_agree_with_dlm(self):
        y = self._level()
        fit = sp.statespace(
            y,
            self._build,
            [0.0, 0.0],
            init="exact",
            param_names=["log_q", "log_r"],
            transform=lambda th: {"q": np.exp(th[0]), "r": np.exp(th[1])},
        )
        assert fit.converged
        best = fit.params.to_numpy()
        for step in ([0.05, 0], [-0.05, 0], [0, 0.05], [0, -0.05]):
            th = best + np.array(step)
            alt = sp.kalman_filter(y, init="exact", **self._build(th))
            assert alt.loglik < fit.loglik
        assert fit.loglik == pytest.approx(fit.filter.loglik)
        assert fit.aic == pytest.approx(-2 * fit.loglik + 4)
        assert fit.bic == pytest.approx(-2 * fit.loglik + 2 * np.log(fit.n_dates))
        # Delta method for exp(theta).
        np.testing.assert_allclose(
            fit.transformed["se"], np.exp(best) * fit.se.to_numpy(), rtol=1e-5
        )
        np.testing.assert_allclose(fit.transformed["estimate"], np.exp(best))
        # sp.dlm maximises the same likelihood up to the diffuse first
        # date: 1e-3 relative is the agreement of two numerical optimisers
        # on a flat likelihood.
        ref = sp.dlm("y ~ 1", pd.DataFrame({"y": y}))
        np.testing.assert_allclose(
            np.exp(best)[::-1], ref.variances["estimate"], rtol=1e-3
        )
        assert fit.coef is fit.params or fit.coef.equals(fit.params)

    def test_table_is_a_wald_table(self):
        y = self._level()
        fit = sp.statespace(y, self._build, [0.0, 0.0], init="exact", alpha=0.1)
        z = stats.norm.ppf(0.95)
        t = fit.table
        np.testing.assert_allclose(t["se"], np.sqrt(np.diag(fit.cov)))
        np.testing.assert_allclose(t["z"], t["estimate"] / t["se"])
        np.testing.assert_allclose(t["upper"] - t["estimate"], z * t["se"])
        np.testing.assert_allclose(t["pvalue"], 2 * stats.norm.sf(np.abs(t["z"])))

    def test_robust_covariance_is_the_sandwich_of_the_other_two(self):
        y = self._level()
        fits = {
            v: sp.statespace(y, self._build, [0.0, 0.0], init="exact", vce=v)
            for v in ("hessian", "opg", "robust")
        }
        H_inv = fits["hessian"].cov.to_numpy()
        opg = np.linalg.inv(fits["opg"].cov.to_numpy())
        # 1e-4: each piece is a finite-difference approximation at an
        # optimum found separately.
        np.testing.assert_allclose(
            fits["robust"].cov.to_numpy(), H_inv @ opg @ H_inv, rtol=1e-4
        )
        np.testing.assert_allclose(
            fits["robust"].params, fits["hessian"].params, atol=1e-8
        )

    def test_optimisers_reach_the_same_point(self):
        y = self._level()
        a = sp.statespace(y, self._build, [0.0, 0.0], init="exact")
        b = sp.statespace(
            y, self._build, [0.5, -0.5], init="exact", method="nelder-mead"
        )
        assert b.loglik == pytest.approx(a.loglik, abs=1e-5)

    def test_forecast_and_summary_delegate_to_the_filter(self):
        y = self._level()
        fit = sp.statespace(y, self._build, [0.0, 0.0], init="exact")
        fc = fit.forecast(2)
        ref = fit.filter.forecast(2)
        pd.testing.assert_frame_equal(fc["obs"], ref["obs"])
        assert "theta1" in fit.summary()
        d = fit.to_dict()
        assert d["loglik"] == pytest.approx(fit.loglik)

    def test_iteration_limit_is_reported(self):
        y = self._level()
        with pytest.warns(ConvergenceWarning, match="scaled gradient"):
            fit = sp.statespace(
                y, self._build, [3.0, -3.0], init="exact", maxiter=1, method="l-bfgs-b"
            )
        assert fit.converged is False

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(method="newton"), "method='newton'"),
            (dict(vce="oim"), "vce='oim'"),
            (dict(engine="c"), "engine='c'"),
            (dict(param_names=["a"]), "param_names must have 2"),
            (dict(burn=500), "burn must be between"),
        ],
    )
    def test_refusals(self, kwargs, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.statespace(self._level(), self._build, [0.0, 0.0], **kwargs)

    def test_build_and_start_refusals(self):
        y = self._level()
        with pytest.raises(MethodIncompatibility, match="non-empty finite"):
            sp.statespace(y, self._build, [])
        with pytest.raises(MethodIncompatibility, match="non-empty finite"):
            sp.statespace(y, self._build, [np.nan, 0.0])
        with pytest.raises(MethodIncompatibility, match="must return a dict"):
            sp.statespace(y, lambda th: 1.0, [0.0])
        with pytest.raises(MethodIncompatibility, match="unknown keys"):
            sp.statespace(y, lambda th: {**self._build(th), "Z": 1}, [0.0, 0.0])
        with pytest.raises(DataInsufficient, match="cannot identify"):
            sp.statespace(y[:2], self._build, [0.0, 0.0])
        with pytest.raises(MethodIncompatibility, match="not finite at the starting"):
            sp.statespace(
                y,
                lambda th: {"F": 1.0, "G": 1.0, "Q": -1.0, "R": th[0]},
                [-1.0],
                init="diffuse",
            )

    def test_alpha_out_of_range_is_refused(self):
        for bad in (3, 0.0, 1.0, np.nan):
            with pytest.raises(MethodIncompatibility, match="alpha"):
                sp.statespace(self._level(), self._build, [0.0, 0.0], alpha=bad)


# ----------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------
def _gdp(n: int = 120, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        0.5 * t + 3 * np.sin(2 * np.pi * t / 20) + np.cumsum(0.3 * rng.normal(size=n))
    )


class TestFilters:
    @pytest.mark.parametrize("lam", [6.25, 1600.0])
    def test_hp_trend_solves_the_penalised_least_squares_problem(self, lam):
        y = _gdp()
        n = len(y)
        res = sp.tsfilter(y, smooth=lam)
        D = np.diff(np.eye(n), 2, axis=0)
        # First-order condition of sum (y - tau)^2 + lam sum (D tau)^2.
        np.testing.assert_allclose(
            (np.eye(n) + lam * D.T @ D) @ res.trend.to_numpy(), y, atol=1e-7
        )
        np.testing.assert_allclose(res.trend + res.cycle, y, atol=1e-10)
        frame = res.to_frame()
        assert frame.shape[0] == n and res.params["smooth"] == lam

    def test_hp_named_penalties_and_gain(self):
        y = _gdp()
        for name, lam in (
            ("annual", 6.25),
            ("quarterly", 1600.0),
            ("monthly", 129600.0),
        ):
            a = sp.tsfilter(y, smooth=name)
            b = sp.tsfilter(y, smooth=lam)
            np.testing.assert_allclose(a.cycle, b.cycle)
        omega = np.array([0.1, 0.5, np.pi])
        gain = sp.tsfilter(y, smooth=1600.0).gain(omega)
        k = 4 * 1600.0 * (1 - np.cos(omega)) ** 2
        np.testing.assert_allclose(gain, k / (1 + k), rtol=1e-10)

    def test_hp_removes_a_line_exactly(self):
        res = sp.tsfilter(3.0 + 0.7 * np.arange(60.0))
        # A straight line has zero second differences.
        np.testing.assert_allclose(res.cycle, 0.0, atol=1e-8)

    def test_one_sided_hp_is_the_end_point_of_the_filter_on_the_past(self):
        y = _gdp(60)
        one = sp.tsfilter(y, one_sided=True)
        for t in (5, 20, 59):
            two = sp.tsfilter(y[: t + 1])
            assert one.trend.iloc[t] == pytest.approx(two.trend.iloc[-1], abs=1e-7)
        np.testing.assert_allclose(one.cycle.iloc[:2], 0.0, atol=1e-10)
        with pytest.raises(MethodIncompatibility, match="one_sided=True is defined"):
            sp.tsfilter(y, method="bk", one_sided=True)

    def test_baxter_king_equals_statsmodels(self):
        from statsmodels.tsa.filters.bk_filter import bkfilter

        y = _gdp()
        res = sp.tsfilter(y, method="bk", low=6, high=32, K=12)
        ref = bkfilter(y, low=6, high=32, K=12)
        np.testing.assert_allclose(res.cycle.to_numpy()[12:-12], ref, atol=1e-10)
        assert res.cycle.isna().sum() == 24
        b = res.params["weights"]
        # The symmetric weights sum to zero, which removes a unit root.
        assert b[0] + 2 * b[1:].sum() == pytest.approx(0.0, abs=1e-12)
        raw = sp.tsfilter(y, method="bk", stationary=True).params["weights"]
        assert abs(raw[0] + 2 * raw[1:].sum()) > 1e-3
        np.testing.assert_allclose(raw - raw[-1], b - b[-1], atol=1e-12)

    @pytest.mark.parametrize("drift", [True, False])
    def test_christiano_fitzgerald_equals_statsmodels(self, drift):
        from statsmodels.tsa.filters.cf_filter import cffilter

        y = _gdp()
        res = sp.tsfilter(y, method="cf", low=6, high=32, drift=drift)
        cycle, _ = cffilter(y, low=6, high=32, drift=drift)
        np.testing.assert_allclose(res.cycle, cycle, atol=1e-9)
        assert res.cycle.notna().all()

    def test_fixed_length_christiano_fitzgerald_is_a_moving_average(self):
        y = _gdp()
        res = sp.tsfilter(y, method="cf", sma_order=8, drift=False)
        b = res.params["weights"]
        assert res.cycle.isna().sum() == 16
        t = 40
        window = y[t - 8 : t + 9]
        assert res.cycle.iloc[t] == pytest.approx(
            b[0] * window[8]
            + sum(b[j] * (window[8 - j] + window[8 + j]) for j in range(1, 9))
        )
        with pytest.raises(MethodIncompatibility, match="sma_order= belongs"):
            sp.tsfilter(y, method="bk", sma_order=8)

    @pytest.mark.parametrize("h, p", [(8, 4), (4, 2)])
    def test_hamilton_cycle_is_a_regression_residual(self, h, p):
        y = _gdp()
        n = len(y)
        start = h + p - 1
        X = np.column_stack([y[start - h - j : n - h - j] for j in range(p)])
        ols = sm.OLS(y[start:], sm.add_constant(X)).fit()
        res = sp.tsfilter(y, method="hamilton", h=h, p=p)
        np.testing.assert_allclose(res.cycle.to_numpy()[start:], ols.resid, atol=1e-8)
        assert res.cycle.isna().sum() == start

    def test_butterworth_splits_the_series(self):
        y = _gdp()
        res = sp.tsfilter(y, method="bw", high=32, order=2)
        np.testing.assert_allclose(res.trend + res.cycle, y, atol=1e-9)
        assert res.params["smooth"] == pytest.approx(np.tan(np.pi / 32) ** -4)
        # A high-pass filter removes a constant and a line.
        flat = sp.tsfilter(5.0 + 0.3 * np.arange(60.0), method="bw")
        np.testing.assert_allclose(flat.cycle, 0.0, atol=1e-8)

    def test_gain_of_a_symmetric_moving_average_and_reporting(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y = _gdp()
        res = sp.tsfilter(y, method="bk", K=12)
        b = res.params["weights"]
        omega = np.array([0.0, 0.3, 1.0, np.pi])
        expected = np.abs(
            b[0] + 2 * (b[1:, None] * np.cos(np.outer(np.arange(1, 13), omega))).sum(0)
        )
        np.testing.assert_allclose(res.gain(omega), expected, atol=1e-12)
        assert res.gain([0.0]).iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert "bk" in res.summary().lower() or "Baxter" in res.summary()
        out = res.plot()
        try:
            assert out is not None
        finally:
            plt.close("all")
        frame = res.to_frame()
        np.testing.assert_allclose(frame.iloc[:, 0], y)

    def test_missing_ends_come_back_as_missing(self):
        y = _gdp(60)
        padded = np.r_[np.nan, y, np.nan, np.nan]
        res = sp.tsfilter(padded)
        ref = sp.tsfilter(y)
        assert res.cycle.isna().sum() == 3
        np.testing.assert_allclose(res.cycle.to_numpy()[1:-2], ref.cycle)
        df = pd.DataFrame(
            {"gdp": y}, index=pd.period_range("2000Q1", periods=60, freq="Q")
        )
        named = sp.tsfilter(df, "gdp")
        assert named.cycle.index.equals(df.index)

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            (dict(method="HP"), MethodIncompatibility, "method='HP'"),
            (dict(smooth="weekly"), MethodIncompatibility, "weekly"),
            (dict(smooth=-1.0), MethodIncompatibility, "not a positive number"),
            (dict(method="bk", low=1), MethodIncompatibility, "not a band"),
            (dict(method="cf", low=40), MethodIncompatibility, "not a band"),
            (dict(method="bk", K=0), MethodIncompatibility, "K=0"),
            (dict(method="bk", K=70), DataInsufficient, "at least 141"),
            (dict(method="bw", order=0), MethodIncompatibility, "order=0"),
            (dict(method="hamilton", h=0), MethodIncompatibility, "positive integers"),
            (dict(method="hamilton", h=100, p=30), DataInsufficient, "at least"),
            (dict(method="cf", sma_order=60), DataInsufficient, "at least 122"),
            (dict(method="cf", sma_order=0), MethodIncompatibility, "sma_order=0"),
        ],
    )
    def test_refusals(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            sp.tsfilter(_gdp(), **kwargs)

    def test_data_refusals(self):
        y = _gdp()
        with pytest.raises(MethodIncompatibility, match="missing or infinite"):
            sp.tsfilter(np.r_[y[:10], np.nan, y[11:]])
        with pytest.raises(DataInsufficient, match="no observed value"):
            sp.tsfilter(np.full(10, np.nan))
        with pytest.raises(DataInsufficient, match="at least 4"):
            sp.tsfilter(y[:3])
        with pytest.raises(DataInsufficient, match="collinear"):
            sp.tsfilter(np.arange(60.0), method="hamilton")
        with pytest.raises(MethodIncompatibility, match="not a column"):
            sp.tsfilter(pd.DataFrame({"a": y}))


# ----------------------------------------------------------------------
# cross-correlations
# ----------------------------------------------------------------------
class TestCrossCorrelogram:
    def _pair(self, n=100, seed=8):
        rng = np.random.default_rng(seed)
        y = rng.normal(size=n + 3)
        x = np.r_[np.zeros(3), y[:-3]][3:] * 0.8 + 0.6 * rng.normal(size=n)
        return x, y[3:]

    def test_lag_h_is_the_correlation_of_x_ahead_with_y(self):
        x, y = self._pair()
        n = len(x)
        out = sp.xcorr(x, y, lags=5)
        dx, dy = x - x.mean(), y - y.mean()
        den = n * x.std() * y.std()
        for h in range(-5, 6):
            if h >= 0:
                num = dx[h:] @ dy[: n - h]
            else:
                num = dx[: n + h] @ dy[-h:]
            assert out.table.loc[h, "xcorr"] == pytest.approx(num / den, abs=1e-12)
        # x was built as y three periods earlier: y leads x by three.
        assert out.table["xcorr"].idxmax() == 3
        assert bool(out.table.loc[3, "outside"]) is True

    def test_swapping_the_series_mirrors_the_table(self):
        x, y = self._pair()
        a = sp.xcorr(x, y, lags=6).table["xcorr"]
        b = sp.xcorr(y, x, lags=6).table["xcorr"]
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy()[::-1], atol=1e-14)

    def test_band_and_default_lags(self):
        x, y = self._pair()
        out = sp.xcorr(x, y, alpha=0.1)
        assert out.table.index.tolist() == list(range(-20, 21))
        np.testing.assert_allclose(out.table["upper"], stats.norm.ppf(0.95) / 10.0)
        np.testing.assert_allclose(out.table["lower"], -out.table["upper"])
        short = sp.xcorr(x[:30], y[:30])
        assert short.table.index.max() == 30 // 2 - 2
        assert out.haugh == {}

    def test_prewhitening_by_separate_autoregressions(self):
        x, y = self._pair()
        out = sp.xcorr(x, y, prewhiten="ar", ar_order=2, lags=4)

        def resid(v):
            X = sm.add_constant(np.column_stack([v[1:-1], v[:-2]]))
            return sm.OLS(v[2:], X).fit().resid

        rx, ry = resid(x), resid(y)
        np.testing.assert_allclose(out.residuals.iloc[:, 0], rx, atol=1e-9)
        np.testing.assert_allclose(out.residuals.iloc[:, 1], ry, atol=1e-9)
        ref = sp.xcorr(rx, ry, lags=4)
        np.testing.assert_allclose(out.table["xcorr"], ref.table["xcorr"], atol=1e-10)
        n = len(rx)
        r = out.table["xcorr"].to_numpy()
        h = out.table.index.to_numpy()
        assert out.n_obs == n == 98
        assert out.haugh["statistic"] == pytest.approx(n * np.sum(r**2))
        assert out.haugh["statistic_adj"] == pytest.approx(
            n**2 * np.sum(r**2 / (n - np.abs(h)))
        )
        assert out.haugh["df"] == 9
        assert out.haugh["pvalue"] == pytest.approx(
            stats.chi2.sf(out.haugh["statistic"], 9)
        )
        assert out.ar_orders == {"x": 2, "y": 2}

    def test_one_filter_for_both_series(self):
        x, y = self._pair()
        out = sp.xcorr(x, y, prewhiten="x", ar_order=1, lags=3)
        X = sm.add_constant(x[:-1])
        fit = sm.OLS(x[1:], X).fit()
        fx = fit.resid
        fy = y[1:] - fit.params[1] * y[:-1]
        ref = sp.xcorr(fx, fy, lags=3)
        # A constant shift of the filtered y does not change a correlation.
        np.testing.assert_allclose(out.table["xcorr"], ref.table["xcorr"], atol=1e-10)
        assert out.haugh == {}

    def test_yule_walker_prewhitening_equals_statsmodels(self):
        from statsmodels.regression.linear_model import yule_walker

        x, y = self._pair()
        out = sp.xcorr(x, y, prewhiten="ar", ar_order=2, ar_method="yw", lags=3)
        for key, v in (("x", x), ("y", y)):
            # Yule-Walker equations on autocovariances with divisor n.
            rho, _ = yule_walker(v, 2, method="mle", demean=True)
            np.testing.assert_allclose(out.ar_coefs[key], rho, atol=1e-10)
        d = x - x.mean()
        a = out.ar_coefs["x"]
        e = d[2:] - a[0] * d[1:-1] - a[1] * d[:-2]
        # The stored residuals are these less their mean, which a
        # correlation does not see.
        np.testing.assert_allclose(out.residuals.iloc[:, 0], e - e.mean(), atol=1e-10)

    def test_order_chosen_by_criterion_and_reporting(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x, y = self._pair()
        out = sp.xcorr(x, y, prewhiten="ar", max_order=4, ic="bic", lags=3)
        for key in ("x", "y"):
            assert 0 <= out.ar_orders[key] <= 4
        # The chosen orders give the same table when passed explicitly.
        fixed = sp.xcorr(
            x,
            y,
            prewhiten="ar",
            ar_order=(out.ar_orders["x"], out.ar_orders["y"]),
            lags=3,
        )
        np.testing.assert_allclose(out.table["xcorr"], fixed.table["xcorr"], atol=1e-10)
        text = out.summary()
        assert "h > 0: y leads x" in text and "Haugh" in text
        ax = out.plot()
        try:
            fig = ax.figure if hasattr(ax, "figure") else ax
            assert fig is not None
        finally:
            plt.close("all")

    def test_columns_of_a_frame_and_missing_ends(self):
        x, y = self._pair()
        df = pd.DataFrame({"x": np.r_[np.nan, x], "y": np.r_[y, np.nan]})
        out = sp.xcorr("x", "y", data=df, lags=2)
        ref = sp.xcorr(x[:-1], y[1:], lags=2)
        np.testing.assert_allclose(out.table["xcorr"], ref.table["xcorr"])
        assert out.n_obs == len(x) - 1
        assert out.names == ("x", "y")

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            (dict(lags=200), DataInsufficient, "lags=200"),
            (dict(prewhiten="z"), MethodIncompatibility, "prewhiten='z'"),
            (dict(prewhiten="ar", ar_method="z"), MethodIncompatibility, "ar_method"),
            (dict(alpha=2), MethodIncompatibility, "alpha=2"),
            (dict(prewhiten="ar", ic="hq"), MethodIncompatibility, "ic"),
        ],
    )
    def test_refusals(self, kwargs, exc, match):
        x, y = self._pair()
        with pytest.raises(exc, match=match):
            sp.xcorr(x, y, **kwargs)

    def test_data_refusals(self):
        x, y = self._pair()
        with pytest.raises(DataInsufficient, match="constant"):
            sp.xcorr(np.ones(100), y)
        with pytest.raises(MethodIncompatibility, match="50 and 100"):
            sp.xcorr(x[:50], y)
        with pytest.raises(MethodIncompatibility, match="missing values between"):
            sp.xcorr(np.r_[x[:5], np.nan, x[6:]], y)


# ----------------------------------------------------------------------
# time-varying-parameter VAR, Kalman version
# ----------------------------------------------------------------------
class TestTVPVARKalman:
    def _data(self, n=60, seed=9):
        rng = np.random.default_rng(seed)
        y = np.zeros((n, 2))
        for t in range(1, n):
            y[t, 0] = 0.5 * y[t - 1, 0] + rng.normal()
            y[t, 1] = 0.3 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + rng.normal()
        return pd.DataFrame(y, columns=["a", "b"])

    def test_each_equation_is_a_dlm(self):
        df = self._data()
        fit = sp.tvp_var(
            df, ["a", "b"], obs_var=[1.0, 1.2], state_var=[0.01, 0.02, 0.0], C0=10.0
        )
        lagged = pd.DataFrame(
            {
                "la": df["a"].shift(1),
                "lb": df["b"].shift(1),
                "a": df["a"],
                "b": df["b"],
            }
        ).dropna()
        for i, (name, v) in enumerate((("a", 1.0), ("b", 1.2))):
            ref = sp.dlm(
                f"{name} ~ la + lb",
                lagged,
                obs_var=v,
                state_var=[0.0, 0.01, 0.02],  # dlm puts the intercept first
                C0=10.0,
            )
            cols = ["la", "lb", "Intercept"]
            np.testing.assert_allclose(
                fit.coef_smoothed[:, i, :], ref.smoothed[cols], atol=1e-8
            )
            np.testing.assert_allclose(
                fit.coef_filtered[:, i, :], ref.filtered[cols], atol=1e-8
            )
            np.testing.assert_allclose(
                fit.se_smoothed[:, i, :],
                ref.smoothed[[c + "_sd" for c in cols]],
                atol=1e-8,
            )
            assert fit.model_info["equation_loglik"][name] == pytest.approx(
                ref.loglik, abs=1e-8
            )
        assert fit.terms == ["L1.a", "L1.b", "_cons"]
        assert fit.n_obs == 59
        # The intercept was given zero innovation variance: it is constant.
        assert np.ptp(fit.coef_smoothed[:, 0, 2]) < 1e-8

    def test_forgetting_without_discounting_is_least_squares(self):
        df = self._data()
        fit = sp.tvp_var(df, ["a", "b"], method="forgetting", lam=1.0, kappa=1.0)
        X = sm.add_constant(df.shift(1).dropna().to_numpy(), prepend=False)
        for i, name in enumerate(("a", "b")):
            ols = sm.OLS(df[name].to_numpy()[1:], X).fit()
            # 1e-5: the start is a prior variance of 1e7, not infinity.
            np.testing.assert_allclose(
                fit.coef_filtered[-1, i], ols.params, rtol=1e-5, atol=1e-6
            )
        assert fit.coef_smoothed is None

    def test_time_sorts_and_labels_the_dates(self):
        df = self._data().assign(year=lambda d: 1950 + np.arange(len(d)))
        kw = dict(obs_var=[1.0, 1.2], state_var=0.01, C0=10.0)
        a = sp.tvp_var(df, ["a", "b"], time="year", **kw)
        b = sp.tvp_var(
            df.sample(frac=1.0, random_state=0), ["a", "b"], time="year", **kw
        )
        np.testing.assert_allclose(b.coef_smoothed, a.coef_smoothed)
        assert list(a.index[:2]) == [1951, 1952]
        # Without variables=, every numeric column but the time column.
        c = sp.tvp_var(df, time="year", **kw)
        assert c.var_names == ["a", "b"]

    def test_coefficient_table(self):
        df = self._data()
        fit = sp.tvp_var(df, ["a", "b"], obs_var=[1.0, 1.2], state_var=0.01, C0=10.0)
        tab = fit.coefficients(alpha=0.2)
        assert len(tab) == 59 * 2 * 3
        z = stats.norm.ppf(0.9)
        np.testing.assert_allclose(tab["upper"] - tab["coef"], z * tab["se"])
        last = tab[(tab["date"] == tab["date"].max()) & (tab["equation"] == "b")]
        np.testing.assert_allclose(last["coef"], fit.coef_smoothed[-1, 1])
        filt = fit.coefficients("filtered")
        first = filt[(filt["date"] == filt["date"].min()) & (filt["equation"] == "a")]
        np.testing.assert_allclose(first["coef"], fit.coef_filtered[0, 0])
        np.testing.assert_allclose(first["se"], fit.se_filtered[0, 0])

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            (dict(method="zz"), MethodIncompatibility, "method='zz'"),
            (dict(lags=0), MethodIncompatibility, "positive integer"),
            (dict(prior="zz"), MethodIncompatibility, "prior='zz'"),
            (dict(alpha=1.5), MethodIncompatibility, "alpha=1.5"),
            (dict(method="forgetting", lam=1.5), MethodIncompatibility, "lam=1.5"),
            (
                dict(method="forgetting", sigma_update="zz"),
                MethodIncompatibility,
                "sigma_update",
            ),
            (dict(method="forgetting", common=True), MethodIncompatibility, "kalman"),
        ],
    )
    def test_refusals(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            sp.tvp_var(self._data(), ["a", "b"], **kwargs)

    def test_data_refusals(self):
        df = self._data()
        with pytest.raises(MethodIncompatibility, match="missing or infinite"):
            sp.tvp_var(df.assign(a=df["a"].where(df.index != 5)), ["a", "b"])
        with pytest.raises(DataInsufficient, match="at least 6"):
            sp.tvp_var(df.iloc[:4], ["a", "b"])
        with pytest.raises(DataInsufficient, match=r"\['a'\] are constant"):
            sp.tvp_var(df.assign(a=2.0), ["a", "b"])
        year = np.arange(len(df), dtype=float)
        year[3] = np.nan
        with pytest.raises(MethodIncompatibility, match="missing value in the time"):
            sp.tvp_var(df.assign(year=year), ["a", "b"], time="year")


class TestNumpyScalarAlpha:
    def test_alpha_may_be_a_numpy_scalar(self):
        rng = np.random.default_rng(6)
        y = np.cumsum(rng.normal(size=120)) + rng.normal(size=120)

        def build(th):
            return {"F": 1.0, "G": 1.0, "Q": np.exp(th[0]), "R": np.exp(th[1])}

        ref = sp.statespace(y, build, [0.0, 0.0], alpha=0.25)
        for alpha in (np.float32(0.25), np.float64(0.25)):
            got = sp.statespace(y, build, [0.0, 0.0], alpha=alpha)
            # a float32 alpha gives a single-precision critical value
            np.testing.assert_allclose(
                got.table.to_numpy(), ref.table.to_numpy(), rtol=1e-6
            )
        with pytest.raises(MethodIncompatibility, match="alpha"):
            sp.statespace(y, build, [0.0, 0.0], alpha=True)
