"""Options and degenerate inputs of ``sp.dlm``.

The random-walk coefficient regression is a joint Gaussian model, so on a
short series every output of the filter and smoother has a closed form
that needs no recursion: the stacked observations are multivariate normal
and the states are a linear projection on them. The tests below compare
against that, against a Kalman filter written out here, and against
ordinary least squares in the limit of zero state variance.
"""

import importlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import (  # noqa: E402
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)

# ``statspai.timeseries.dlm`` the attribute is the function; the module of
# the same name is reached through the import system.
dlm_module = importlib.import_module("statspai.timeseries.dlm")

V_TRUE = 0.16
W_TRUE = np.array([0.02, 0.05])
M0 = np.array([0.3, -0.2])
C0 = np.array([[2.0, 0.4], [0.4, 1.5]])


def _frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    slope = 1.5 + np.cumsum(0.2 * rng.normal(size=n))
    y = 0.5 + slope * x + 0.4 * rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "t": np.arange(n)})


def _design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df["x"].to_numpy()])


def _joint_moments(X, V, W, m0, C0_):
    """Mean and covariance of the stacked observations, and the covariance
    of every state with them, from Cov(beta_t, beta_s) = C0 + min(t, s) W."""
    n, k = X.shape
    Wm = np.diag(W)
    cov_y = np.empty((n, n))
    cov_by = np.empty((n, k, n))
    for t in range(n):
        for s in range(n):
            c = C0_ + (min(t, s) + 1) * Wm
            cov_y[t, s] = X[t] @ c @ X[s] + (V if t == s else 0.0)
            cov_by[t, :, s] = c @ X[s]
    return X @ m0, cov_y, cov_by


def _kalman(y, X, V, W, m0, C0_):
    n, k = X.shape
    m, C = m0.copy(), C0_.copy()
    ms, Cs, fs, Qs, ll = [], [], [], [], 0.0
    for t in range(n):
        R = C + np.diag(W)
        f = X[t] @ m
        Q = X[t] @ R @ X[t] + V
        K = R @ X[t] / Q
        m = m + K * (y[t] - f)
        C = R - np.outer(K, K) * Q
        ll += stats.norm.logpdf(y[t], f, np.sqrt(Q))
        ms.append(m)
        Cs.append(C)
        fs.append(f)
        Qs.append(Q)
    return np.array(ms), np.array(Cs), np.array(fs), np.array(Qs), ll


def _fixed(df, **kw):
    kw.setdefault("obs_var", V_TRUE)
    kw.setdefault("state_var", W_TRUE)
    kw.setdefault("m0", M0)
    kw.setdefault("C0", C0)
    return sp.dlm("y ~ x", df, **kw)


class TestFilterAndSmootherAgainstClosedForms:
    def test_filter_equals_a_kalman_recursion_written_out(self):
        df = _frame()
        fit = _fixed(df)
        m, C, f, Q, ll = _kalman(
            df["y"].to_numpy(), _design(df), V_TRUE, W_TRUE, M0, C0
        )
        # Same recursion in exact arithmetic; the package uses the Joseph
        # form of the update, so only rounding separates them.
        np.testing.assert_allclose(fit.filtered[["Intercept", "x"]], m, atol=1e-10)
        np.testing.assert_allclose(
            fit.filtered[["Intercept_sd", "x_sd"]],
            np.sqrt(np.einsum("tii->ti", C)),
            atol=1e-10,
        )
        np.testing.assert_allclose(fit.fitted, f, atol=1e-10)
        np.testing.assert_allclose(fit.model_info["forecast_variance"], Q, atol=1e-10)
        assert fit.loglik == pytest.approx(ll, abs=1e-9)

    def test_loglik_is_the_joint_normal_density_of_the_observations(self):
        df = _frame(8)
        mean, cov_y, _ = _joint_moments(_design(df), V_TRUE, W_TRUE, M0, C0)
        expected = stats.multivariate_normal(mean, cov_y).logpdf(df["y"].to_numpy())
        # Prediction-error decomposition of the same density.
        assert _fixed(df).loglik == pytest.approx(expected, abs=1e-9)

    def test_smoother_is_the_projection_of_the_states_on_all_the_data(self):
        df = _frame(8)
        X = _design(df)
        mean, cov_y, cov_by = _joint_moments(X, V_TRUE, W_TRUE, M0, C0)
        fit = _fixed(df)
        resid = df["y"].to_numpy() - mean
        for t in range(len(df)):
            gain = cov_by[t] @ np.linalg.inv(cov_y)
            post_mean = M0 + gain @ resid
            post_cov = C0 + (t + 1) * np.diag(W_TRUE) - gain @ cov_by[t].T
            np.testing.assert_allclose(
                fit.smoothed[["Intercept", "x"]].iloc[t], post_mean, atol=1e-9
            )
            np.testing.assert_allclose(
                fit.smoothed[["Intercept_sd", "x_sd"]].iloc[t],
                np.sqrt(np.diag(post_cov)),
                atol=1e-9,
            )

    def test_filtered_equals_smoothed_at_the_last_date(self):
        fit = _fixed(_frame())
        # Nothing later than the last date exists to revise it.
        np.testing.assert_allclose(
            fit.filtered.iloc[-1].to_numpy(), fit.smoothed.iloc[-1].to_numpy(), atol=0
        )
        np.testing.assert_allclose(
            fit.params.to_numpy(), fit.smoothed[["Intercept", "x"]].iloc[-1].to_numpy()
        )
        assert fit.coef is fit.params

    def test_prior_can_be_a_scalar_a_vector_or_a_matrix(self):
        df = _frame()
        a = _fixed(df, m0=0.5, C0=3.0)
        b = _fixed(df, m0=[0.5, 0.5], C0=[3.0, 3.0])
        c = _fixed(df, m0=np.full(2, 0.5), C0=3.0 * np.eye(2))
        for other in (b, c):
            assert other.loglik == a.loglik
            pd.testing.assert_frame_equal(other.smoothed, a.smoothed)

    def test_state_var_scalar_is_one_value_per_coefficient(self):
        df = _frame()
        a = _fixed(df, state_var=0.03)
        b = _fixed(df, state_var=[0.03, 0.03])
        pd.testing.assert_frame_equal(a.smoothed, b.smoothed)
        assert a.variances["estimate"].tolist() == [V_TRUE, 0.03, 0.03]

    def test_constant_zeroes_the_state_variance_of_that_term(self):
        df = _frame()
        a = _fixed(df, constant=["x"])
        b = _fixed(df, state_var=[W_TRUE[0], 0.0])
        pd.testing.assert_frame_equal(a.smoothed, b.smoothed)
        assert a.variances.loc["state:x", "estimate"] == 0.0
        assert a.model_info["constant_terms"] == ["x"]
        # A coefficient that cannot move has one smoothed value.
        assert np.ptp(a.smoothed["x"].to_numpy()) < 1e-10

    def test_time_sorts_and_labels_the_rows(self):
        df = _frame()
        shuffled = df.sample(frac=1.0, random_state=3).assign(
            date=lambda d: 2000 + d["t"]
        )
        a = _fixed(df)
        b = _fixed(shuffled, time="date")
        np.testing.assert_allclose(b.smoothed.to_numpy(), a.smoothed.to_numpy())
        assert b.loglik == pytest.approx(a.loglik, abs=1e-12)
        assert b.smoothed.index.tolist() == list(range(2000, 2040))
        assert b.fitted.index.tolist() == list(range(2000, 2040))

    @pytest.mark.parametrize("level", [0.5, 0.8, 0.99])
    def test_level_sets_the_width_of_every_band(self, level):
        fit = _fixed(_frame(), level=level)
        z = stats.norm.ppf(0.5 + level / 2.0)
        for frame in (fit.smoothed, fit.filtered):
            for nm in ("Intercept", "x"):
                np.testing.assert_allclose(
                    frame[f"{nm}_upper"] - frame[nm], z * frame[f"{nm}_sd"], atol=1e-12
                )
                np.testing.assert_allclose(
                    frame[nm] - frame[f"{nm}_lower"], z * frame[f"{nm}_sd"], atol=1e-12
                )

    def test_python_recursions_equal_the_compiled_ones(self):
        df = _frame(25)
        y, X = df["y"].to_numpy().copy(), _design(df)
        kern = dlm_module._kernels()
        py = dlm_module._filter_py(y, X, V_TRUE, W_TRUE, M0, C0)
        nb = kern["filter"](y, X, V_TRUE, W_TRUE, M0, C0)
        for a, b in zip(py, nb):
            np.testing.assert_allclose(a, b, atol=1e-12)
        m, C, R = py[0], py[1], py[2]
        for a, b in zip(dlm_module._smooth_py(m, C, R), kern["smooth"](m, C, R)):
            np.testing.assert_allclose(a, b, atol=1e-12)
        z = np.random.default_rng(5).standard_normal((len(y) + 1, 2))
        np.testing.assert_allclose(
            dlm_module._backward_sample_py(m, C, R, z),
            kern["backward"](m, C, R, z),
            atol=1e-12,
        )

    def test_python_filter_skips_a_missing_observation(self):
        # sp.dlm refuses missing rows, but the recursion documents that a
        # NaN is a pure prediction step: the likelihood is then the joint
        # normal density of the observations that are there.
        df = _frame(8)
        X = _design(df)
        y = df["y"].to_numpy().copy()
        y[3] = np.nan
        keep = ~np.isnan(y)
        mean, cov_y, _ = _joint_moments(X, V_TRUE, W_TRUE, M0, C0)
        expected = stats.multivariate_normal(
            mean[keep], cov_y[np.ix_(keep, keep)]
        ).logpdf(y[keep])
        out = dlm_module._filter_py(y, X, V_TRUE, W_TRUE, M0, C0)
        assert out[5] == pytest.approx(expected, abs=1e-9)


class TestBackwardSampling:
    def _filter(self, n=6):
        df = _frame(n)
        y, X = df["y"].to_numpy().copy(), _design(df)
        m, C, R, *_ = dlm_module._filter_py(y, X, V_TRUE, W_TRUE, M0, C0)
        return df, y, X, m, C, R

    def test_zero_shocks_give_the_smoothed_means(self):
        _, _, _, m, C, R = self._filter()
        s, _ = dlm_module._smooth_py(m, C, R)
        draw = dlm_module._backward_sample_py(m, C, R, np.zeros_like(m))
        # The draw is the smoothed mean plus a linear map of the shocks.
        np.testing.assert_allclose(draw, s, atol=1e-10)

    def test_draws_have_the_joint_posterior_covariance_of_the_path(self):
        # The sampler is linear in its normal shocks, path = mean + A z,
        # so A A' is the covariance it samples from. It must equal the
        # covariance of the whole state path given the data, computed
        # here from the joint normal distribution without any recursion.
        df, y, X, m, C, R = self._filter()
        n, k = X.shape
        base = dlm_module._backward_sample_py(m, C, R, np.zeros((n + 1, k)))
        A = np.empty(((n + 1) * k, (n + 1) * k))
        for i in range((n + 1) * k):
            z = np.zeros((n + 1) * k)
            z[i] = 1.0
            A[:, i] = (
                dlm_module._backward_sample_py(m, C, R, z.reshape(n + 1, k)) - base
            ).ravel()
        Wm = np.diag(W_TRUE)
        prior = np.empty(((n + 1) * k, (n + 1) * k))
        for t in range(n + 1):
            for s in range(n + 1):
                prior[t * k : (t + 1) * k, s * k : (s + 1) * k] = C0 + min(t, s) * Wm
        H = np.zeros((n, (n + 1) * k))
        for t in range(n):
            H[t, (t + 1) * k : (t + 2) * k] = X[t]
        S = H @ prior @ H.T + V_TRUE * np.eye(n)
        post = prior - prior @ H.T @ np.linalg.solve(S, H @ prior)
        # 1e-7: the sampler adds a 1e-14 ridge before its first Cholesky
        # factor and takes eigenvalue square roots of near-singular blocks.
        np.testing.assert_allclose(A @ A.T, post, atol=1e-7)


class TestMaximumLikelihood:
    def test_constant_coefficients_under_a_diffuse_prior_are_least_squares(self):
        import statsmodels.api as sm

        df = _frame(60, seed=4)
        ols = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
        fit = sp.dlm("y ~ x", df, constant=["Intercept", "x"])
        # 1e-5 relative: the prior variance is 1e7, not infinite, and the
        # observation variance comes out of a numerical optimiser.
        np.testing.assert_allclose(fit.params, ols.params, rtol=1e-5)
        np.testing.assert_allclose(
            fit.smoothed[["Intercept_sd", "x_sd"]].iloc[-1], ols.bse, rtol=1e-5
        )
        # With the coefficients integrated out under a flat prior the
        # likelihood of the variance is the restricted one: RSS / (n - k).
        assert fit.variances.loc["obs", "estimate"] == pytest.approx(
            ols.ssr / (len(df) - 2), rel=1e-5
        )
        assert np.ptp(fit.smoothed["x"].to_numpy()) < 1e-6

    def test_estimates_beat_every_nearby_point_of_the_likelihood(self):
        df = _frame(60, seed=2)
        fit = sp.dlm("y ~ x", df)
        est = fit.variances["estimate"].to_numpy()
        for j in range(3):
            for factor in (0.7, 1.4):
                other = np.maximum(est, 1e-8).copy()
                other[j] *= factor
                alt = sp.dlm("y ~ x", df, obs_var=other[0], state_var=other[1:])
                # 1e-7 of slack for the stopping rule of the optimiser.
                assert alt.loglik <= fit.loglik + 1e-7

    def test_fixed_observation_variance_is_kept_and_the_rest_estimated(self):
        df = _frame(60, seed=2)
        fit = sp.dlm("y ~ x", df, obs_var=0.3)
        assert fit.variances.loc["obs", "estimate"] == 0.3
        est = fit.variances["estimate"].to_numpy()
        for factor in (0.7, 1.4):
            other = np.maximum(est[1:], 1e-8) * factor
            alt = sp.dlm("y ~ x", df, obs_var=0.3, state_var=other)
            assert alt.loglik <= fit.loglik + 1e-7

    def test_fixed_state_variance_leaves_only_the_observation_variance(self):
        df = _frame(60, seed=2)
        fit = sp.dlm("y ~ x", df, state_var=[0.01, 0.04])
        assert fit.variances["estimate"].iloc[1:].tolist() == [0.01, 0.04]
        v = fit.variances.loc["obs", "estimate"]
        for factor in (0.8, 1.25):
            alt = sp.dlm("y ~ x", df, obs_var=v * factor, state_var=[0.01, 0.04])
            assert alt.loglik <= fit.loglik + 1e-7

    def test_rescaling_the_outcome_rescales_the_fit(self):
        df = _frame(60, seed=2)
        a = sp.dlm("y ~ x", df, constant=["Intercept"])
        b = sp.dlm("y ~ x", df.assign(y=100.0 * df["y"]), constant=["Intercept"])
        # The model is scale equivariant; 1e-3 relative is the precision
        # of a flat likelihood maximised numerically from scaled starts.
        np.testing.assert_allclose(
            b.variances["estimate"], 1e4 * a.variances["estimate"], rtol=1e-3
        )
        # The prior variance 1e7 is not rescaled with the data, so it is
        # relatively tighter for the larger series: 1e-3 at the first date.
        np.testing.assert_allclose(b.smoothed["x"], 100.0 * a.smoothed["x"], rtol=1e-3)


class TestForecast:
    def test_mean_and_interval_follow_the_random_walk(self):
        df = _frame()
        fit = _fixed(df, level=0.9)
        future = pd.DataFrame({"x": [0.3, -1.2, 2.0], "unused": 1.0})
        fc = fit.forecast(3, future)
        m_last = fit.filtered[["Intercept", "x"]].iloc[-1].to_numpy()
        _, C, _, _, _ = _kalman(df["y"].to_numpy(), _design(df), V_TRUE, W_TRUE, M0, C0)
        z = stats.norm.ppf(0.95)
        for h in range(3):
            xf = np.array([1.0, future["x"].iloc[h]])
            sd = np.sqrt(xf @ (C[-1] + (h + 1) * np.diag(W_TRUE)) @ xf + V_TRUE)
            assert fc["mean"].iloc[h] == pytest.approx(xf @ m_last, abs=1e-10)
            assert fc["lower"].iloc[h] == pytest.approx(xf @ m_last - z * sd, abs=1e-9)
            assert fc["upper"].iloc[h] == pytest.approx(xf @ m_last + z * sd, abs=1e-9)

    def test_extra_rows_of_future_data_are_not_used(self):
        fit = _fixed(_frame())
        future = pd.DataFrame({"x": [0.3, -1.2, 2.0]})
        pd.testing.assert_frame_equal(
            fit.forecast(2, future), fit.forecast(2, future.iloc[:2])
        )

    def test_local_level_needs_no_data_and_has_a_flat_mean(self):
        df = _frame()
        fit = sp.dlm("y ~ 1", df, obs_var=0.5, state_var=0.1, C0=4.0)
        fc = fit.forecast(4)
        assert np.ptp(fc["mean"].to_numpy()) == 0.0
        assert fc["mean"].iloc[0] == pytest.approx(fit.params["Intercept"])
        c_last = fit.filtered["Intercept_sd"].iloc[-1] ** 2
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(
            (fc["upper"] - fc["lower"]) / (2 * z),
            np.sqrt(c_last + 0.1 * np.arange(1, 5) + 0.5),
            atol=1e-10,
        )
        pd.testing.assert_frame_equal(fc, fit.forecast(4, pd.DataFrame(index=range(4))))

    def test_refusals(self):
        fit = _fixed(_frame())
        with pytest.raises(MethodIncompatibility, match="Pass data="):
            fit.forecast(2)
        with pytest.raises(MethodIncompatibility, match="lacks the regressors"):
            fit.forecast(2, pd.DataFrame({"z": [1.0, 2.0]}))
        with pytest.raises(MethodIncompatibility, match="1 rows; 2 steps"):
            fit.forecast(2, pd.DataFrame({"x": [1.0]}))


class TestGibbs:
    def test_first_sweeps_equal_the_conditionals_written_out(self):
        # Two sweeps replayed with the same random stream: states by
        # backward sampling, then each variance from its inverse-gamma
        # conditional with shape (alpha0 + n) / 2 and scale
        # (delta0 + sum of squares) / 2.
        df = _frame(30)
        y, X = df["y"].to_numpy().copy(), _design(df)
        n, k = X.shape
        prior_v, prior_w = (3.0, 0.4), (5.0, 0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fit = sp.dlm(
                "y ~ x",
                df,
                method="gibbs",
                m0=M0,
                C0=C0,
                obs_var_prior=prior_v,
                state_var_prior=prior_w,
                draws=2,
                burnin=0,
                seed=11,
            )
        rng = np.random.default_rng(11)
        V = 0.5 * np.var(y)
        W = np.full(k, 0.01 * np.var(y))
        expected = []
        for _ in range(2):
            m, C, R, *_ = dlm_module._filter_py(y, X, V, W, M0, C0)
            theta = dlm_module._backward_sample_py(
                m, C, R, rng.standard_normal((n + 1, k))
            )
            e = y - np.einsum("tk,tk->t", X, theta[1:])
            V = (prior_v[1] + e @ e) / 2.0 / rng.gamma((prior_v[0] + n) / 2.0)
            ss = (np.diff(theta, axis=0) ** 2).sum(axis=0)
            W = np.array(
                [
                    (prior_w[1] + ss[q]) / 2.0 / rng.gamma((prior_w[0] + n) / 2.0)
                    for q in range(k)
                ]
            )
            expected.append(np.append(V, W))
        np.testing.assert_allclose(fit.draws.to_numpy(), expected, rtol=1e-9)

    def test_seed_reproduces_and_thin_keeps_the_number_of_draws(self):
        df = _frame(30)
        kw = dict(method="gibbs", m0=M0, C0=C0, draws=40, burnin=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            a = sp.dlm("y ~ x", df, seed=7, **kw)
            b = sp.dlm("y ~ x", df, seed=7, **kw)
            c = sp.dlm("y ~ x", df, seed=7, thin=3, **kw)
        pd.testing.assert_frame_equal(a.draws, b.draws)
        pd.testing.assert_frame_equal(a.smoothed, b.smoothed)
        assert c.draws.shape == (40, 3)
        assert list(a.variances.columns) == [
            "estimate",
            "sd",
            "lower",
            "median",
            "upper",
            "ess",
        ]
        np.testing.assert_allclose(a.variances["estimate"], a.draws.mean().to_numpy())

    def test_fixed_variances_leave_only_the_states_to_sample(self):
        df = _frame(30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            g = _fixed(df, method="gibbs", draws=600, burnin=0, seed=2)
        exact = _fixed(df)
        assert (g.draws["obs"] == V_TRUE).all()
        assert (g.draws["state:x"] == W_TRUE[1]).all()
        # With the variances fixed the 600 state paths are independent
        # draws from the smoothing distribution: their mean is within six
        # Monte Carlo standard errors of the smoother at every date, and
        # their spread within 25% of the smoothed sd (the sd of a sample
        # standard deviation over 600 draws is about 3%).
        for nm in ("Intercept", "x"):
            mc_se = exact.smoothed[f"{nm}_sd"] / np.sqrt(600)
            assert (np.abs(g.smoothed[nm] - exact.smoothed[nm]) < 6 * mc_se).all()
            ratio = g.smoothed[f"{nm}_sd"] / exact.smoothed[f"{nm}_sd"]
            assert ratio.between(0.75, 1.25).all()
        # The filter output of a Gibbs fit is at the posterior-mean
        # variances, here the fixed ones.
        pd.testing.assert_frame_equal(g.filtered, exact.filtered)
        assert g.loglik == pytest.approx(exact.loglik, abs=1e-12)

    def test_bands_are_quantiles_of_the_stored_paths(self):
        df = _frame(30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            g = _fixed(df, method="gibbs", draws=50, burnin=0, seed=2, level=0.8)
        rng = np.random.default_rng(2)
        y, X = df["y"].to_numpy().copy(), _design(df)
        m, C, R, *_ = dlm_module._filter_py(y, X, V_TRUE, W_TRUE, M0, C0)
        paths = np.array(
            [
                dlm_module._backward_sample_py(
                    m, C, R, rng.standard_normal((len(y) + 1, 2))
                )[1:]
                for _ in range(50)
            ]
        )
        np.testing.assert_allclose(g.smoothed["x"], paths[:, :, 1].mean(axis=0))
        np.testing.assert_allclose(
            g.smoothed["x_lower"], np.quantile(paths[:, :, 1], 0.1, axis=0)
        )
        np.testing.assert_allclose(
            g.smoothed["Intercept_upper"], np.quantile(paths[:, :, 0], 0.9, axis=0)
        )

    def test_constant_term_has_one_value_under_a_proper_prior(self):
        df = _frame(30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            g = sp.dlm(
                "y ~ x",
                df,
                method="gibbs",
                constant=["x"],
                C0=100.0,
                draws=60,
                burnin=10,
                seed=1,
            )
        assert (g.draws["state:x"] == 0.0).all()
        # 1e-6: square roots of rounding-level conditional variances.
        assert np.ptp(g.smoothed["x"].to_numpy()) < 1e-6
        assert np.ptp(g.smoothed["x_sd"].to_numpy()) < 1e-6

    def test_constant_term_has_one_value_under_the_default_prior(self):
        df = _frame(30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            g = sp.dlm(
                "y ~ x",
                df,
                method="gibbs",
                constant=["x"],
                draws=60,
                burnin=10,
                seed=1,
            )
        # A coefficient with zero state variance is the same number at
        # every date in every draw: carried back unchanged, so exactly.
        assert np.ptp(g.smoothed["x"].to_numpy()) == 0.0
        assert np.ptp(g.smoothed["x_sd"].to_numpy()) == 0.0
        assert np.ptp(g.smoothed["x_lower"].to_numpy()) == 0.0

    @pytest.mark.parametrize("prior", [1e7, 100.0])
    def test_paths_with_a_constant_term_have_the_smoother_moments(self, prior):
        # Variances fixed: the draws of the paths come from the smoothing
        # distribution, whose mean and sd the Kalman smoother gives in
        # closed form. Checked for the constant coefficient and for the
        # moving one, at the first date (where the diffuse prior bites),
        # the second and the last.
        df = _frame(30)
        kw = dict(constant=["x"], obs_var=V_TRUE, state_var=0.05, C0=prior)
        exact = sp.dlm("y ~ x", df, **kw).smoothed
        draws = 4000
        g = sp.dlm("y ~ x", df, method="gibbs", draws=draws, burnin=0, seed=3, **kw)
        for col in ("Intercept", "x"):
            for row in (0, 1, 29):
                mean, sd = exact[col].iloc[row], exact[f"{col}_sd"].iloc[row]
                # 4 Monte Carlo standard errors of a mean of independent
                # draws (the variances are fixed, so draws are independent)
                assert abs(g.smoothed[col].iloc[row] - mean) < 4 * sd / np.sqrt(draws)
                # sd of a sample sd: sd / sqrt(2 draws); 4 of them is 4.5%
                assert g.smoothed[f"{col}_sd"].iloc[row] == pytest.approx(sd, rel=0.05)

    def test_smoother_keeps_a_constant_term_constant_under_the_default_prior(self):
        # The smoothed constant coefficient is its filtered value at the
        # last date, at every date. Under C0 = 1e7 the textbook recursion
        # C + J (S - R) J' gave another standard error at the first date.
        df = _frame(30)
        fit = sp.dlm("y ~ x", df, constant=["x"], obs_var=V_TRUE, state_var=0.05)
        assert (fit.smoothed["x"] == fit.filtered["x"].iloc[-1]).all()
        assert (fit.smoothed["x_sd"] == fit.filtered["x_sd"].iloc[-1]).all()

    def test_smoother_with_a_constant_term_in_extended_precision(self):
        # The same recursions in 50-digit arithmetic, where the diffuse
        # prior costs nothing: mean and sd of both coefficients at the
        # first two dates and one in the middle.
        mp = pytest.importorskip("mpmath")
        df = _frame(30)
        fit = sp.dlm("y ~ x", df, constant=["x"], obs_var=V_TRUE, state_var=0.05)
        with mp.workdps(50):
            V, W = mp.mpf(V_TRUE), mp.diag([mp.mpf("0.05"), 0])
            rows = [mp.matrix([[1, mp.mpf(float(v))]]) for v in df["x"]]
            m, C = mp.matrix([0, 0]), mp.eye(2) * mp.mpf(10) ** 7
            ms, Cs, Rs = [m], [C], []
            for x_t, y_t in zip(rows, df["y"]):
                R = C + W
                K = R * x_t.T / ((x_t * R * x_t.T)[0] + V)
                m = m + K * (mp.mpf(float(y_t)) - (x_t * m)[0])
                C = R - K * x_t * R
                ms.append(m), Cs.append(C), Rs.append(R)
            s, S = ms[-1], Cs[-1]
            exact = {}
            for t in range(len(rows) - 1, 0, -1):
                J = Cs[t] * Rs[t] ** -1
                s, S = ms[t] + J * (s - ms[t]), Cs[t] + J * (S - Rs[t]) * J.T
                exact[t] = (
                    [float(v) for v in s],
                    [float(mp.sqrt(S[i, i])) for i in (0, 1)],
                )
        for row in (0, 1, 15):
            mean, sd = exact[row + 1]
            got = fit.smoothed.iloc[row]
            # 1e-7: the inverse of R at the first dates has condition
            # number about 1e8 in double precision.
            np.testing.assert_allclose(got[["Intercept", "x"]], mean, rtol=1e-7)
            np.testing.assert_allclose(got[["Intercept_sd", "x_sd"]], sd, rtol=1e-7)

    def test_compiled_backward_sampler_equals_python_with_a_constant_term(self):
        df = _frame(25)
        y, X = df["y"].to_numpy(), _design(df)
        W = np.array([0.03, 0.0])
        m, C, R, *_ = dlm_module._filter_py(
            y, X, V_TRUE, W, np.zeros(2), 1e7 * np.eye(2)
        )
        z = np.random.default_rng(5).standard_normal(m.shape)
        py = dlm_module._backward_sample_py(m, C, R, z)
        np.testing.assert_allclose(
            dlm_module._kernels()["backward"](m, C, R, z), py, rtol=1e-10, atol=1e-10
        )
        assert np.ptp(py[:, 1]) == 0.0 and np.ptp(py[:, 0]) > 0.0

    def test_no_mixing_warning_when_no_variance_is_sampled(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            _fixed(_frame(30), method="gibbs", draws=50, burnin=0, seed=2)

    def test_priors_must_be_positive(self):
        with pytest.raises(MethodIncompatibility, match="positive pairs"):
            sp.dlm("y ~ x", _frame(), method="gibbs", obs_var_prior=(0.0, 1.0))
        with pytest.raises(MethodIncompatibility, match="positive pairs"):
            sp.dlm("y ~ x", _frame(), method="gibbs", state_var_prior=(1.0, -1.0))


class TestRefusalsAndDegenerateInputs:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(method="em"), "'mle' or 'gibbs'"),
            (dict(level=1.0), "level must be in"),
            (dict(time="nope"), "time column"),
            (dict(constant=["z"]), "not in the model"),
            (dict(m0=[1.0, 2.0, 3.0]), "m0 must be"),
            (dict(C0=np.eye(3)), "C0 must be"),
            (dict(C0=[1.0, 2.0, 3.0]), "C0 must be"),
            (dict(state_var=[0.1, 0.1, 0.1]), "state_var must be"),
            (dict(state_var=-0.1), "state_var must be"),
            (dict(obs_var=0.0), "obs_var must be positive"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.dlm("y ~ x", _frame(), **kwargs)

    def test_data_must_be_a_frame(self):
        with pytest.raises(MethodIncompatibility, match="DataFrame"):
            sp.dlm("y ~ x", _frame().to_numpy())

    def test_missing_rows_are_refused_not_dropped(self):
        df = _frame()
        gap = df.assign(y=df["y"].where(df["t"] != 7))
        with pytest.raises(MethodIncompatibility, match="1 rows have missing"):
            sp.dlm("y ~ x", gap)

    def test_too_few_observations(self):
        with pytest.raises(DataInsufficient, match="4 observations for 2"):
            sp.dlm("y ~ x", _frame(4))

    def test_summary_plot_and_dict(self):
        fit = _fixed(_frame())
        text = fit.summary()
        assert "Dynamic linear model (mle)" in text
        assert "state:x" in text
        d = fit.to_dict()
        assert d["n_obs"] == 40
        assert d["variances"]["obs"]["estimate"] == V_TRUE
        assert "forecast_variance" not in d["model_info"]
        fig = fit.plot()
        try:
            assert len(fig.axes) == 2
            line = fig.axes[1].lines[0]
            np.testing.assert_allclose(line.get_ydata(), fit.smoothed["x"].to_numpy())
        finally:
            plt.close(fig)
        fig = fit.plot(terms=["x"])
        try:
            assert [ax.get_title() for ax in fig.axes] == ["x"]
        finally:
            plt.close(fig)

    def test_constant_series_gets_an_answer_or_a_named_refusal(self):
        df = pd.DataFrame({"y": np.ones(30)})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sp.dlm("y ~ 1", df)
        except DataInsufficient:
            return
        np.testing.assert_allclose(fit.smoothed["Intercept"], 1.0, atol=1e-6)

    def test_singular_prior_covariance_is_allowed(self):
        # A coefficient known before the first date (prior variance zero)
        # is a valid prior; only the zero matrix and indefinite ones are not.
        fit = _fixed(_frame(), C0=[1.0, 0.0], m0=[0.0, 1.5])
        assert np.isfinite(fit.loglik)

    def test_exact_fit_is_refused_when_the_observation_variance_is_estimated(self):
        df = _frame()
        exact = df.assign(y=1.0 + 2.0 * df["x"])
        with pytest.raises(DataInsufficient, match="exact linear function"):
            sp.dlm("y ~ x", exact)
        # With both variances given the series can be filtered.
        fit = _fixed(exact)
        np.testing.assert_allclose(fit.smoothed["x"].iloc[-1], 2.0, atol=0.2)

    def test_missing_time_is_refused(self):
        df = _frame().astype({"t": float})
        df = df.assign(t=df["t"].where(df["t"] != 5))
        with pytest.raises(MethodIncompatibility, match="time"):
            _fixed(df, time="t")

    def test_time_with_a_repeated_index(self):
        df = _frame()
        repeated = df.set_axis(np.zeros(len(df), dtype=int), axis=0)
        a = _fixed(df, time="t")
        b = _fixed(repeated, time="t")
        np.testing.assert_allclose(b.smoothed.to_numpy(), a.smoothed.to_numpy())

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(state_var=np.nan),
            dict(state_var=np.inf),
            dict(obs_var=np.inf),
            dict(C0=-1.0),
            dict(C0=np.nan),
            dict(C0=[1.0, -1.0]),
            dict(C0=np.array([[1.0, 2.0], [2.0, 1.0]])),  # indefinite
            dict(C0=np.array([[1.0, 0.5], [0.0, 1.0]])),  # not symmetric
            dict(m0=np.nan),
        ],
    )
    def test_invalid_variances_are_refused(self, kwargs):
        base = dict(obs_var=V_TRUE, state_var=0.01)
        base.update(kwargs)
        with pytest.raises(MethodIncompatibility):
            sp.dlm("y ~ x", _frame(), **base)


class TestKnownInitialState:
    def test_zero_prior_covariance_is_the_known_initial_state(self):
        """C0 = 0 says the coefficients before the first observation equal
        m0. The likelihood is the Kalman recursion started from C = 0, and
        the limit of the one with a vanishing prior covariance."""
        df = _frame()
        zero = _fixed(df, C0=0.0)
        tiny = _fixed(df, C0=1e-12)
        *_, ll = _kalman(
            df["y"].to_numpy(), _design(df), V_TRUE, W_TRUE, M0, np.zeros((2, 2))
        )
        # same recursion, Joseph-form update: rounding only
        assert zero.loglik == pytest.approx(ll, abs=1e-9)
        # d loglik / d C0 is O(1) here, so 1e-12 moves it by about 1e-11
        assert zero.loglik == pytest.approx(tiny.loglik, abs=1e-8)
        same = _fixed(df, C0=np.zeros((2, 2)))
        assert same.loglik == zero.loglik
