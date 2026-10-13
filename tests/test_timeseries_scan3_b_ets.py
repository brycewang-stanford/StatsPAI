"""Options and degenerate inputs of ``sp.ets`` and ``sp.simple_forecast``.

An additive-error ETS model is a linear innovations state space model
``y_t = w' x_{t-1} + e_t``, ``x_t = F x_{t-1} + g e_t``. Fitted values,
states, point forecasts and forecast variances all follow from ``F``,
``g`` and ``w``, which the tests write out and iterate by brute force.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import DataInsufficient, MethodIncompatibility  # noqa: E402


def _series(n: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = np.tile([2.0, -1.0, -3.0, 2.0], n // 4 + 1)[:n]
    return 20.0 + 0.2 * t + season + 0.5 * rng.normal(size=n)


def _system(fit):
    """``F``, ``g``, ``w`` of a fitted additive-error model, in the order
    of the columns of ``fit.states``."""
    p = fit.params
    alpha = p["alpha"]
    beta = p.get("beta", 0.0)
    gamma = p.get("gamma", 0.0)
    phi = p.get("phi", 1.0)
    cols = list(fit.states.columns)
    k = len(cols)
    F = np.zeros((k, k))
    g = np.zeros(k)
    w = np.zeros(k)
    F[0, 0] = 1.0
    w[0] = 1.0
    g[0] = alpha
    if "slope" in cols:
        F[0, 1] = phi
        F[1, 1] = phi
        w[1] = phi
        g[1] = beta
    if "season" in cols:
        s = cols.index("season")
        m = fit.period
        F[s, s + m - 1] = 1.0  # s_t = s_{t-m} + gamma e_t
        for j in range(1, m):
            F[s + j, s + j - 1] = 1.0  # the lags shift down
        w[s + m - 1] = 1.0
        g[s] = gamma
    return F, g, w


ADDITIVE = [
    ("ANN", 1),
    ("AAN", 1),
    ("AAdN", 1),
    ("ANA", 4),
    ("AAA", 4),
    ("AAdA", 4),
]


class TestAdditiveModelsAsStateSpace:
    @pytest.mark.parametrize("model, period", ADDITIVE)
    def test_states_and_fitted_values_follow_the_recursion(self, model, period):
        y = _series()
        fit = sp.ets(y, model, period=period)
        F, g, w = _system(fit)
        x = fit.states.iloc[0].to_numpy(dtype=float)
        for t in range(len(y)):
            fitted = w @ x
            # Rounding only: the same recursion at the same parameters.
            assert fit.fitted_values[t] == pytest.approx(fitted, abs=1e-9)
            e = y[t] - fitted
            assert fit.residuals[t] == pytest.approx(e, abs=1e-9)
            x = F @ x + g * e
            np.testing.assert_allclose(fit.states.iloc[t + 1], x, atol=1e-9)

    @pytest.mark.parametrize("model, period", ADDITIVE)
    def test_forecast_mean_and_variance_from_the_moving_average_form(
        self, model, period
    ):
        fit = sp.ets(_series(), model, period=period)
        F, g, w = _system(fit)
        x = fit.states.iloc[-1].to_numpy(dtype=float)
        fc = fit.forecast(9, level=90)
        z = stats.norm.ppf(0.95)
        power = np.eye(len(x))
        acc = 1.0
        for h in range(1, 10):
            mean = w @ power @ x
            assert fc["forecast"].iloc[h - 1] == pytest.approx(mean, abs=1e-9)
            half = z * np.sqrt(fit.sigma2 * acc)
            assert fc["upper_90"].iloc[h - 1] - mean == pytest.approx(half, abs=1e-9)
            assert mean - fc["lower_90"].iloc[h - 1] == pytest.approx(half, abs=1e-9)
            # y_{n+h} = mean + e_h + sum_j (w' F^{j-1} g) e_{h-j}
            acc += float(w @ power @ g) ** 2
            power = F @ power

    def test_seasonal_model_on_fewer_than_three_cycles(self):
        # Starting values then come from a harmonic regression, not from
        # a classical decomposition; the fit is still the recursion.
        y = _series(10)
        fit = sp.ets(y, "ANA", period=4)
        F, g, w = _system(fit)
        x = fit.states.iloc[0].to_numpy(dtype=float)
        for t in range(len(y)):
            assert fit.fitted_values[t] == pytest.approx(w @ x, abs=1e-9)
            x = F @ x + g * (y[t] - w @ x)
        comp = fit.components()
        assert list(comp.columns) == ["y", "level", "season", "remainder"]
        np.testing.assert_allclose(comp["season"], fit.states["season"].iloc[1:])

    def test_simple_exponential_smoothing_by_hand(self):
        y = _series()
        fit = sp.ets(y, "ANN", alpha=0.3)
        level = fit.initial_state["l"]
        sse = 0.0
        for t, v in enumerate(y):
            assert fit.fitted_values[t] == pytest.approx(level, abs=1e-10)
            sse += (v - level) ** 2
            level += 0.3 * (v - level)
        fc = fit.forecast(6, level=95)
        # Flat forecast at the last level.
        np.testing.assert_allclose(fc["forecast"], level, atol=1e-10)
        # alpha is fixed: the initial level and the variance are estimated.
        assert fit.fixed == {"alpha": 0.3}
        assert fit.n_params == 2
        assert fit.sigma2 == pytest.approx(sse / (len(y) - 1), rel=1e-12)
        assert fit.log_likelihood == pytest.approx(
            -0.5 * len(y) * np.log(sse), rel=1e-12
        )
        half = stats.norm.ppf(0.975) * np.sqrt(fit.sigma2 * (1 + np.arange(6) * 0.3**2))
        np.testing.assert_allclose(fc["upper_95"] - fc["forecast"], half, atol=1e-10)

    def test_fixed_level_start_is_the_least_squares_one(self):
        # With alpha fixed, the only free parameter is the initial level
        # and the fitted values are linear in it: the optimum solves a
        # one-dimensional least-squares problem in closed form.
        y = _series()
        fit = sp.ets(y, "ANN", alpha=0.3)
        n = len(y)

        def errors(l0):
            level, out = l0, np.empty(n)
            for t in range(n):
                out[t] = y[t] - level
                level += 0.3 * out[t]
            return out

        e0 = errors(0.0)
        slope = errors(1.0) - e0
        best = -(e0 @ slope) / (slope @ slope)
        # 1e-5: the stopping rule of the optimiser (1e-4 in -2 log L).
        assert fit.initial_state["l"] == pytest.approx(best, rel=1e-5)

    def test_information_criteria(self):
        y = _series()
        for model, period in ADDITIVE:
            fit = sp.ets(y, model, period=period)
            n, k = fit.n, fit.n_params
            neg2 = -2.0 * fit.log_likelihood
            assert fit.aic == pytest.approx(neg2 + 2 * k)
            assert fit.bic == pytest.approx(neg2 + np.log(n) * k)
            assert fit.aicc == pytest.approx(fit.aic + 2 * k * (k + 1) / (n - k - 1))
            assert fit.sigma2 == pytest.approx(
                np.sum(fit.residuals**2) / (n - k + 1), rel=1e-12
            )
            assert fit.mse == pytest.approx(np.mean(fit.residuals**2), rel=1e-12)

    def test_damping_parameter_can_be_fixed(self):
        fit = sp.ets(_series(), "AAN", phi=0.9)
        assert fit.model == "ETS(A,Ad,N)"
        assert fit.damped is True
        assert fit.params["phi"] == 0.9
        named = sp.ets(_series(), "AAdN", phi=0.9)
        assert named.log_likelihood == pytest.approx(fit.log_likelihood, abs=1e-8)

    def test_rescaling_an_additive_model(self):
        y = _series()
        a = sp.ets(y, "AAN")
        b = sp.ets(1000.0 * y, "AAN")
        # Scale equivariance, to the precision of the optimiser.
        np.testing.assert_allclose(b.params, a.params, atol=2e-3)
        np.testing.assert_allclose(
            b.forecast(4)["forecast"], 1000.0 * a.forecast(4)["forecast"], rtol=1e-4
        )


class TestMultiplicativeError:
    def test_local_level_with_relative_errors(self):
        y = _series()
        fit = sp.ets(y, "MNN")
        alpha = fit.params["alpha"]
        level = fit.initial_state["l"]
        sumlog = 0.0
        for t, v in enumerate(y):
            assert fit.fitted_values[t] == pytest.approx(level, abs=1e-9)
            e = (v - level) / level
            assert fit.residuals[t] == pytest.approx(e, abs=1e-10)
            sumlog += np.log(abs(level))
            level *= 1 + alpha * e
        # -2 log L = n log(sum e^2) + 2 sum log |fitted|, constants dropped.
        assert -2 * fit.log_likelihood == pytest.approx(
            len(y) * np.log(np.sum(fit.residuals**2)) + 2 * sumlog, rel=1e-12
        )
        # y_{n+h} = l_n prod_{j<h} (1 + alpha e_j) (1 + e_h) with independent
        # errors, so E[y^2] = l_n^2 (1 + alpha^2 s2)^(h-1) (1 + s2).
        fc = fit.forecast(5, level=95)
        h = np.arange(1, 6)
        var = level**2 * ((1 + alpha**2 * fit.sigma2) ** (h - 1) * (1 + fit.sigma2) - 1)
        np.testing.assert_allclose(fc["forecast"], level, atol=1e-9)
        np.testing.assert_allclose(
            fc["upper_95"] - fc["forecast"],
            stats.norm.ppf(0.975) * np.sqrt(var),
            rtol=1e-10,
        )

    def test_multiplicative_trend_recursion_and_simulated_intervals(self):
        y = _series()
        fit = sp.ets(y, "MMN")
        alpha, beta = fit.params["alpha"], fit.params["beta"]
        level, growth = fit.states.iloc[0]
        for t, v in enumerate(y):
            mu = level * growth
            assert fit.fitted_values[t] == pytest.approx(mu, abs=1e-8)
            e = (v - mu) / mu
            level, growth = mu * (1 + alpha * e), growth * (1 + beta * e)
        fc = fit.forecast(4, level=80, n_paths=200, seed=1)
        # Point forecast: the level compounded at the last growth rate.
        np.testing.assert_allclose(
            fc["forecast"], level * growth ** np.arange(1, 5), rtol=1e-9
        )
        # No analytic variance for this class: the bounds are quantiles of
        # simulated paths and so depend on the seed and on nothing else.
        pd.testing.assert_frame_equal(
            fc, fit.forecast(4, level=80, n_paths=200, seed=1)
        )
        assert (fc["lower_80"] < fc["forecast"]).all()
        assert (fc["forecast"] < fc["upper_80"]).all()

    def test_holt_winters_multiplicative_moments_agree_with_simulation(self):
        fit = sp.ets(_series(), "MAM", period=4)
        fc = fit.forecast(4, level=95)
        sim = fit.simulate(4, n_paths=20000, seed=3)
        sd = (fc["upper_95"] - fc["lower_95"]) / (2 * stats.norm.ppf(0.975))
        # 20,000 paths: the Monte Carlo standard error of a mean is
        # sd / 141 and of a standard deviation sd / 200. Bounds of six
        # standard errors.
        assert (np.abs(sim.mean(axis=1) - fc["forecast"]) < 6 * sd / 141).all()
        assert (np.abs(sim.std(axis=1) / sd - 1) < 6 / 200).all()

    def test_ljung_box_gives_up_no_degrees_of_freedom_for_ets(self):
        fit = sp.ets(_series(), "AAA", period=4)
        out = sp.ljungbox(fit)
        assert fit.n_arma_params == 0
        assert out.index.tolist() == [8] and out["df"].tolist() == [8]

    @pytest.mark.parametrize("model", ["MNN", "AMN", "ANM"])
    def test_needs_a_positive_series(self, model):
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            sp.ets(_series() - 25.0, model, period=4)


class TestSimulatedIntervals:
    def test_simulate_true_uses_the_quantiles_of_normal_paths(self):
        y = _series()
        fit = sp.ets(y, "ANN")
        alpha = fit.params["alpha"]
        last = fit.states["level"].iloc[-1]
        fc = fit.forecast(4, level=80, simulate=True, n_paths=300, seed=5)
        innov = np.random.default_rng(5).normal(0.0, np.sqrt(fit.sigma2), size=(300, 4))
        level = np.full(300, last)
        paths = np.empty((300, 4))
        for h in range(4):
            paths[:, h] = level + innov[:, h]
            level = level + alpha * innov[:, h]
        np.testing.assert_allclose(
            fc["lower_80"],
            np.quantile(paths, 0.1, axis=0, method="median_unbiased"),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            fc["upper_80"],
            np.quantile(paths, 0.9, axis=0, method="median_unbiased"),
            atol=1e-9,
        )
        np.testing.assert_allclose(fc["forecast"], last, atol=1e-9)

    def test_bootstrap_resamples_centred_residuals(self):
        y = _series()
        fit = sp.ets(y, "ANN")
        alpha = fit.params["alpha"]
        last = fit.states["level"].iloc[-1]
        fc = fit.forecast(3, level=80, bootstrap=True, n_paths=200, seed=9)
        pool = fit.residuals - fit.residuals.mean()
        innov = np.random.default_rng(9).choice(pool, size=(200, 3), replace=True)
        level = np.full(200, last)
        paths = np.empty((200, 3))
        for h in range(3):
            paths[:, h] = level + innov[:, h]
            level = level + alpha * innov[:, h]
        np.testing.assert_allclose(
            fc["upper_80"],
            np.quantile(paths, 0.9, axis=0, method="median_unbiased"),
            atol=1e-9,
        )

    def test_simulate_returns_reproducible_paths(self):
        fit = sp.ets(_series(), "AAN")
        a = fit.simulate(5, n_paths=3, seed=1)
        b = fit.simulate(5, n_paths=3, seed=1)
        pd.testing.assert_frame_equal(a, b)
        assert a.shape == (5, 3)
        assert a.index.tolist() == [48, 49, 50, 51, 52]


class TestBoxCox:
    @pytest.mark.parametrize("lam", [0.0, 0.5])
    def test_model_of_the_transform_back_transformed(self, lam):
        y = np.exp(_series() / 10.0)
        z = np.log(y) if lam == 0.0 else (y**lam - 1.0) / lam
        inner = sp.ets(z, "AAN")
        fit = sp.ets(y, "AAN", boxcox=lam)

        def back(v):
            v = np.asarray(v, dtype=float)
            return np.exp(v) if lam == 0.0 else (lam * v + 1.0) ** (1.0 / lam)

        assert fit.boxcox == lam
        np.testing.assert_allclose(fit.params, inner.params, atol=1e-12)
        np.testing.assert_allclose(fit.fitted_values, back(inner.fitted_values))
        np.testing.assert_allclose(fit.residuals, inner.residuals, atol=1e-12)
        np.testing.assert_allclose(fit.response_residuals, y - fit.fitted_values)
        # Quantiles commute with a monotone map: every column is the
        # back-transform of the forecast table of the transformed series.
        np.testing.assert_allclose(
            fit.forecast(5).to_numpy(), back(inner.forecast(5).to_numpy()), rtol=1e-12
        )

    @pytest.mark.parametrize("lam", [0.0, 0.5])
    def test_bias_adjustment_is_the_second_order_mean(self, lam):
        y = np.exp(_series() / 10.0)
        z = np.log(y) if lam == 0.0 else (y**lam - 1.0) / lam
        inner = sp.ets(z, "AAN").forecast(5, level=(80, 95))
        fc = sp.ets(y, "AAN", boxcox=lam, biasadj=True).forecast(5, level=(80, 95))
        median = sp.ets(y, "AAN", boxcox=lam).forecast(5, level=(80, 95))
        var = (
            (inner["upper_95"] - inner["lower_95"]) / (2 * stats.norm.ppf(0.975))
        ) ** 2
        mu = inner["forecast"]
        if lam == 0.0:
            mean = np.exp(mu) * (1 + var / 2)
        else:
            base = lam * mu + 1
            mean = base ** (1 / lam) * (1 + var * (1 - lam) / (2 * base**2))
        np.testing.assert_allclose(fc["forecast"], mean, rtol=1e-12)
        # The mean of a right-skewed forecast distribution is above its
        # median; the interval end points are not adjusted.
        assert (fc["forecast"] > median["forecast"]).all()
        np.testing.assert_allclose(fc["lower_95"], median["lower_95"])
        np.testing.assert_allclose(fc["upper_80"], median["upper_80"])

    def test_automatic_choice_is_among_additive_models(self):
        y = np.exp(_series() / 10.0)
        fit = sp.ets(y, period=4, boxcox=0)
        assert not fit.candidates["model"].str.contains("M").any()
        assert fit.boxcox == 0.0

    def test_auto_uses_the_guerrero_parameter(self):
        y = np.exp(_series() / 10.0)
        fit = sp.ets(y, "ANN", period=4, boxcox="auto")
        assert fit.boxcox == pytest.approx(sp.boxcox_lambda(y, 4))

    def test_refusals(self):
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            sp.ets(_series() - 25.0, "ANN", boxcox=0)
        with pytest.raises(MethodIncompatibility, match="not a number or 'auto'"):
            sp.ets(_series(), "ANN", boxcox="log")
        with pytest.raises(MethodIncompatibility, match="not finite"):
            sp.ets(_series(), "ANN", boxcox=np.inf)


class TestMissingStretch:
    def test_states_move_on_with_a_zero_innovation(self):
        y = _series()
        gap = y.copy()
        gap[20:23] = np.nan
        fit = sp.ets(gap, "AAN")
        F, g, w = _system(fit)
        x = fit.states.iloc[0].to_numpy(dtype=float)
        sse = 0.0
        for t in range(len(gap)):
            fitted = w @ x
            assert fit.fitted_values[t] == pytest.approx(fitted, abs=1e-9)
            if np.isnan(gap[t]):
                assert np.isnan(fit.residuals[t])
                e = 0.0
            else:
                e = gap[t] - fitted
                sse += e * e
            x = F @ x + g * e
        assert fit.n == 45
        assert fit.n_missing == 3
        # The missing periods are left out of the likelihood.
        assert fit.log_likelihood == pytest.approx(-0.5 * 45 * np.log(sse), rel=1e-10)
        assert fit.sigma2 == pytest.approx(sse / (45 - fit.n_params + 1), rel=1e-10)
        assert fit.forecast(2).index.tolist() == [48, 49]

    def test_missing_ends_are_trimmed(self):
        y = _series()
        padded = np.concatenate([[np.nan, np.nan], y, [np.nan]])
        a = sp.ets(y, "ANN")
        b = sp.ets(padded, "ANN")
        assert b.n == a.n and b.n_missing == 0
        np.testing.assert_allclose(b.fitted_values, a.fitted_values)

    def test_bootstrap_pool_leaves_the_missing_residuals_out(self):
        gap = _series()
        gap[20:23] = np.nan
        fit = sp.ets(gap, "ANN")
        fc = fit.forecast(2, level=80, bootstrap=True, n_paths=100, seed=3)
        res = fit.residuals[np.isfinite(fit.residuals)]
        innov = np.random.default_rng(3).choice(
            res - res.mean(), size=(100, 2), replace=True
        )
        first = fit.states["level"].iloc[-1] + innov[:, 0]
        assert fc["lower_80"].iloc[0] == pytest.approx(
            np.quantile(first, 0.1, method="median_unbiased"), abs=1e-9
        )


class TestModelChoice:
    def test_chosen_model_minimises_the_criterion(self):
        y = _series()
        for ic in ("aicc", "aic", "bic"):
            fit = sp.ets(y, period=4, ic=ic)
            table = fit.candidates
            assert table[ic].is_monotonic_increasing
            assert table["model"].iloc[0] == fit.model
            assert getattr(fit, ic) == pytest.approx(table[ic].iloc[0])
            # Every candidate is the named model fitted on its own.
            row = table.iloc[-1]
            name = row["model"][4:-1].replace(",", "")
            alone = sp.ets(y, name, period=4)
            assert alone.log_likelihood == pytest.approx(
                row["log_likelihood"], abs=1e-6
            )

    def test_default_search_set(self):
        fit = sp.ets(_series(), period=4)
        models = set(fit.candidates["model"])
        # Restricted: no additive error with a multiplicative season, no
        # multiplicative trend.
        assert models == {
            f"ETS({e},{t},{s})"
            for e in "AM"
            for t in ("N", "A", "Ad")
            for s in "NAM"
            if not (e == "A" and s == "M")
        }

    def test_switches_widen_or_narrow_the_search(self):
        y = _series()
        additive = set(sp.ets(y, period=4, additive_only=True).candidates["model"])
        assert additive == {f"ETS(A,{t},{s})" for t in ("N", "A", "Ad") for s in "NA"}
        wide = set(
            sp.ets(
                y[:24], period=4, restrict=False, allow_multiplicative_trend=True
            ).candidates["model"]
        )
        assert len(wide) == 30
        assert "ETS(A,M,A)" in wide and "ETS(A,N,M)" in wide
        damped = set(sp.ets(y, "ZZN", damped=True).candidates["model"])
        assert damped == {"ETS(A,N,N)", "ETS(A,Ad,N)", "ETS(M,N,N)", "ETS(M,Ad,N)"}
        plain = set(sp.ets(y, "ZZN", damped=False).candidates["model"])
        assert plain == {"ETS(A,N,N)", "ETS(A,A,N)", "ETS(M,N,N)", "ETS(M,A,N)"}

    def test_non_positive_series_is_searched_among_additive_models(self):
        fit = sp.ets(_series() - 25.0, period=4)
        assert not fit.candidates["model"].str.contains("M").any()

    def test_one_cycle_or_a_long_period_drops_the_seasonal_models(self):
        y = _series()
        fit = sp.ets(y[:10], period=12)
        assert fit.season == "N"
        with pytest.warns(UserWarning, match="period=30 is above 24"):
            long = sp.ets(y, period=30)
        assert long.season == "N"
        with pytest.raises(MethodIncompatibility, match="29 seasonal starting"):
            sp.ets(y, "ANA", period=30)

    def test_named_model_spellings(self):
        y = _series()
        a = sp.ets(y, "AAdN")
        b = sp.ets(y, "ets(a, ad, n)")
        c = sp.ets(y, "AAN", damped=True)
        assert a.model == b.model == c.model == "ETS(A,Ad,N)"
        assert b.log_likelihood == a.log_likelihood == c.log_likelihood
        assert a.candidates is None

    def test_damping_a_model_without_trend_is_refused(self):
        with pytest.raises(MethodIncompatibility, match="damp"):
            sp.ets(_series(), "ANN", damped=True)
        # A name that says damped against damped=False.
        with pytest.raises(MethodIncompatibility, match="damped=False"):
            sp.ets(_series(), "AAdN", damped=False)
        # Not contradictory: the trend is left to be chosen.
        assert sp.ets(_series(), "ZZN", damped=True).model in (
            "ETS(A,N,N)",
            "ETS(A,Ad,N)",
            "ETS(M,N,N)",
            "ETS(M,Ad,N)",
        )


class TestInputsAndRefusals:
    def test_column_of_a_frame_and_the_forecast_index(self):
        y = _series()
        idx = pd.period_range("2015Q1", periods=len(y), freq="Q")
        frame = pd.DataFrame({"sales": y}, index=idx)
        fit = sp.ets("sales", "ANN", data=frame)
        ref = sp.ets(y, "ANN")
        np.testing.assert_allclose(fit.fitted_values, ref.fitted_values)
        fc = fit.forecast(3)
        assert fc.index.equals(pd.period_range("2027Q1", periods=3, freq="Q"))
        comp = fit.components()
        assert list(comp.columns) == ["sales", "level", "remainder"]
        assert comp.index.equals(idx)
        np.testing.assert_allclose(comp["level"], fit.states["level"].iloc[1:])
        dated = pd.Series(
            y, index=pd.date_range("2020-01-31", periods=len(y), freq="ME")
        )
        assert sp.ets(dated, "ANN").forecast(2).index[0] == pd.Timestamp("2024-01-31")
        stepped = pd.Series(y, index=np.arange(10, 10 + 5 * len(y), 5))
        assert sp.ets(stepped, "ANN").forecast(2).index.tolist() == [250, 255]

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            (dict(model="XYZ"), MethodIncompatibility, "not an ETS specification"),
            (dict(model="AN"), MethodIncompatibility, "not an ETS specification"),
            (dict(ic="hqic"), MethodIncompatibility, "ic='hqic'"),
            (dict(bounds="none"), MethodIncompatibility, "bounds='none'"),
            (dict(model="ANA"), MethodIncompatibility, "period=1"),
            (dict(period=1.5), MethodIncompatibility, "positive integer"),
            (dict(period=0), MethodIncompatibility, "positive integer"),
            (dict(model="ANN", beta=0.1), MethodIncompatibility, "has no beta"),
            (dict(model="ANN", alpha=1.5), DataInsufficient, "no model could"),
        ],
    )
    def test_invalid_arguments(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            sp.ets(_series(), **kwargs)

    def test_degenerate_series(self):
        with pytest.raises(DataInsufficient, match="constant"):
            sp.ets(np.full(20, 3.0))
        with pytest.raises(DataInsufficient, match="3 observations"):
            sp.ets([1.0, 2.0, 4.0])
        with pytest.raises(DataInsufficient, match="no finite values"):
            sp.ets(np.full(5, np.nan))
        with pytest.raises(DataInsufficient, match="no model could be fitted"):
            sp.ets(_series()[:7], "AAA", period=4)

    def test_wrong_ways_to_name_the_series(self):
        frame = pd.DataFrame({"a": _series(), "b": _series(seed=1)})
        with pytest.raises(MethodIncompatibility, match="pass data="):
            sp.ets("a")
        with pytest.raises(MethodIncompatibility, match="name of one of its columns"):
            sp.ets("c", data=frame)
        with pytest.raises(MethodIncompatibility, match="2 columns"):
            sp.ets(frame)
        one = sp.ets(frame[["a"]], "ANN")
        assert one.components().columns[0] == "a"

    def test_forecast_arguments(self):
        fit = sp.ets(_series(), "ANN")
        with pytest.raises(MethodIncompatibility, match="horizon must be at least 1"):
            fit.forecast(0)
        with pytest.raises(MethodIncompatibility, match="coverage"):
            fit.forecast(1, level=150)
        # 0.9 and 90 are the same request; duplicates collapse.
        fc = fit.forecast(2, level=(0.9, 90, 50))
        assert list(fc.columns) == [
            "forecast",
            "lower_50",
            "upper_50",
            "lower_90",
            "upper_90",
        ]

    def test_summary_and_plot(self):
        fit = sp.ets(_series(), "AAA", period=4, alpha=0.2)
        text = fit.summary()
        assert "ETS(A,A,A)  (period 4)" in text
        assert "(fixed)" in text
        dated = pd.Series(
            _series(), index=pd.period_range("2010Q1", periods=48, freq="Q")
        )
        ax = sp.ets(dated, "AAN").plot(horizon=4)
        try:
            assert len(ax.get_lines()[0].get_xdata()) == 48
            comp = sp.ets(dated, "AAN").components()
            assert list(comp.columns) == ["y", "level", "slope", "remainder"]
        finally:
            plt.close(ax.figure)
        ax = fit.plot(horizon=4, level=80)
        try:
            lines = {ln.get_label(): ln for ln in ax.get_lines()}
            np.testing.assert_allclose(
                lines["forecast"].get_ydata(), fit.forecast(4)["forecast"]
            )
            np.testing.assert_allclose(lines["fitted"].get_ydata(), fit.fitted_values)
        finally:
            plt.close(ax.figure)


# ----------------------------------------------------------------------
# benchmark methods
# ----------------------------------------------------------------------
class TestSimpleForecast:
    def test_mean(self):
        y = _series()
        n = len(y)
        fit = sp.simple_forecast(y, "mean")
        sigma = np.sqrt(np.sum((y - y.mean()) ** 2) / (n - 1))
        assert fit.sigma == pytest.approx(sigma)
        fc = fit.forecast(3, level=95)
        np.testing.assert_allclose(fc["forecast"], y.mean())
        np.testing.assert_allclose(
            fc["upper_95"] - fc["forecast"],
            stats.norm.ppf(0.975) * sigma * np.sqrt(1 + 1 / n),
        )

    def test_naive(self):
        y = _series()
        n = len(y)
        fit = sp.simple_forecast(y, "naive")
        assert np.isnan(fit.fitted_values[0])
        np.testing.assert_allclose(fit.fitted_values[1:], y[:-1])
        sigma = np.sqrt(np.sum(np.diff(y) ** 2) / (n - 1))
        assert fit.sigma == pytest.approx(sigma)
        fc = fit.forecast(4, level=80)
        np.testing.assert_allclose(fc["forecast"], y[-1])
        np.testing.assert_allclose(
            fc["forecast"] - fc["lower_80"],
            stats.norm.ppf(0.9) * sigma * np.sqrt(np.arange(1, 5)),
        )

    def test_drift(self):
        y = _series()
        n = len(y)
        fit = sp.simple_forecast(y, "drift")
        b = (y[-1] - y[0]) / (n - 1)
        assert fit.drift == pytest.approx(b)
        sigma = np.sqrt(np.sum((np.diff(y) - b) ** 2) / (n - 2))
        assert fit.sigma == pytest.approx(sigma)
        h = np.arange(1, 6)
        fc = fit.forecast(5, level=95)
        np.testing.assert_allclose(fc["forecast"], y[-1] + h * b)
        np.testing.assert_allclose(
            fc["upper_95"] - fc["forecast"],
            stats.norm.ppf(0.975) * sigma * np.sqrt(h * (1 + h / (n - 1))),
        )

    def test_seasonal_naive(self):
        y = _series()
        n = len(y)
        fit = sp.simple_forecast(y, "snaive", period=4)
        assert np.isnan(fit.fitted_values[:4]).all()
        np.testing.assert_allclose(fit.fitted_values[4:], y[:-4])
        sigma = np.sqrt(np.sum((y[4:] - y[:-4]) ** 2) / (n - 4))
        assert fit.sigma == pytest.approx(sigma)
        fc = fit.forecast(10, level=95)
        np.testing.assert_allclose(fc["forecast"], np.tile(y[-4:], 3)[:10])
        np.testing.assert_allclose(
            fc["upper_95"] - fc["forecast"],
            stats.norm.ppf(0.975) * sigma * np.sqrt(np.arange(10) // 4 + 1),
        )

    @pytest.mark.parametrize("method", ["mean", "naive", "drift", "snaive"])
    def test_bootstrap_intervals_from_resampled_paths(self, method):
        y = _series()
        fit = sp.simple_forecast(y, method, period=4)
        fc = fit.forecast(6, level=80, bootstrap=True, n_paths=150, seed=4)
        res = fit.residuals[np.isfinite(fit.residuals)]
        innov = np.random.default_rng(4).choice(
            res - res.mean(), size=(150, 6), replace=True
        )
        if method == "mean":
            paths = y.mean() + innov
        elif method == "naive":
            paths = y[-1] + np.cumsum(innov, axis=1)
        elif method == "drift":
            paths = y[-1] + np.arange(1, 7) * fit.drift + np.cumsum(innov, axis=1)
        else:
            paths = np.empty_like(innov)
            for k in range(6):
                prev = y[-4 + k] if k < 4 else paths[:, k - 4]
                paths[:, k] = prev + innov[:, k]
        np.testing.assert_allclose(
            fc["lower_80"], np.quantile(paths, 0.1, axis=0, method="median_unbiased")
        )
        np.testing.assert_allclose(
            fc["upper_80"], np.quantile(paths, 0.9, axis=0, method="median_unbiased")
        )
        # Point forecasts are not simulated.
        np.testing.assert_allclose(fc["forecast"], fit.forecast(6)["forecast"])
        sim = fit.simulate(6, n_paths=150, seed=4)
        np.testing.assert_allclose(sim.to_numpy(), paths.T)

    def test_simulate_with_normal_innovations(self):
        y = _series()
        fit = sp.simple_forecast(y, "naive")
        sim = fit.simulate(3, n_paths=4, bootstrap=False, seed=8)
        innov = np.random.default_rng(8).normal(0.0, fit.sigma, size=(4, 3))
        np.testing.assert_allclose(sim.to_numpy(), (y[-1] + np.cumsum(innov, 1)).T)

    def test_boxcox_and_bias_adjustment(self):
        y = np.exp(_series() / 10.0)
        inner = sp.simple_forecast(np.log(y), "drift")
        fit = sp.simple_forecast(y, "drift", boxcox=0)
        np.testing.assert_allclose(
            fit.forecast(4).to_numpy(), np.exp(inner.forecast(4).to_numpy())
        )
        np.testing.assert_allclose(
            fit.fitted_values[1:], np.exp(inner.fitted_values[1:])
        )
        np.testing.assert_allclose(fit.residuals[1:], inner.residuals[1:])
        adj = sp.simple_forecast(y, "drift", boxcox=0, biasadj=True).forecast(
            4, level=95
        )
        h = np.arange(1, 5)
        var = inner.sigma**2 * h * (1 + h / (len(y) - 1))
        np.testing.assert_allclose(
            adj["forecast"], np.exp(inner.forecast(4)["forecast"]) * (1 + var / 2)
        )
        sim = fit.simulate(2, n_paths=3, seed=1)
        np.testing.assert_allclose(
            sim.to_numpy(), np.exp(inner.simulate(2, n_paths=3, seed=1).to_numpy())
        )

    def test_aliases_and_refusals(self):
        y = _series()
        a = sp.simple_forecast(y, "Random-Walk")
        assert a.method == "naive"
        with pytest.raises(MethodIncompatibility, match="not a benchmark method"):
            sp.simple_forecast(y, "holt")
        with pytest.raises(MethodIncompatibility, match="period >= 2"):
            sp.simple_forecast(y, "snaive")
        with pytest.raises(DataInsufficient, match="more than one cycle"):
            sp.simple_forecast(y[:4], "snaive", period=4)
        with pytest.raises(DataInsufficient, match="at least 3"):
            sp.simple_forecast(y[:2], "drift")
        with pytest.raises(DataInsufficient, match="at least 2"):
            sp.simple_forecast(y[:1], "naive")
        with pytest.raises(DataInsufficient, match="at least 2"):
            sp.simple_forecast(y[:1], "mean")
        with pytest.raises(MethodIncompatibility, match="missing value"):
            sp.simple_forecast(np.r_[y[:5], np.nan, y[6:]], "naive")
        with pytest.raises(MethodIncompatibility, match="horizon must be at least 1"):
            a.forecast(0)

    def test_summary(self):
        y = _series()
        assert "Drift method" in sp.simple_forecast(y, "drift").summary()
        assert "period 4" in sp.simple_forecast(y, "snaive", period=4).summary()
        assert "mean       :" in repr(sp.simple_forecast(y, "mean"))
