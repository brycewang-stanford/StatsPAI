"""Correctness and edge cases of ``tvp_var``.

Cross-implementation numbers are in
``tests/reference_parity/test_tvp_var_parity.py``; this file holds the
known-truth recoveries, the derived quantities (impulse responses,
roots, forecasts) and the argument checks.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility, StatsPAIWarning
from statspai.timeseries.tvp_var import TVPVARResult, tvp_var


def _simulate(
    seed: int, T: int = 200, moving: bool = True
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bivariate VAR(1); the own lag of ``x`` goes smoothly 0.2 -> 0.8."""
    rng = np.random.default_rng(seed)
    grid = np.arange(T + 1)
    a = 0.2 + 0.6 / (1.0 + np.exp(-(grid - T / 2) / (T / 10)))
    if not moving:
        a = np.full(T + 1, 0.5)
    y = np.zeros((T + 1, 2))
    for t in range(1, T + 1):
        y[t, 0] = a[t] * y[t - 1, 0] + 0.2 * y[t - 1, 1] + rng.normal()
        y[t, 1] = 0.1 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + 0.5 + rng.normal()
    return pd.DataFrame(y, columns=["x", "z"]), a[1:]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _simulate(0)[0]


@pytest.fixture(scope="module")
def forgetting(data: pd.DataFrame) -> TVPVARResult:
    return tvp_var(data, lags=2, method="forgetting", lam=0.98, kappa=0.95)


# --------------------------------------------------------------------- #
#  known truth
# --------------------------------------------------------------------- #


def test_smoothed_path_tracks_a_moving_coefficient() -> None:
    """Monte Carlo, 12 samples of 200 dates, 90% pointwise bands.

    Over 30 samples the root mean squared error of the path averaged
    0.107 (worst sample 0.25) and the bands covered the truth at 90.6% of
    the dates on average. A sample in which the variance is estimated at
    zero has a flat path and poor coverage, so the bounds are on the
    averages: 0.16 for the error, and coverage between 0.75 and 0.99.
    """
    rmse, cover, w_moving, w_fixed = [], [], [], []
    for seed in range(12):
        df, truth = _simulate(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StatsPAIWarning)
            fit = tvp_var(df, lags=1, alpha=0.10)
        tab = fit.coefficients()
        own = tab[(tab["equation"] == "x") & (tab["term"] == "L1.x")]
        est = own["coef"].to_numpy()
        np.testing.assert_array_equal(est, fit.coef_smoothed[:, 0, 0])
        rmse.append(np.sqrt(np.mean((est - truth) ** 2)))
        inside = (own["lower"].to_numpy() <= truth) & (truth <= own["upper"].to_numpy())
        cover.append(inside.mean())
        W = fit.variances[fit.terms].to_numpy()
        w_moving.append(W[0, 0])
        w_fixed.append(np.delete(W.ravel(), 0))
    assert np.mean(rmse) < 0.16
    assert 0.75 < np.mean(cover) < 0.99
    # the five constant coefficients: innovation variance at zero in the
    # typical sample; the moving one is orders of magnitude above
    assert np.median(np.array(w_fixed)) < 1e-8
    assert np.median(w_moving) > 1e-4


def test_zero_state_variance_is_ols_at_every_date(data: pd.DataFrame) -> None:
    fit = tvp_var(data, lags=2, state_var=0.0)
    v = sp.var(data, lags=2)
    for i, name in enumerate(fit.var_names):
        ols = v.coefs[name]["coef"].to_numpy()
        # constant coefficients under a diffuse prior: the smoother returns
        # the full-sample least squares fit at every date (prior ~ 1e-9)
        np.testing.assert_allclose(
            fit.coef_smoothed[:, i], np.tile(ols, (fit.n_obs, 1)), atol=1e-7
        )
        np.testing.assert_allclose(fit.coef_filtered[-1, i], ols, atol=1e-7)
        # and the error variance is the ML one, RSS / T, up to the k
        # diffuse dates: (T - k) / T of sigma_u is the REML-type value
        k = len(fit.terms)
        want = v.sigma_u.to_numpy()[i, i] * fit.n_obs / (fit.n_obs - k)
        assert fit.variances.loc[name, "obs"] == pytest.approx(want, rel=1e-4)


def test_constant_coefficients_are_reported_as_constant() -> None:
    """Constant-coefficient data, a sample where the estimate is on the
    boundary: variances at zero, a warning, and the OLS coefficients."""
    df, _ = _simulate(102, moving=False)
    with pytest.warns(StatsPAIWarning, match="estimated at zero"):
        fit = tvp_var(df, lags=1, common=True)
    assert len(fit.model_info["constant_coefficients"]) == 6
    assert "constant" in fit.summary()
    assert fit.variances[fit.terms].to_numpy().max() < 1e-10
    v = sp.var(df, lags=1)
    for i, name in enumerate(fit.var_names):
        # variance ~ 1e-14 instead of 0: paths flat to about 1e-6
        np.testing.assert_allclose(
            fit.coef_smoothed[-1, i], v.coefs[name]["coef"].to_numpy(), atol=1e-5
        )
    assert np.ptp(fit.coef_smoothed, axis=0).max() < 1e-5


def test_constant_data_give_small_variances_in_every_sample() -> None:
    """The likelihood does not always put the variance exactly at zero
    (it did in 3 of 10 samples with ``common=True``); it is always small
    next to an error variance of one, and the time average of each path
    stays near least squares."""
    for seed in range(100, 106):
        df, _ = _simulate(seed, moving=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StatsPAIWarning)
            fit = tvp_var(df, lags=1, common=True)
        assert fit.variances[fit.terms].to_numpy().max() < 2e-3
        v = sp.var(df, lags=1)
        ols = np.array([v.coefs[n]["coef"].to_numpy() for n in fit.var_names])
        assert np.abs(fit.coef_smoothed.mean(axis=0) - ols).max() < 0.1


def test_common_variance_is_shared_within_an_equation(data: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatsPAIWarning)
        fit = tvp_var(data, lags=1, common=True)
        free = tvp_var(data, lags=1)
    W = fit.variances[fit.terms].to_numpy()
    np.testing.assert_allclose(W, np.tile(W[:, :1], (1, 3)), rtol=0, atol=0)
    # the restricted likelihood cannot beat the free one
    assert fit.loglik <= free.loglik + 1e-6
    assert fit.model_info["common"] is True


def test_forgetting_tracks_the_moving_coefficient() -> None:
    df, truth = _simulate(3, T=400)
    fit = tvp_var(df, lags=1, method="forgetting", lam=0.97)
    est = fit.coef_filtered[:, 0, 0]
    # a one-sided filter lags the truth; early dates are few-observation fits
    assert est[-50:].mean() - est[60:110].mean() > 0.3
    assert np.sqrt(np.mean((est[60:] - truth[60:]) ** 2)) < 0.25


# --------------------------------------------------------------------- #
#  derived quantities
# --------------------------------------------------------------------- #


def test_irf_at_constant_coefficients_is_sp_irf(data: pd.DataFrame) -> None:
    fit = tvp_var(data, lags=2, method="forgetting", lam=1.0, kappa=1.0)
    v = sp.var(data, lags=2)
    ours = fit.irf(at=-1, periods=8)
    assert list(ours.columns) == ["date", "shock", "response", "period", "irf"]
    assert len(ours) == 2 * 2 * 9
    theirs = sp.irf(v, periods=8, sigma_df="ml")["irf"]
    for (shock, resp), grp in ours.groupby(["shock", "response"]):
        assert list(grp["period"]) == list(range(9))
        # same coefficients (to 1e-9) and the same covariance, divisor T
        np.testing.assert_allclose(
            grp["irf"].to_numpy(), theirs[f"{shock} -> {resp}"], atol=1e-7
        )
    plain = fit.irf(at=-1, periods=3, orthogonal=False)
    A1 = fit.coef_filtered[-1][:, :2]
    A2 = fit.coef_filtered[-1][:, 2:4]
    phi2 = A1 @ A1 + A2
    row = plain[(plain.shock == "z") & (plain.response == "x") & (plain.period == 2)]
    assert row["irf"].item() == pytest.approx(phi2[0, 1], rel=1e-12)
    cum = fit.irf(at=-1, periods=3, orthogonal=False, cumulative=True)
    np.testing.assert_allclose(
        cum.groupby(["shock", "response"], sort=False)["irf"].last().to_numpy(),
        plain.groupby(["shock", "response"], sort=False)["irf"].sum().to_numpy(),
        rtol=1e-12,
    )


def test_irf_dates_by_label_position_and_list(forgetting: TVPVARResult) -> None:
    fit = forgetting
    last = fit.irf(periods=2)
    assert last["date"].unique().tolist() == [fit.index[-1]]
    both = fit.irf(at=[50, -1], periods=2)
    assert both["date"].unique().tolist() == [fit.index[50], fit.index[-1]]
    pd.testing.assert_frame_equal(
        both[both["date"] == fit.index[-1]].reset_index(drop=True), last
    )
    # the covariance of that date is used: impact response = its Cholesky
    first = both[(both.date == fit.index[50]) & (both.period == 0)]
    chol = np.linalg.cholesky(fit.sigma_t[50])
    np.testing.assert_allclose(first["irf"].to_numpy(), chol.T.ravel(), rtol=1e-12)
    with pytest.raises(MethodIncompatibility, match="outside"):
        fit.irf(at=10_000)
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        fit.irf(periods=-1)
    with pytest.raises(MethodIncompatibility, match="smoothed"):
        fit.irf(kind="smoothed")


def test_dates_from_a_time_column() -> None:
    df, _ = _simulate(5, T=80)
    df["date"] = pd.date_range("2000-01-01", periods=len(df), freq="QS")
    shuffled = df.sample(frac=1.0, random_state=0)
    fit = tvp_var(shuffled, ["x", "z"], lags=1, method="forgetting", time="date")
    ordered = tvp_var(df, ["x", "z"], lags=1, method="forgetting")
    np.testing.assert_array_equal(fit.coef_filtered, ordered.coef_filtered)
    assert fit.index[0] == pd.Timestamp("2000-04-01")
    one = fit.irf(at="2010-01-01", periods=1)
    assert one["date"].iloc[0] == pd.Timestamp("2010-01-01")
    with pytest.raises(MethodIncompatibility, match="not among"):
        fit.irf(at="1990-01-01")


def test_stability_is_the_companion_root(forgetting: TVPVARResult) -> None:
    fit = forgetting
    stab = fit.stability()
    assert list(stab.columns) == ["max_root", "explosive"]
    assert stab.index.equals(fit.index)
    for pos in (30, 120, len(stab) - 1):
        B = fit.coef_filtered[pos]
        comp = np.zeros((4, 4))
        comp[:2] = B[:, :4]
        comp[2:, :2] = np.eye(2)
        want = np.abs(np.linalg.eigvals(comp)).max()
        assert stab["max_root"].iloc[pos] == pytest.approx(want, rel=1e-12)
    assert stab.attrs["n_explosive"] == int(stab["explosive"].sum())
    # the first dates fit the diffuse prior exactly and are wild
    assert (stab["explosive"] == (stab["max_root"] >= 1.0)).all()


def test_explosive_dates_are_flagged() -> None:
    rng = np.random.default_rng(7)
    y = np.zeros((120, 1))
    for t in range(1, 120):
        rho = 0.5 if t < 60 else 1.1
        y[t] = rho * y[t - 1] + rng.normal()
    fit = tvp_var(pd.DataFrame(y, columns=["y"]), lags=1, method="forgetting", lam=0.9)
    stab = fit.stability()
    assert stab["explosive"].iloc[-10:].all()
    assert not stab["explosive"].iloc[40:58].any()
    assert "explosive at" in fit.summary()


def test_forecast_iterates_the_last_filtered_var(
    data: pd.DataFrame, forgetting: TVPVARResult
) -> None:
    fit = forgetting
    fc = fit.forecast(3, alpha=0.10)
    assert list(fc.index) == [1, 2, 3]
    B = fit.coef_filtered[-1]
    d = data.to_numpy()
    x1 = np.r_[d[-1], d[-2], 1.0]
    f1 = B @ x1
    f2 = B @ np.r_[f1, d[-1], 1.0]
    np.testing.assert_allclose(fc.loc[1, ["x", "z"]].to_numpy(float), f1, rtol=1e-12)
    np.testing.assert_allclose(fc.loc[2, ["x", "z"]].to_numpy(float), f2, rtol=1e-12)
    S = fit.sigma_t[-1]
    assert fc.loc[1, "x_se"] == pytest.approx(np.sqrt(S[0, 0]), rel=1e-12)
    A1 = B[:, :2]
    mse2 = S + A1 @ S @ A1.T
    assert fc.loc[2, "z_se"] == pytest.approx(np.sqrt(mse2[1, 1]), rel=1e-12)
    assert fc.loc[1, "x_upper"] - fc.loc[1, "x"] == pytest.approx(
        1.6448536269514722 * fc.loc[1, "x_se"], rel=1e-10
    )
    with pytest.raises(MethodIncompatibility):
        fit.forecast(0)


def test_kalman_sigma_and_forecast(data: pd.DataFrame) -> None:
    fit = tvp_var(data, lags=1, obs_var=[1.0, 1.2], state_var=0.001)
    S = fit.sigma.to_numpy()
    # correlations of the standardised prediction errors, variances the
    # median forecast variance, both after the k = 3 diffuse dates
    err, var = fit.model_info["pred_error"][3:], fit.model_info["pred_var"][3:]
    d = data.to_numpy()
    X = np.column_stack([d[:-1], np.ones(len(d) - 1)])
    rebuilt = d[2:] - np.einsum("tik,tk->ti", fit.coef_filtered[:-1], X[1:])
    np.testing.assert_allclose(err[1:], rebuilt[3:], rtol=1e-9, atol=1e-12)
    corr = np.corrcoef(err / np.sqrt(var), rowvar=False)
    sd = np.sqrt(np.median(var, axis=0))
    np.testing.assert_allclose(S, corr * np.outer(sd, sd), rtol=1e-12)
    assert (np.diag(S) > [1.0, 1.2]).all()  # obs_var plus coefficient risk
    assert fit.sigma_t is None
    assert np.linalg.eigvalsh(S).min() > 0
    fc = fit.forecast(1)
    want = fit.coef_filtered[-1] @ np.r_[data.to_numpy()[-1], 1.0]
    np.testing.assert_allclose(fc.loc[1, ["x", "z"]].to_numpy(float), want)
    # smoothing cannot raise the variance, and the last date is unsmoothed
    assert (fit.se_smoothed <= fit.se_filtered * (1 + 1e-9)).all()
    np.testing.assert_allclose(fit.coef_smoothed[-1], fit.coef_filtered[-1])


def test_minnesota_prior_shrinks_towards_its_mean(data: pd.DataFrame) -> None:
    tight = tvp_var(
        data,
        lags=2,
        method="forgetting",
        lam=1.0,
        kappa=1.0,
        prior="minnesota",
        prior_tightness=1e-10,
        prior_own_lag=0.9,
    )
    B = tight.coef_filtered[-1]
    np.testing.assert_allclose(B[:, :2], 0.9 * np.eye(2), atol=1e-5)
    np.testing.assert_allclose(B[:, 2:4], 0.0, atol=1e-5)
    # the intercept keeps the diffuse variance: it is the mean of the
    # quasi-difference y_t - 0.9 y_{t-1}
    d = data.to_numpy()
    np.testing.assert_allclose(B[:, 4], (d[2:] - 0.9 * d[1:-1]).mean(axis=0), atol=1e-4)


# --------------------------------------------------------------------- #
#  result protocol
# --------------------------------------------------------------------- #


def test_tidy_table_summary_and_dict(data: pd.DataFrame) -> None:
    fit = tvp_var(data, lags=1, obs_var=[1.0, 1.0], state_var=0.0005)
    tab = fit.coefficients(alpha=0.32)
    assert list(tab.columns) == [
        "date",
        "equation",
        "term",
        "coef",
        "se",
        "lower",
        "upper",
    ]
    assert len(tab) == fit.n_obs * 2 * 3
    row = tab[
        (tab.date == fit.index[17]) & (tab.equation == "z") & (tab.term == "L1.x")
    ]
    assert row["coef"].item() == fit.coef_smoothed[17, 1, 0]
    assert row["se"].item() == fit.se_smoothed[17, 1, 0]
    filt = fit.coefficients(kind="filtered")
    assert filt["coef"].iloc[-1] == fit.coef_filtered[-1, 1, 2]
    text = fit.summary()
    assert "Time-varying-parameter VAR (kalman)" in text and "L1.x" in text
    out = fit.to_dict()
    assert out["method"] == "kalman" and out["lags"] == 1
    assert set(out["variances"]) == {"x", "z"}
    assert fit.cite() is not None
    with pytest.raises(MethodIncompatibility):
        fit.coefficients(kind="other")
    with pytest.raises(MethodIncompatibility):
        fit.coefficients(alpha=1.5)


def test_plot(forgetting: TVPVARResult) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    fig = forgetting.plot(equation="x", terms=["L1.x", "_cons"])
    assert len(fig.axes) == 2
    with pytest.raises(MethodIncompatibility):
        forgetting.plot(equation="nope")
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------- #
#  argument checks
# --------------------------------------------------------------------- #


def test_too_few_dates() -> None:
    df, _ = _simulate(1, T=12)
    with pytest.raises(DataInsufficient, match="time-varying coefficients"):
        tvp_var(df.iloc[:9], lags=2)
    with pytest.raises(DataInsufficient):
        tvp_var(df.iloc[:9], lags=2, method="forgetting")
    assert tvp_var(df.iloc[:10], lags=2, method="forgetting").n_obs == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "mcmc"},
        {"lags": 0},
        {"alpha": 0.0},
        {"variables": ["x", "nope"]},
        {"variables": ["x", "x"]},
        {"time": "nope"},
        {"prior": "flat"},
        {"C0": -1.0},
        {"C0": [1.0, 2.0]},
        {"m0": [1.0, 2.0]},
        {"prior": "minnesota", "prior_tightness": 0.0},
        {"obs_var": [1.0, -1.0]},
        {"state_var": [0.1, 0.1]},
        {"state_var": -0.1},
        {"state_var": 0.1, "common": True},
        {"method": "forgetting", "lam": 0.0},
        {"method": "forgetting", "lam": 1.2},
        {"method": "forgetting", "kappa": 0.0},
        {"method": "forgetting", "common": True},
        {"method": "forgetting", "state_var": 0.1},
        {"method": "forgetting", "sigma_update": "smoothed"},
        {"method": "forgetting", "sigma0": np.eye(3)},
        {"method": "forgetting", "sigma0": -np.eye(2)},
    ],
)
def test_bad_arguments(data: pd.DataFrame, kwargs: dict) -> None:
    with pytest.raises(MethodIncompatibility):
        tvp_var(data, **kwargs)


def test_missing_values_and_non_frames(data: pd.DataFrame) -> None:
    bad = data.copy()
    bad.iloc[20, 0] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing"):
        tvp_var(bad, lags=1)
    with pytest.raises(MethodIncompatibility, match="DataFrame"):
        tvp_var(data.to_numpy(), lags=1)  # type: ignore[arg-type]


def test_error_variance_at_zero_is_reported() -> None:
    """A random walk fitted with a random-walk intercept: the likelihood
    gives the whole variation to the intercept."""
    rng = np.random.default_rng(13)
    walk = np.cumsum(rng.normal(size=150))
    noise = rng.normal(size=150)
    df = pd.DataFrame({"w": walk, "e": noise})
    with pytest.warns(StatsPAIWarning, match="error variance of w"):
        fit = tvp_var(df, lags=1)
    assert fit.model_info["degenerate_equations"] == ["w"]
    assert fit.variances.loc["w", "obs"] < 1e-8
    # sigma comes from the prediction errors and stays usable
    S = fit.sigma.to_numpy()
    assert 0.6 < S[0, 0] < 1.6 and np.linalg.eigvalsh(S).min() > 0
    assert fit.irf(periods=2)["irf"].abs().max() > 0.5
