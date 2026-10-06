"""Markov-switching regression: known truths, invariances and errors."""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries import _mswitch_core as core
from statspai.timeseries.mswitch import MarkovSwitchingResult, mswitch

P_TRUE = np.array([[0.95, 0.05], [0.10, 0.90]])


def _simulate(n: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = rng.choice(2, p=P_TRUE[s[t - 1]])
    z = rng.normal(size=n)
    u = np.zeros(n)
    e = rng.normal(scale=0.7, size=n)
    for t in range(1, n):
        u[t] = 0.5 * u[t - 1] + e[t]
    df = pd.DataFrame(
        {
            "y": np.array([-1.0, 2.0])[s] + rng.normal(size=n),
            "yv": np.array([0.5, 2.0])[s] * rng.normal(size=n),
            "yz": np.array([-1.0, 1.5])[s] * z + 0.8 * rng.normal(size=n),
            "yar": np.array([0.0, 3.0])[s] + u,
            "z": z,
        }
    )
    return df, s


@pytest.fixture(scope="module")
def sim() -> Tuple[pd.DataFrame, np.ndarray]:
    return _simulate(600, 11)


@pytest.fixture(scope="module")
def fit_mean(sim: Tuple[pd.DataFrame, np.ndarray]) -> MarkovSwitchingResult:
    return mswitch("y", data=sim[0], seed=0)


def _within(est: float, se: float, truth: float, k: float = 3.5) -> bool:
    return abs(est - truth) < k * se


def test_recovers_means_and_transitions(
    sim: Tuple[pd.DataFrame, np.ndarray], fit_mean: MarkovSwitchingResult
) -> None:
    fit = fit_mean
    assert fit.converged and fit.states == 2 and fit.n_obs == 600
    const = fit.params[fit.params["term"] == "const"]
    for row, truth in zip(const.itertuples(), (-1.0, 2.0)):
        assert _within(row.estimate, row.se, truth)
    tt = fit.transition_table.set_index(["from", "to"])
    for (i, j), truth in np.ndenumerate(P_TRUE):
        row = tt.loc[(f"state{i + 1}", f"state{j + 1}")]
        assert _within(row["estimate"], row["se"], truth)
        assert row["ci_lower"] < row["estimate"] < row["ci_upper"]
    sigma = fit.params[fit.params["term"] == "sigma"].iloc[0]
    assert _within(sigma["estimate"], sigma["se"], 1.0)
    np.testing.assert_allclose(fit.transition.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        fit.durations["estimate"], 1.0 / (1.0 - np.diag(fit.transition))
    )


def test_smoothed_probabilities_classify_the_states(
    sim: Tuple[pd.DataFrame, np.ndarray], fit_mean: MarkovSwitchingResult
) -> None:
    s = sim[1]
    hit_smooth = ((fit_mean.smoothed["state2"] > 0.5).to_numpy() == (s == 1)).mean()
    hit_filter = ((fit_mean.filtered["state2"] > 0.5).to_numpy() == (s == 1)).mean()
    assert hit_smooth > 0.95
    assert hit_smooth >= hit_filter
    for prob in (fit_mean.smoothed, fit_mean.filtered, fit_mean.predicted):
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-12)
        assert prob.to_numpy().min() >= 0.0
    # the last smoothed probability is the last filtered one
    np.testing.assert_allclose(
        fit_mean.smoothed.iloc[-1], fit_mean.filtered.iloc[-1], atol=1e-14
    )


def test_recovers_switching_variance_slope_and_ar(
    sim: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    df, s = sim
    fv = mswitch("yv", data=df, switch_variance=True, constant="common", seed=0)
    assert fv.model_info["order_rule"] == "increasing standard deviation"
    sig = fv.params[fv.params["term"] == "sigma"]
    for row, truth in zip(sig.itertuples(), (0.5, 2.0)):
        assert _within(np.log(row.estimate), row.se / row.estimate, np.log(truth))
    assert ((fv.smoothed["state2"] > 0.5).to_numpy() == (s == 1)).mean() > 0.85

    fz = mswitch("yz", data=df, switch="z", constant="common", seed=0)
    assert fz.model_info["order_rule"].startswith("increasing first switching")
    slope = fz.params[fz.params["term"] == "z"]
    for row, truth in zip(slope.itertuples(), (-1.0, 1.5)):
        assert _within(row.estimate, row.se, truth)

    fa = mswitch("yar", data=df, model="ar", ar=1, seed=0)
    assert fa.n_obs == 599
    par = fa.params.set_index(["state", "term"])
    assert _within(*par.loc[("all", "ar.L1"), ["estimate", "se"]], 0.5)
    assert _within(*par.loc[("state1", "const"), ["estimate", "se"]], 0.0)
    assert _within(*par.loc[("state2", "const"), ["estimate", "se"]], 3.0)


def test_score_is_zero_and_hessian_matches_finite_differences(
    sim: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    df = sim[0].iloc[:250]
    fit = mswitch("yar", data=df, model="ar", ar=2, switch_variance=True, seed=0)
    spec = fit._state["spec"]
    none = np.empty((len(df), 0))
    dat = core.Data(df["yar"].to_numpy(), none, none, 2)
    theta = fit.theta.to_numpy()
    assert core.loglik(spec, dat, theta) == pytest.approx(fit.loglik, rel=1e-13)
    grad = core.gradient(spec, dat, theta)
    assert np.abs(grad).max() < 1e-4
    # complex-step score against a central difference of the likelihood
    point = theta + 0.01
    num = np.empty(len(theta))
    for i in range(len(theta)):
        h = np.zeros(len(theta))
        h[i] = 1e-5
        up, down = core.loglik(spec, dat, point + h), core.loglik(spec, dat, point - h)
        num[i] = (up - down) / 2e-5
    np.testing.assert_allclose(core.gradient(spec, dat, point), num, atol=1e-5)
    hess = core.hessian(spec, dat, theta)
    np.testing.assert_allclose(np.linalg.inv(-hess), fit.vcov.to_numpy(), rtol=1e-8)
    assert fit.aic == pytest.approx(-2 * fit.loglik + 2 * fit.n_params)
    assert fit.bic == pytest.approx(-2 * fit.loglik + fit.n_params * np.log(fit.n_obs))


def test_dr_and_ar_coincide_without_lags_and_differ_with(
    sim: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    df = sim[0].iloc[:300]
    a = mswitch("y", data=df, model="dr", seed=0)
    b = mswitch("y", data=df, model="ar", seed=0)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
    c = mswitch("yar", data=df, model="dr", ar=1, seed=0)
    d = mswitch("yar", data=df, model="ar", ar=1, seed=0)
    assert abs(c.loglik - d.loglik) > 1.0


def test_label_order_and_inputs_do_not_matter(
    sim: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    df = sim[0].iloc[:300]
    a = mswitch("y", data=df, seed=0)
    b = mswitch(df["y"].to_numpy(), seed=123, starts=3)
    c = mswitch(-df["y"], seed=0)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-8)
    np.testing.assert_allclose(a.theta, b.theta, atol=1e-5)
    const = a.params.loc[a.params["term"] == "const", "estimate"].to_numpy()
    assert const[0] < const[1]
    # mirrored data: the states swap, and are relabelled by the constant
    mirrored = c.params.loc[c.params["term"] == "const", "estimate"].to_numpy()
    np.testing.assert_allclose(mirrored, -const[::-1], atol=1e-5)
    np.testing.assert_allclose(
        c.transition.to_numpy(), a.transition.to_numpy()[::-1, ::-1], atol=1e-5
    )
    np.testing.assert_allclose(c.smoothed["state1"], a.smoothed["state2"], atol=1e-5)
    assert a.model_info["order_rule"] == "increasing constant"
    assert list(a.filtered.index) == list(df.index)


def test_robust_vce_and_alpha(sim: Tuple[pd.DataFrame, np.ndarray]) -> None:
    df = sim[0].iloc[:300]
    a = mswitch("y", data=df, seed=0)
    r = mswitch("y", data=df, seed=0, vce="robust", alpha=0.1)
    np.testing.assert_allclose(a.theta, r.theta, atol=1e-10)
    ratio = r.params["se"] / a.params["se"]
    assert ((ratio > 0.6) & (ratio < 1.6)).all() and not np.allclose(ratio, 1.0)
    width_a = a.params["ci_upper"] - a.params["ci_lower"]
    assert (width_a.iloc[:2] / a.params["se"].iloc[:2]).round(3).eq(3.92).all()
    width_r = r.params["ci_upper"] - r.params["ci_lower"]
    assert (width_r.iloc[:2] / r.params["se"].iloc[:2]).round(3).eq(3.29).all()


def test_forecast(sim: Tuple[pd.DataFrame, np.ndarray]) -> None:
    df = sim[0].iloc[:300]
    a = mswitch("y", data=df, seed=0)
    f = a.forecast(60)
    mu = a.params.loc[a.params["term"] == "const", "estimate"].to_numpy()
    trans = a.transition.to_numpy()
    last = a.filtered.iloc[-1].to_numpy()
    np.testing.assert_allclose(f.loc[0, "forecast"], last @ trans @ mu, rtol=1e-12)
    np.testing.assert_allclose(
        f.loc[1, ["state1", "state2"]].to_numpy(dtype=float), last @ trans @ trans
    )
    ergodic = np.array(a.model_info["ergodic"])
    assert f.loc[59, "forecast"] == pytest.approx(ergodic @ mu, abs=1e-3)

    b = mswitch("yar", data=df, model="ar", ar=2, seed=0)
    fb = b.forecast(2)
    st = b._state
    spec = st["spec"]
    dig = spec.digits()
    y = df["yar"].to_numpy()
    # one step ahead by enumeration of (s_{T+1}, s_T, s_{T-1})
    phi, mub, pb = st["phi"][0], st["mu"], st["P"]
    total = 0.0
    for j, w in enumerate(st["filt_exp"][-1]):
        s0, s1 = dig[j, 0], dig[j, 1]
        for nxt in range(2):
            mean = mub[nxt] + phi[0] * (y[-1] - mub[s0]) + phi[1] * (y[-2] - mub[s1])
            total += w * pb[s0, nxt] * mean
    assert fb.loc[0, "forecast"] == pytest.approx(total, rel=1e-12)
    assert b.forecast(200).loc[199, "forecast"] == pytest.approx(
        np.array(b.model_info["ergodic"]) @ mub, abs=1e-6
    )
    with pytest.raises(MethodIncompatibility):
        mswitch("yz", data=df, switch="z", seed=0, starts=1).forecast(1)
    with pytest.raises(MethodIncompatibility):
        a.forecast(0)


def test_fitted_are_one_step_predictions(fit_mean: MarkovSwitchingResult) -> None:
    mu = fit_mean.params.loc[fit_mean.params["term"] == "const", "estimate"]
    expect = fit_mean.predicted.to_numpy() @ mu.to_numpy()
    np.testing.assert_allclose(fit_mean.fitted, expect, atol=1e-12)
    np.testing.assert_allclose(
        fit_mean.model_info["ergodic"], fit_mean.predicted.iloc[0]
    )


def test_summary_dict_and_plot(fit_mean: MarkovSwitchingResult) -> None:
    text = fit_mean.summary()
    assert "Markov-switching dynamic regression" in text
    assert "ordered by increasing constant" in text
    out = fit_mean.to_dict()
    assert out["states"] == 2 and len(out["transition"]) == 2
    assert len(fit_mean.starts) == 5 and fit_mean.model_info["n_distinct_maxima"] >= 1
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib

    matplotlib.use("Agg")
    fig = fit_mean.plot()
    assert len(fig.axes) == 2
    plt.close(fig)
    with pytest.raises(MethodIncompatibility):
        fit_mean.plot("nonsense")


def test_single_regime_data_is_reported_not_hidden() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = mswitch(y, seed=0)
    # either a (weak) two-state maximum with finite numbers, or a loud note
    assert np.isfinite(fit.loglik)
    if not fit.converged or fit.vcov.isna().to_numpy().any():
        assert caught and fit.model_info["notes"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(model="tar"),
        dict(vce="hc3"),
        dict(states=1),
        dict(ar=-1),
        dict(constant="yes"),
        dict(switch_ar=True),
        dict(constant=False),
        dict(model="ar", ar=5),
        dict(model="ar", ar=1, states=4),
        dict(alpha=1.5),
        dict(starts=0),
        dict(x="missing"),
    ],
)
def test_bad_arguments(kwargs: dict, sim: Tuple[pd.DataFrame, np.ndarray]) -> None:
    with pytest.raises(MethodIncompatibility):
        mswitch("y", data=sim[0], **kwargs)


def test_bad_data(sim: Tuple[pd.DataFrame, np.ndarray]) -> None:
    df = sim[0]
    with pytest.raises(MethodIncompatibility):
        mswitch("y")
    with pytest.raises(MethodIncompatibility):
        mswitch("nope", data=df)
    hole = df["y"].copy()
    hole.iloc[10] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing"):
        mswitch(hole)
    with pytest.raises(MethodIncompatibility, match="length"):
        mswitch(df["y"].to_numpy(), x=np.ones(10))
    with pytest.raises(DataInsufficient):
        mswitch(df["y"].to_numpy()[:15])
