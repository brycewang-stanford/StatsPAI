"""Tests for sp.risk_class_selection (Azevedo, Beauchamp, Karlsson Linner 2026).

Known truths: a DGP that satisfies Assumptions 1-5 of the paper with
normal covariates, so the risk of every scenario has a closed form and the
population ``t80`` can be computed on a large draw; identities (Scenario 3
with ``r2_future = R2c`` is Scenario 2, Scenario 1 inside the class is
``sp.pooled_price_ratio`` on the class's risks, standardizing the
predictor changes nothing); and bootstrap calibration against the Monte
Carlo spread of the estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)

# DGP: w ~ N(0, 1); Gf = 0.3 w + V, V ~ N(0, 0.49); Gc = Gf + eps,
# eps ~ N(0, 0.36); L = 1{-1.5 + 0.5 Gf + 0.6 w + eta > 0}.
B0, BG, BW, TH, SV2, SE2 = -1.5, 0.5, 0.6, 0.3, 0.49, 0.36
VAR_GC = TH**2 + SV2 + SE2  # 0.94
_A = (1 / SE2) / (1 / SE2 + 1 / SV2)
_B = (1 / SV2) / (1 / SE2 + 1 / SV2)
_C2 = 1 / (1 / SE2 + 1 / SV2)


def _r2_future() -> float:
    """Liability R2 of the probit of L on Gf: Cov[L*, Gf]^2 / (Var Gf Var L*)."""
    var_gf = TH**2 + SV2
    cov = BG * var_gf + BW * TH
    var_l = BG**2 * var_gf + BW**2 + 2 * BG * BW * TH + 1
    return cov**2 / (var_gf * var_l)


def _draw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    gf = TH * w + rng.normal(scale=np.sqrt(SV2), size=n)
    gc = gf + rng.normal(scale=np.sqrt(SE2), size=n)
    loss = (B0 + BG * gf + BW * w + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"loss": loss, "w": w, "gc": gc, "gf": gf})


def _true_risks(df: pd.DataFrame) -> dict:
    w, gc, gf = (df[c].to_numpy() for c in ("w", "gc", "gf"))
    r1 = stats.norm.cdf((B0 + (BW + BG * TH) * w) / np.sqrt(1 + BG**2 * SV2))
    mean_gf = _A * gc + _B * TH * w
    r2 = stats.norm.cdf((B0 + BG * mean_gf + BW * w) / np.sqrt(1 + BG**2 * _C2))
    r3 = stats.norm.cdf(B0 + BG * gf + BW * w)
    return {"1": r1, "2": r2, "3": r3}


def _t80(p: np.ndarray) -> float:
    return float(sp.pooled_price_ratio(p).estimate)


@pytest.fixture(scope="module")
def population_t80() -> dict:
    pop = _draw(2_000_000, 99)
    risks = _true_risks(pop)
    r1 = risks["1"]
    inside = (r1 >= 0.75 * r1.mean()) & (r1 <= 1.25 * r1.mean())
    return {s: _t80(r[inside]) for s, r in risks.items()}


@pytest.fixture(scope="module")
def big_fit():
    df = _draw(100_000, 1)
    return sp.risk_class_selection(
        df, "loss", ["w"], pgi="gc", r2_future=_r2_future(), n_boot=0
    )


def test_parameters_recover_the_dgp(big_fit):
    # Standardizing Gc rescales Gf by 1 / sd(Gc): beta_g by sd, theta by
    # 1 / sd, the variances by 1 / Var(Gc); beta_w is unchanged.
    sd = np.sqrt(VAR_GC)
    p = big_fit.params
    f = p["future"]["3"]
    # Tolerances are about 4 sampling SEs at n = 100,000.
    assert p["theta"]["w"] == pytest.approx(TH / sd, abs=0.012)
    assert f["sigma_eps2"] == pytest.approx(SE2 / VAR_GC, abs=0.03)
    assert f["sigma_v2"] == pytest.approx(SV2 / VAR_GC, abs=0.03)
    assert f["beta_g"] == pytest.approx(BG * sd, abs=0.04)
    assert f["beta_w"]["w"] == pytest.approx(BW, abs=0.04)
    assert f["beta_w"]["Intercept"] == pytest.approx(B0, abs=0.04)
    assert p["r2_current"] == pytest.approx(_r2_future() * (1 - SE2 / VAR_GC), abs=0.01)


def test_t80_matches_population_value(big_fit, population_t80):
    # Population t80 from 2e6 draws with the closed-form risks; the
    # tolerances are about 4 sampling SEs of t80 at n = 100,000.
    tab = big_fit.table
    assert tab.loc["1", "t80"] == pytest.approx(population_t80["1"], abs=0.008)
    assert tab.loc["2", "t80"] == pytest.approx(population_t80["2"], abs=0.03)
    assert tab.loc["3", "t80"] == pytest.approx(population_t80["3"], abs=0.03)
    # Selection grows with the consumers' information (paper, Table IV).
    assert tab["t80"].is_monotonic_increasing


def test_scenario1_is_pooled_price_ratio_on_the_class():
    df = _draw(3000, 2)
    res = sp.risk_class_selection(df, "loss", ["w"], n_boot=0)
    r1 = res.risk["risk_scenario1"].to_numpy()
    inside = res.risk["in_class"].to_numpy()
    m = r1.mean()
    np.testing.assert_array_equal(inside, (r1 >= 0.75 * m) & (r1 <= 1.25 * m))
    assert res.table.loc["1", "t80"] == _t80(r1[inside])
    assert res.t_no_underwriting == _t80(r1)
    fit = sp.probit(data=df, y="loss", x=["w"])
    X = np.column_stack([np.ones(len(df)), df["w"]])
    np.testing.assert_allclose(
        r1, stats.norm.cdf(X @ fit.params[["Intercept", "w"]].to_numpy()), rtol=1e-12
    )


def test_scenario2_risks_are_probit_predictions_on_the_raw_predictor():
    df = _draw(3000, 3)
    res = sp.risk_class_selection(df, "loss", ["w"], pgi="gc", n_boot=0)
    fit = sp.probit(data=df, y="loss", x=["gc", "w"])
    X = np.column_stack([np.ones(len(df)), df["gc"], df["w"]])
    r2 = stats.norm.cdf(X @ fit.params[["Intercept", "gc", "w"]].to_numpy())
    r2 = r2[res.risk["in_class"].to_numpy()]
    np.testing.assert_allclose(res.curves["2"]["r"], np.unique(r2), rtol=1e-9)


def test_future_equal_to_current_reproduces_scenario2():
    df = _draw(3000, 4)
    first = sp.risk_class_selection(
        df, "loss", ["w"], pgi="gc", r2_future=0.5, n_boot=0
    )
    r2c = first.params["r2_current"]
    res = sp.risk_class_selection(df, "loss", ["w"], pgi="gc", r2_future=r2c, n_boot=0)
    f = res.params["future"]["3"]
    assert f["sigma_eps2"] == 0.0 and f["c"] == 0.0
    assert f["beta_g"] == res.params["gamma_scenario2"]["gc"]
    pd.testing.assert_series_equal(
        f["beta_w"], res.params["gamma_scenario2"].drop("gc"), check_names=False
    )
    assert res.table.loc["3", "t80"] == pytest.approx(
        res.table.loc["2", "t80"], abs=1e-12
    )
    np.testing.assert_allclose(res.curves["3"]["t"], res.curves["2"]["t"], atol=1e-12)


def test_predictor_scale_does_not_matter():
    df = _draw(3000, 5)
    a = sp.risk_class_selection(df, "loss", ["w"], pgi="gc", r2_future=0.4, n_boot=0)
    df2 = df.assign(gc=5.0 + 3.0 * df["gc"])
    b = sp.risk_class_selection(df2, "loss", ["w"], pgi="gc", r2_future=0.4, n_boot=0)
    np.testing.assert_allclose(a.table["t80"], b.table["t80"], atol=1e-8)


def test_no_underwriting_and_at():
    df = _draw(3000, 6)
    res = sp.risk_class_selection(df, "loss", ["w"], risk_class=None, n_boot=0)
    assert res.table.loc["1", "n_class"] == len(df)
    assert res.table.loc["1", "t80"] == res.t_no_underwriting
    # Predicting everyone at w = 0 gives one risk value: no private risk.
    res0 = sp.risk_class_selection(df, "loss", ["w"], at={"w": 0.0}, n_boot=0)
    assert res0.table.loc["1", "n_class"] == len(df)
    assert res0.table.loc["1", "t80"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_is_reproducible_and_brackets_estimate():
    df = _draw(2000, 7)
    kw = dict(pgi="gc", r2_future={"3L": 0.4}, n_boot=30, seed=11)
    a = sp.risk_class_selection(df, "loss", ["w"], **kw)
    b = sp.risk_class_selection(df, "loss", ["w"], **kw)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert list(a.table.index) == ["1", "2", "3L"]
    assert (a.table["se"] > 0).all()
    assert (a.table["ci_lower"] < a.table["ci_upper"]).all()
    assert a.boot_draws.shape == (30, 3)


@pytest.mark.slow
def test_bootstrap_se_tracks_monte_carlo_spread():
    # 40 Monte Carlo samples of n = 2000: the mean bootstrap SE (49 draws)
    # of the Scenario 2 t80 must be within 35% of the Monte Carlo SD of
    # the estimate (the SD itself is estimated from 40 draws, about 11%
    # relative SE, so 35% is about 3 of its SEs).
    est, ses = [], []
    for r in range(40):
        df = _draw(2000, 1000 + r)
        res = sp.risk_class_selection(df, "loss", ["w"], pgi="gc", n_boot=49, seed=r)
        est.append(res.table.loc["2", "t80"])
        ses.append(res.table.loc["2", "se"])
    mc_sd = np.std(est, ddof=1)
    assert np.mean(ses) == pytest.approx(mc_sd, rel=0.35)


@pytest.mark.slow
def test_bootstrap_interval_coverage(population_t80):
    # 200 samples of n = 3000 with 199 bootstrap draws each; 95% percentile
    # interval coverage of the population t80 of Scenario 2. With 200
    # replications the binomial SE at 0.95 is 0.015. Observed 2026-10-28:
    # coverage 0.94, mean estimate 0.319 against 0.326, Monte Carlo SD 0.0559
    # against a mean bootstrap SE of 0.0541.
    hits = []
    for r in range(200):
        df = _draw(3000, 5000 + r)
        res = sp.risk_class_selection(df, "loss", ["w"], pgi="gc", n_boot=199, seed=r)
        lo, hi = res.table.loc["2", ["ci_lower", "ci_upper"]]
        hits.append(lo <= population_t80["2"] <= hi)
    assert 0.88 <= np.mean(hits) <= 0.99


def test_errors():
    df = _draw(1000, 8)
    with pytest.raises(MethodIncompatibility):
        sp.risk_class_selection(df, "loss", ["w"], r2_future=0.3)
    with pytest.raises(IdentificationFailure):
        sp.risk_class_selection(df, "loss", ["w"], pgi="gc", r2_future=0.001, n_boot=0)
    with pytest.raises(MethodIncompatibility):
        sp.risk_class_selection(df, "loss", ["w"], risk_class=(1.2, 0.8))
    with pytest.raises(MethodIncompatibility):
        sp.risk_class_selection(df, "gc", ["w"])
    with pytest.raises(MethodIncompatibility):
        sp.risk_class_selection(df, "loss", ["w"], at={"gc": 0.0})
    with pytest.raises(MethodIncompatibility):
        sp.risk_class_selection(df, "loss", ["w"], pgi="gc", r2_future={"2": 0.3})
    with pytest.raises(DataInsufficient):
        sp.risk_class_selection(df, "loss", ["w"], risk_class=(0.999, 1.001), n_boot=0)
