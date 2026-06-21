"""Round-4 coverage margin: synth.bsts (CausalImpact / bsts_synth).

Pure-numpy Bayesian structural time series. Covers:
- causal_impact with covariates (regression / ridge path),
- local-linear-trend model (use_trend Kalman branches),
- the att<0 Bayesian p-value branch,
- bsts_synth on a real donor panel,
- the input-validation error branches.

No mocking of numeric paths -- real time series with a genuine level
shift in the post period.
"""

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.synth import bsts as _bsts


def _make_ts(n_pre=40, n_post=20, effect=6.0, seed=0):
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    x1 = np.cumsum(rng.normal(0, 1, T)) + 50.0
    x2 = np.cumsum(rng.normal(0, 1, T)) + 20.0
    y = 0.6 * x1 + 0.3 * x2 + rng.normal(0, 0.5, T)
    y[n_pre:] += effect
    idx = np.arange(T)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=idx)
    return df, n_pre


def test_causal_impact_with_covariates_positive_effect():
    df, n_pre = _make_ts(effect=8.0)
    res = _bsts.causal_impact(
        data=df,
        pre_period=(0, n_pre - 1),
        post_period=(n_pre, len(df) - 1),
        outcome="y",
        covariates=["x1", "x2"],
        n_simulations=200,
        seed=1,
    )
    assert np.isfinite(res.estimate)
    assert res.estimate > 0  # genuine positive intervention
    assert 0.0 <= res.pvalue <= 1.0


def test_causal_impact_negative_effect_branch():
    # Negative effect -> exercises the att<0 p-value branch.
    df, n_pre = _make_ts(effect=-8.0, seed=2)
    res = _bsts.causal_impact(
        data=df,
        pre_period=(0, n_pre - 1),
        post_period=(n_pre, len(df) - 1),
        outcome="y",
        covariates=["x1"],
        n_simulations=200,
        seed=3,
    )
    assert res.estimate < 0
    assert 0.0 <= res.pvalue <= 1.0


def test_causal_impact_local_linear_trend():
    df, n_pre = _make_ts(effect=5.0, seed=4)
    res = _bsts.causal_impact(
        data=df,
        pre_period=(0, n_pre - 1),
        post_period=(n_pre, len(df) - 1),
        outcome="y",
        covariates=["x1", "x2"],
        model="local_linear_trend",
        n_simulations=150,
        seed=5,
    )
    assert np.isfinite(res.estimate)


def test_causal_impact_no_outcome_uses_first_column():
    df, n_pre = _make_ts(effect=7.0, seed=6)
    res = _bsts.causal_impact(
        data=df,
        pre_period=(0, n_pre - 1),
        post_period=(n_pre, len(df) - 1),
        n_simulations=120,
        seed=7,
    )
    assert np.isfinite(res.estimate)


def test_causal_impact_errors():
    df, n_pre = _make_ts()
    with pytest.raises(MethodIncompatibility):
        _bsts.causal_impact(data=[1, 2, 3], pre_period=(0, 1), post_period=(2, 3))
    with pytest.raises(ValueError):
        _bsts.causal_impact(
            data=df,
            pre_period=(0, n_pre - 1),
            post_period=(n_pre, len(df) - 1),
            outcome="nope",
        )
    # Overlapping pre/post -> pre must end before post.
    with pytest.raises(ValueError):
        _bsts.causal_impact(
            data=df,
            pre_period=(0, len(df) - 1),
            post_period=(5, len(df) - 1),
            outcome="y",
        )


def _make_panel(n_donors=5, n_pre=25, n_post=12, seed=0):
    rng = np.random.default_rng(seed)
    T = n_pre + n_post
    years = np.arange(2000, 2000 + T)
    rows = []
    donor_series = {}
    for d in range(n_donors):
        donor_series[d] = np.cumsum(rng.normal(0, 1, T)) + 30 + d
        for i, yr in enumerate(years):
            rows.append((f"d{d}", yr, donor_series[d][i]))
    treat_pre = 0.5 * donor_series[0] + 0.5 * donor_series[1] + rng.normal(0, 0.3, T)
    treat = treat_pre.copy()
    treat[n_pre:] += 9.0
    for i, yr in enumerate(years):
        rows.append(("T", yr, treat[i]))
    df = pd.DataFrame(rows, columns=["unit", "year", "y"])
    treatment_time = years[n_pre]
    return df, treatment_time


def test_bsts_synth_real_panel():
    df, tt = _make_panel()
    res = _bsts.bsts_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        n_simulations=150,
        seed=11,
    )
    assert np.isfinite(res.estimate)
    assert res.estimate > 0


def test_bsts_synth_unit_not_found():
    df, tt = _make_panel()
    with pytest.raises(DataInsufficient):
        _bsts.bsts_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="ZZZ",
            treatment_time=tt,
        )


def test_estimate_beta_ridge_empty_regressors():
    out = _bsts._estimate_beta_ridge(np.zeros(5), np.zeros((5, 0)))
    assert out.shape == (0,)


def test_causal_impact_no_post_observations_raises():
    df, n_pre = _make_ts(n_pre=40, n_post=20)
    # Post-period window beyond the index -> empty post mask.
    with pytest.raises(DataInsufficient, match="post-period"):
        _bsts.causal_impact(
            data=df,
            pre_period=(0, n_pre - 1),
            post_period=(10_000, 20_000),
            outcome="y",
        )


def test_bsts_synth_with_covariates():
    df, tt = _make_panel(seed=21)
    # Attach a covariate column so the averaged-covariate path runs.
    rng = np.random.default_rng(21)
    df = df.copy()
    df["pop"] = rng.normal(100, 5, len(df))
    res = _bsts.bsts_synth(
        data=df,
        outcome="y",
        unit="unit",
        time="year",
        treated_unit="T",
        treatment_time=tt,
        covariates=["pop"],
        n_simulations=120,
        seed=22,
    )
    assert np.isfinite(res.estimate)


def test_bsts_synth_no_pre_periods_raises():
    # Treatment at the earliest period leaves no usable pre-treatment data
    # -> one of the pre-period guard ValueErrors fires.
    df, tt = _make_panel(seed=23)
    earliest = df["year"].min()
    with pytest.raises(DataInsufficient):
        _bsts.bsts_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=earliest,
            n_simulations=50,
            seed=24,
        )


def test_bsts_synth_single_unit_no_donors_raises():
    # Only the treated unit present -> no donors.
    rows = [("T", 2000 + i, float(i)) for i in range(20)]
    df = pd.DataFrame(rows, columns=["unit", "year", "y"])
    with pytest.raises(DataInsufficient):
        _bsts.bsts_synth(
            data=df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit="T",
            treatment_time=2010,
        )


def test_simulate_counterfactual_default_rng():
    # rng=None default-branch in _simulate_counterfactual.
    df, n_pre = _make_ts(effect=4.0, seed=31)
    y_pre = df["y"].values[:n_pre].astype(float)
    X_pre = df[["x1"]].values[:n_pre].astype(float)
    X_post = df[["x1"]].values[n_pre:].astype(float)
    fitted = _bsts._fit_model(y_pre, X_pre, use_trend=False)
    draws = _bsts._simulate_counterfactual(fitted, X_post, n_simulations=30)
    assert draws.shape == (30, X_post.shape[0])
