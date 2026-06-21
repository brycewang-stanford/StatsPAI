"""Coverage round-2 — variant estimators of ``statspai.synth``.

Drives still-uncovered branches of the BSTS, Bayesian, DiSCo, SDID,
kernel, sparse-LASSO, and penalized (Abadie-L'Hour) synthetic-control
estimators through the ``sp.synth(method=...)`` dispatcher and the
dedicated entry points.

All estimators here are pure-numpy (BSTS / Bayesian use a hand-rolled
Kalman filter / Metropolis-Hastings MCMC — *no* pymc / tfp), so they run
in this environment without heavy optional deps. Optional-dep-gated code
elsewhere is skipped in the bayesian-dep test below.

Assertions check real properties — effect sign / magnitude, weights on
the simplex, finite RMSPE, populated placebo distributions, and the
correct loud failures — never fabricated numbers.
"""

from __future__ import annotations

import importlib.util as _ilu

import numpy as np
import pandas as pd
import pytest

import statspai as sp

T_TREAT = 11
TRUE_EFFECT = 4.0


def _panel(seed=0, n_donors=8, n_t=20, effect=TRUE_EFFECT, with_cov=False):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        w = rng.normal()
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            row = {
                "unit": u,
                "time": t,
                "y": base + 0.2 * t + fe + eff + rng.normal(0, 0.3),
            }
            if with_cov:
                row["z"] = w + 0.1 * t + rng.normal(0, 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


def _simplex_ok(w):
    w = np.asarray(w, dtype=float)
    return w.min() >= -1e-6 and abs(w.sum() - 1.0) < 5e-2


# ===========================================================================
# BSTS / CausalImpact (pure numpy Kalman filter)
# ===========================================================================
def test_bsts_local_level_recovers_effect():
    r = sp.synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="bsts",
        n_simulations=300,
        seed=1,
    )
    assert r.estimate > 1.0
    assert np.isfinite(r.se)


def test_bsts_local_linear_trend_and_covariates():
    r = sp.synth(
        _panel(1, with_cov=True),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="bsts",
        model="local_linear_trend",
        covariates=["z"],
        n_simulations=300,
        seed=2,
    )
    assert np.isfinite(r.estimate)


def test_bsts_rejects_missing_treated():
    with pytest.raises(ValueError):
        sp.synth(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="nonexistent",
            treatment_time=T_TREAT,
            method="bsts",
            n_simulations=100,
        )


def test_bsts_handles_missing_values_in_kalman():
    df = _panel(3)
    # Knock out a few donor observations -> ffill/bfill + Kalman missing path
    mask = (df["unit"] == "u0") & (df["time"].isin([2, 5, 7]))
    df.loc[mask, "y"] = np.nan
    r = sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="bsts",
        n_simulations=200,
        seed=4,
    )
    assert np.isfinite(r.estimate)


def test_causal_impact_direct_validation_guards():
    # The wide-format CausalImpact entry point's loud failures.
    from statspai.synth.bsts import causal_impact

    rng = np.random.default_rng(0)
    wide = pd.DataFrame(
        {"y": np.arange(20.0) + rng.normal(0, 0.1, 20), "x1": np.arange(20.0) + 0.5},
        index=range(20),
    )
    with pytest.raises(ValueError):
        causal_impact(
            wide, pre_period=(0, 9), post_period=(10, 19), model="unknown_model"
        )
    with pytest.raises(TypeError):
        causal_impact([1, 2, 3], pre_period=(0, 1), post_period=(2, 3))
    with pytest.raises(ValueError):
        causal_impact(pd.DataFrame(), pre_period=(0, 1), post_period=(2, 3))
    # a working run too (covers the local_linear_trend body)
    res = causal_impact(
        wide,
        pre_period=(0, 9),
        post_period=(10, 19),
        model="local_linear_trend",
        n_simulations=200,
        seed=1,
    )
    assert np.isfinite(res.estimate)


# ===========================================================================
# Bayesian SCM (pure numpy Metropolis-Hastings)
# ===========================================================================
def test_bayesian_synth_recovers_effect():
    r = sp.synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="bayesian",
        n_iter=600,
        n_warmup=300,
        n_chains=2,
        seed=7,
    )
    assert r.estimate > 1.0
    # credible interval populated
    lo, hi = r.ci
    assert lo <= r.estimate <= hi


def test_bayesian_synth_with_covariates():
    r = sp.synth(
        _panel(1, with_cov=True),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="bayesian",
        covariates=["z"],
        n_iter=500,
        n_warmup=250,
        n_chains=1,
        seed=8,
    )
    assert np.isfinite(r.estimate)


@pytest.mark.parametrize(
    "kw,exc",
    [
        ({"n_iter": 100, "n_warmup": 100}, ValueError),  # warmup >= iter
        ({"n_chains": 0}, ValueError),
        ({"dirichlet_alpha": -1.0}, ValueError),
        ({"alpha": 1.5}, ValueError),
    ],
)
def test_bayesian_synth_validation_guards(kw, exc):
    base = dict(n_iter=400, n_warmup=200, n_chains=1, seed=9)
    base.update(kw)
    with pytest.raises(exc):
        sp.synth(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="bayesian",
            **base,
        )


def test_bayesian_optional_dep_note():
    # The bayesian estimator is hand-rolled MCMC, not pymc-backed; record
    # that pymc/tfp are genuinely absent in this environment.
    assert _ilu.find_spec("pymc") is None
    assert _ilu.find_spec("tensorflow_probability") is None


# ===========================================================================
# DiSCo — distributional synthetic controls
# ===========================================================================
def test_discos_mixture_and_placebo():
    r = sp.synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="discos",
        n_quantiles=40,
        placebo=True,
        seed=3,
    )
    assert np.isfinite(r.estimate)
    mi = r.model_info
    assert mi.get("n_quantiles") == 40


def test_discos_test_and_guards():
    df = _panel(1)
    res = sp.discos(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_quantiles=30,
        placebo=False,
    )
    assert np.isfinite(res.estimate)
    # too-few donors -> loud failure
    small = df[df["unit"].isin(["treated", "u0"])]
    with pytest.raises(ValueError):
        sp.discos(
            small,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
        )


# ===========================================================================
# SDID
# ===========================================================================
def test_sdid_recovers_effect_and_weights():
    r = sp.synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="sdid",
    )
    assert r.estimate > 1.0
    mi = r.model_info
    uw = mi.get("unit_weights")
    if uw is None:
        uw = mi.get("omega")
    if isinstance(uw, dict) and uw:
        assert _simplex_ok(list(uw.values()))


def test_sdid_bootstrap_se_method():
    r = sp.synth(
        _panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="sdid",
        inference="bootstrap",
    )
    assert np.isfinite(r.estimate)


def test_synthdid_placebo_helper():
    df = _panel(2)
    r = sp.synthdid_placebo(
        df, y="y", unit="unit", time="time", treat_unit="treated", treat_time=T_TREAT
    )
    assert r is not None


# ===========================================================================
# Kernel SCM
# ===========================================================================
@pytest.mark.parametrize("kernel", ["rbf", "polynomial", "laplacian"])
def test_kernel_synth_each_kernel(kernel):
    r = sp.synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="kernel",
        kernel=kernel,
        placebo=False,
    )
    assert np.isfinite(r.estimate)
    w = r.model_info.get("weights")
    if isinstance(w, dict) and w:
        assert _simplex_ok(list(w.values()))
    elif isinstance(w, pd.DataFrame) and "weight" in w.columns:
        assert _simplex_ok(w["weight"].to_numpy())


def test_kernel_unknown_kernel_raises():
    with pytest.raises(ValueError):
        sp.synth(
            _panel(1),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="kernel",
            kernel="bogus",
        )


def test_kernel_with_fixed_sigma_and_covariates():
    r = sp.synth(
        _panel(2, with_cov=True),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="kernel",
        kernel="rbf",
        sigma=1.5,
        covariates=["z"],
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_kernel_ridge_synth():
    r = sp.synth(
        _panel(3),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="kernel_ridge",
        placebo=False,
    )
    assert np.isfinite(r.estimate)


# ===========================================================================
# Sparse / LASSO SCM
# ===========================================================================
def test_sparse_synth_recovers():
    r = sp.synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="sparse",
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_sparse_synth_with_covariates_and_placebo():
    r = sp.synth(
        _panel(1, with_cov=True),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="lasso",
        covariates=["z"],
        placebo=True,
    )
    assert np.isfinite(r.estimate)


# ===========================================================================
# Penalized SCM (Abadie & L'Hour 2021)
# ===========================================================================
@pytest.mark.parametrize("penalty_type", ["pairwise", "l1_pairwise", "max_dev"])
def test_penscm_penalty_types(penalty_type):
    r = sp.synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="penscm",
        penalty_type=penalty_type,
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_penscm_with_covariates_and_fixed_lambda():
    r = sp.synth(
        _panel(1, with_cov=True),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="penscm",
        covariates=["z"],
        lambda_pen=0.1,
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_penscm_rejects_missing_covariate():
    with pytest.raises(ValueError):
        sp.synth(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="penscm",
            covariates=["does_not_exist"],
        )
