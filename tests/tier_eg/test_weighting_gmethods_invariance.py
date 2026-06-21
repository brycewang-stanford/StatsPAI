"""Tier E/G metamorphic checks for weighting and g-method estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp

from ._helpers import (
    assert_finite_estimate,
    assert_invariant,
    assert_raises_clean,
    assert_scaled,
    coef,
    stderr,
)

COVARIATES = ["x1", "x2"]


def _make_cia_data(n: int = 500, tau: float = 2.0, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.45 * x1 - 0.35 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    d = rng.binomial(1, p, size=n)
    y0 = 0.8 + 0.7 * x1 - 0.4 * x2 + rng.normal(scale=0.45, size=n)
    y = y0 + tau * d
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def _make_g_estimation_data(
    n: int = 450, tau: float = 1.4, seed: int = 321
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.4 * x1 + 0.2 * x2)))
    d = rng.binomial(1, p, size=n)
    y = 1.0 + tau * d + 0.6 * x1 - 0.3 * x2 + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def _ipw(df: pd.DataFrame):
    return sp.ipw(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        estimand="ATE",
        n_bootstrap=30,
        seed=11,
    )


def _aipw(df: pd.DataFrame):
    return sp.aipw(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        estimand="ATE",
        n_folds=4,
        seed=11,
    )


def _gcomp(df: pd.DataFrame):
    return sp.g_computation(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        estimand="ATE",
        n_boot=30,
        seed=11,
    )


def _gest(df: pd.DataFrame):
    return sp.g_estimation(
        df,
        y="y",
        treatments=["d"],
        covariates_by_stage=[COVARIATES],
        n_bootstrap=30,
        random_state=11,
    )


def _cbps(df: pd.DataFrame):
    return sp.cbps(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        estimand="ATE",
        variant="over",
        n_bootstrap=30,
        seed=11,
    )


def _overlap(df: pd.DataFrame):
    return sp.overlap_weights(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        estimand="ATO",
        n_bootstrap=30,
        seed=11,
    )


def _ebalance(df: pd.DataFrame):
    return sp.ebalance(df, y="y", treat="d", covariates=COVARIATES, moments=1)


def test_ipw_point_estimate_metamorphic_relations():
    df = _make_cia_data()
    base = _ipw(df)
    assert_finite_estimate(base)

    shuffled = _ipw(df.sample(frac=1.0, random_state=7).reset_index(drop=True))
    assert_invariant(coef(base), coef(shuffled), rtol=1e-11, atol=1e-11)

    shifted = _ipw(df.assign(y=df["y"] + 9.0))
    assert_invariant(coef(base), coef(shifted), rtol=1e-11, atol=1e-11)

    scaled = _ipw(df.assign(y=-3.5 * df["y"]))
    assert_scaled(coef(base), coef(scaled), -3.5, rtol=1e-11, atol=1e-11)

    flipped = _ipw(df.assign(d=1 - df["d"]))
    assert_scaled(coef(base), coef(flipped), -1.0, rtol=1e-10, atol=1e-10)


def test_aipw_point_and_se_respect_outcome_affine_transforms():
    df = _make_cia_data()
    base = _aipw(df)
    assert_finite_estimate(base)

    shifted = _aipw(df.assign(y=df["y"] - 12.0))
    assert_invariant(coef(base), coef(shifted), rtol=1e-11, atol=1e-11)
    assert_invariant(stderr(base), stderr(shifted), rtol=1e-11, atol=1e-11, what="se")

    scaled = _aipw(df.assign(y=2.25 * df["y"]))
    assert_scaled(coef(base), coef(scaled), 2.25, rtol=1e-11, atol=1e-11)
    assert_scaled(stderr(base), stderr(scaled), 2.25, rtol=1e-11, atol=1e-11, what="se")


def test_g_computation_metamorphic_relations():
    df = _make_cia_data()
    base = _gcomp(df)
    assert_finite_estimate(base)

    shuffled = _gcomp(df.sample(frac=1.0, random_state=13).reset_index(drop=True))
    assert_invariant(coef(base), coef(shuffled), rtol=1e-11, atol=1e-11)

    shifted = _gcomp(df.assign(y=df["y"] + 5.0))
    assert_invariant(coef(base), coef(shifted), rtol=1e-11, atol=1e-11)

    scaled = _gcomp(df.assign(y=-1.75 * df["y"]))
    assert_scaled(coef(base), coef(scaled), -1.75, rtol=1e-11, atol=1e-11)

    flipped = _gcomp(df.assign(d=1 - df["d"]))
    assert_scaled(coef(base), coef(flipped), -1.0, rtol=1e-11, atol=1e-11)


def test_g_estimation_metamorphic_relations():
    df = _make_g_estimation_data()
    base = _gest(df)
    assert_finite_estimate(base)

    shifted = _gest(df.assign(y=df["y"] + 100.0))
    assert_invariant(coef(base), coef(shifted), rtol=1e-11, atol=1e-11)

    scaled = _gest(df.assign(y=3.0 * df["y"]))
    assert_scaled(coef(base), coef(scaled), 3.0, rtol=1e-11, atol=1e-11)
    assert_scaled(stderr(base), stderr(scaled), 3.0, rtol=1e-11, atol=1e-11, what="se")


def test_matching_weight_estimators_respect_outcome_affine_transforms():
    df = _make_cia_data(n=700)
    for fit in (_ebalance, _cbps, _overlap):
        base = fit(df)
        assert_finite_estimate(base)

        shifted = fit(df.assign(y=df["y"] + 4.0))
        assert_invariant(coef(base), coef(shifted), rtol=1e-9, atol=1e-9)

        scaled = fit(df.assign(y=-2.0 * df["y"]))
        assert_scaled(coef(base), coef(scaled), -2.0, rtol=1e-9, atol=1e-9)
        assert_scaled(
            stderr(base), stderr(scaled), 2.0, rtol=1e-9, atol=1e-9, what="se"
        )


def test_entropy_balance_weights_are_finite_and_normalized():
    result = _ebalance(_make_cia_data())
    weights = np.asarray(result.model_info["weights"], dtype=float)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0)
    assert_invariant(weights.sum(), 1.0, rtol=1e-12, atol=1e-12, what="weights")
    assert result.model_info["eff_sample_size"] > 1.0


def test_weighting_and_g_methods_fail_loudly_on_bad_inputs():
    df = _make_cia_data()
    one_arm = df.assign(d=1)
    nonbinary = df.assign(d=np.where(np.arange(len(df)) % 3 == 0, 2, df["d"]))
    continuous = df.assign(d=df["x1"])

    assert_raises_clean(
        lambda: _ipw(one_arm),
        ValueError,
        match="both treated and control",
    )
    assert_raises_clean(lambda: _aipw(nonbinary), ValueError, match="binary")
    assert_raises_clean(
        lambda: _gcomp(continuous),
        ValueError,
        match="requires binary treatment",
    )
    assert_raises_clean(
        lambda: sp.g_estimation(
            df,
            y="y",
            treatments=["d"],
            covariates_by_stage=[["x1", "missing_covariate"]],
            n_bootstrap=10,
        ),
        ValueError,
        match="Columns not found",
    )
    assert_raises_clean(lambda: _overlap(nonbinary), ValueError, match="binary")
