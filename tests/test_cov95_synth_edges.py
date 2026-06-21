"""Edge/error-path coverage: augsynth, gsynth, sequential_sdid, sensitivity."""

import importlib

import numpy as np
import pandas as pd
import pytest

augsynth_mod = importlib.import_module("statspai.synth.augsynth")
gsynth_mod = importlib.import_module("statspai.synth.gsynth")
seqsdid = importlib.import_module("statspai.synth.sequential_sdid")
sensitivity = importlib.import_module("statspai.synth.sensitivity")
cluster_mod = importlib.import_module("statspai.synth.cluster")
expdes = importlib.import_module("statspai.synth.experimental_design")

try:
    from statspai.exceptions import DataInsufficient
except Exception:  # pragma: no cover
    DataInsufficient = ValueError


def _panel(
    n_units=9, n_periods=18, treatment_time=12, effect=4.0, seed=81, with_cov=False
):
    rng = np.random.default_rng(seed)
    alphas = rng.normal(10, 2, n_units)
    betas = rng.normal(0.5, 0.1, n_units)
    common = rng.normal(0, 0.4, n_periods)
    records = []
    for i in range(n_units):
        for ti, t in enumerate(range(1, n_periods + 1)):
            y = alphas[i] + betas[i] * t + common[ti] + rng.normal(0, 0.2)
            if i == 0 and t >= treatment_time:
                y += effect
            row = {"unit": f"u{i}", "time": t, "outcome": y}
            if with_cov:
                row["x1"] = alphas[i] + 0.05 * t + rng.normal(0, 0.1)
            records.append(row)
    return pd.DataFrame(records)


COMMON = dict(
    outcome="outcome", unit="unit", time="time", treated_unit="u0", treatment_time=12
)


# ---------------- augsynth ----------------
def test_augsynth_native_basic():
    df = _panel()
    res = augsynth_mod.augsynth(df, **COMMON, placebo=True)
    assert np.isfinite(res.estimate)
    mi = res.model_info or {}
    assert "ridge_lambda" in mi or "weights" in mi


def test_augsynth_explicit_ridge_and_covariates():
    df = _panel(with_cov=True)
    res = augsynth_mod.augsynth(
        df, **COMMON, ridge_lambda=0.5, covariates=["x1"], placebo=False
    )
    assert np.isfinite(res.estimate)


def test_augsynth_unknown_backend():
    df = _panel()
    with pytest.raises(ValueError):
        augsynth_mod.augsynth(df, **COMMON, backend="matlab")


def test_augsynth_r_backend_covariates_not_impl():
    df = _panel(with_cov=True)
    with pytest.raises(NotImplementedError):
        augsynth_mod.augsynth(df, **COMMON, covariates=["x1"], backend="r")


def test_augsynth_r_backend_ridge_not_impl():
    df = _panel()
    with pytest.raises(NotImplementedError):
        augsynth_mod.augsynth(df, **COMMON, ridge_lambda=0.5, backend="augsynth")


def test_augsynth_missing_col():
    df = _panel()
    with pytest.raises(ValueError):
        augsynth_mod.augsynth(
            df,
            outcome="nope",
            unit="unit",
            time="time",
            treated_unit="u0",
            treatment_time=12,
        )


def test_augsynth_treated_not_found():
    df = _panel()
    with pytest.raises(ValueError):
        augsynth_mod.augsynth(df, **{**COMMON, "treated_unit": "ghost"})


def test_augsynth_too_few_pre():
    df = _panel()
    with pytest.raises((DataInsufficient, ValueError)):
        augsynth_mod.augsynth(df, **{**COMMON, "treatment_time": 2})


# ---------------- gsynth ----------------
def test_gsynth_unknown_backend():
    df = _panel()
    with pytest.raises(ValueError):
        gsynth_mod.gsynth(df, **COMMON, backend="matlab")


def test_gsynth_r_backend_covariates_not_impl():
    df = _panel(with_cov=True)
    with pytest.raises(NotImplementedError):
        gsynth_mod.gsynth(df, **COMMON, covariates=["x1"], backend="r")


def test_gsynth_r_backend_nfactors_not_impl():
    df = _panel()
    with pytest.raises(NotImplementedError):
        gsynth_mod.gsynth(df, **COMMON, n_factors=2, backend="gsynth")


def test_gsynth_too_few_pre():
    df = _panel()
    with pytest.raises((DataInsufficient, ValueError)):
        gsynth_mod.gsynth(df, **{**COMMON, "treatment_time": 2}, backend="native")


# ---------------- sequential_sdid ----------------
def _staggered_cohort_panel(seed=91):
    rng = np.random.default_rng(seed)
    records = []
    adopt = {"a": 8, "b": 8, "c": 12, "d": None, "e": None, "f": None, "g": None}
    for u, g in adopt.items():
        a = rng.normal(10, 1)
        for t in range(1, 17):
            y = a + 0.4 * t + rng.normal(0, 0.2)
            if g is not None and t >= g:
                y += 3.0
            records.append(
                {
                    "unit": u,
                    "time": t,
                    "outcome": y,
                    "cohort": g if g is not None else 0,
                }
            )
    return pd.DataFrame(records)


@pytest.mark.parametrize("se_method", ["placebo", "bootstrap"])
def test_sequential_sdid_se_methods(se_method):
    df = _staggered_cohort_panel()
    res = seqsdid.sequential_sdid(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        cohort="cohort",
        never_treated_value=0,
        se_method=se_method,
        n_reps=20,
        cohort_weights="size",
        seed=0,
    )
    assert np.isfinite(res.estimate)


def test_sequential_sdid_equal_weights():
    df = _staggered_cohort_panel()
    res = seqsdid.sequential_sdid(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        cohort="cohort",
        never_treated_value=0,
        cohort_weights="equal",
        n_reps=15,
        seed=1,
    )
    assert np.isfinite(res.estimate)
    assert isinstance(res.summary(), str)


# ---------------- sensitivity edge cases ----------------
def test_synth_donor_sensitivity_default_k():
    df = _panel()
    out = sensitivity.synth_donor_sensitivity(
        df, **COMMON, k=None, n_samples=15, seed=0
    )
    assert isinstance(out, pd.DataFrame) and len(out) >= 1


def test_synth_time_placebo_default_ntimes():
    df = _panel()
    out = sensitivity.synth_time_placebo(df, **COMMON, n_placebo_times=None)
    assert isinstance(out, pd.DataFrame) and len(out) >= 1


def test_synth_rmspe_filter_default_thresholds():
    df = _panel()
    out = sensitivity.synth_rmspe_filter(df, **COMMON, thresholds=None)
    assert isinstance(out, pd.DataFrame) and len(out) >= 1


# ---------------- cluster extra branches ----------------
def test_cluster_spectral():
    df = _panel(n_units=10)
    res = cluster_mod.cluster_synth(
        df, **COMMON, n_clusters=2, cluster_method="spectral", placebo=False, seed=0
    )
    assert np.isfinite(res.estimate)


def test_cluster_augment_with_placebo():
    df = _panel(n_units=10)
    res = cluster_mod.cluster_synth(
        df, **COMMON, n_clusters=3, augment=True, max_augment=2, placebo=True, seed=1
    )
    assert np.isfinite(res.estimate)


# ---------------- experimental_design error/option paths ----------------
def test_expdes_negative_concentration():
    df = _panel()
    with pytest.raises(ValueError):
        expdes.synth_experimental_design(
            df,
            unit="unit",
            time="time",
            outcome="outcome",
            k=2,
            concentration_weight=-1.0,
            n_random=10,
        )


def test_expdes_unknown_candidates():
    df = _panel()
    with pytest.raises(ValueError):
        expdes.synth_experimental_design(
            df,
            unit="unit",
            time="time",
            outcome="outcome",
            k=2,
            candidates=["u0", "ghost"],
            n_random=10,
        )


def test_expdes_unknown_donors():
    df = _panel()
    with pytest.raises(ValueError):
        expdes.synth_experimental_design(
            df,
            unit="unit",
            time="time",
            outcome="outcome",
            k=2,
            donors=["u1", "ghost"],
            n_random=10,
        )


def test_expdes_pre_period_empty():
    df = _panel()
    with pytest.raises(ValueError):
        expdes.synth_experimental_design(
            df,
            unit="unit",
            time="time",
            outcome="outcome",
            k=2,
            pre_period=(1000, 2000),
            n_random=10,
        )


def test_expdes_explicit_candidates_and_donors():
    df = _panel(n_units=10)
    res = expdes.synth_experimental_design(
        df,
        unit="unit",
        time="time",
        outcome="outcome",
        k=2,
        candidates=["u0", "u1", "u2", "u3"],
        donors=["u4", "u5", "u6", "u7", "u8"],
        pre_period=(1, 12),
        n_random=30,
        random_state=0,
    )
    assert hasattr(res, "summary")
