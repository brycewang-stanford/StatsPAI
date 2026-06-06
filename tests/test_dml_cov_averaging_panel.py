"""Coverage campaign — DML sensitivity rendering, weighted model averaging,
weighted panel DML, and overlap/orthogonality diagnostics warnings.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Drives the remaining reachable lines of
``dml/_sensitivity.py`` (result ``summary()`` / ``plot()``),
``dml/model_averaging.py`` (sample-weighted candidate fits + input validation),
``dml/panel_dml.py`` (sample-weighted two-way demeaning + score), and
``dml/_diagnostics.py`` (overlap / orthogonality warning paths).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402


@pytest.fixture(scope="module")
def plr_data():
    rng = np.random.default_rng(0)
    n, p = 500, 4
    X = rng.standard_normal((n, p))
    g = X @ rng.standard_normal(p) * 0.3
    d = g + rng.standard_normal(n)
    y = 2.0 * d + g + rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["d"] = d
    df["y"] = y
    df["w"] = rng.uniform(0.5, 1.5, n)
    return df, [f"x{i}" for i in range(p)]


@pytest.fixture(scope="module")
def plr_result(plr_data):
    df, X = plr_data
    return sp.dml(df, y="y", d="d", X=X, model_y="rf", model_d="rf", n_folds=3)


# ─── sensitivity rendering ───────────────────────────────────────────────


def test_sensitivity_summary_and_plot(plr_result, plr_data):
    _, X = plr_data
    sens = sp.dml_sensitivity(plr_result, benchmark_covariates=[X[0], X[1]])
    s = sens.summary()
    assert isinstance(s, str) and len(s) > 0
    out = sens.plot()
    assert out is not None
    plt.close("all")


# ─── weighted model averaging + validation ───────────────────────────────


def test_model_averaging_weighted(plr_data):
    df, X = plr_data
    res = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=X,
        n_folds=3,
        seed=0,
        sample_weight=df["w"].to_numpy(),
    )
    coef = float(getattr(res, "coef", getattr(res, "estimate", np.nan)))
    assert np.isfinite(coef)


def test_model_averaging_missing_column_raises(plr_data):
    df, X = plr_data
    with pytest.raises(ValueError, match="not found"):
        sp.dml_model_averaging(
            df, y="y", treat="d", covariates=X + ["does_not_exist"], n_folds=3
        )


def test_model_averaging_no_covariates_raises(plr_data):
    df, _ = plr_data
    with pytest.raises(ValueError, match="covariate"):
        sp.dml_model_averaging(df, y="y", treat="d", covariates=[], n_folds=3)


def test_model_averaging_empty_candidates_raises(plr_data):
    df, X = plr_data
    with pytest.raises(ValueError, match="candidate"):
        sp.dml_model_averaging(
            df, y="y", treat="d", covariates=X, candidates=[], n_folds=3
        )


def test_model_averaging_bad_weight_raises(plr_data):
    df, X = plr_data
    with pytest.raises(ValueError, match="non-finite|zero total mass|non-negative"):
        sp.dml_model_averaging(
            df,
            y="y",
            treat="d",
            covariates=X,
            n_folds=3,
            sample_weight=np.zeros(len(df)),
        )


# ─── weighted panel DML ──────────────────────────────────────────────────


def test_panel_dml_weighted(plr_data):
    from sklearn.ensemble import RandomForestRegressor

    df, X = plr_data
    n = len(df)
    df = df.copy()
    df["unit"] = np.repeat(np.arange(n // 4), 4)[:n]
    df["time"] = np.tile(np.arange(4), n // 4 + 1)[:n]
    rf = RandomForestRegressor(n_estimators=40, random_state=0)
    res = sp.dml_panel(
        df,
        y="y",
        treat="d",
        covariates=X,
        unit="unit",
        time="time",
        ml_g=rf,
        ml_m=rf,
        n_folds=3,
        sample_weight="w",
    )
    assert np.isfinite(float(getattr(res, "coef", getattr(res, "estimate", np.nan))))


# ─── diagnostics warnings (poor overlap) ─────────────────────────────────


def test_diagnostics_overlap_warning():
    # near-deterministic propensity → poor overlap → warning paths fire
    rng = np.random.default_rng(9)
    n, p = 500, 3
    X = rng.standard_normal((n, p))
    lin = X @ np.array([4.0, 4.0, 4.0])  # very steep → propensities near 0/1
    d = (lin + 0.1 * rng.standard_normal(n) > 0).astype(float)
    y = 2.0 * d + X.sum(1) * 0.2 + rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["d"] = d
    df["y"] = y
    res = sp.dml(
        df,
        y="y",
        d="d",
        X=[f"x{i}" for i in range(p)],
        model_y="rf",
        model_d="rf",
        model="IRM",
        n_folds=3,
    )
    diag = sp.dml_diagnostics(res, clip=0.01)
    s = diag.summary()
    assert isinstance(s, str)
