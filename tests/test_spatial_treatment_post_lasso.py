"""``sp.spatial_treatment(method='dml', learner='post_lasso')`` (Pollmann 2026).

The paper's application fits both first steps by "post-LASSO procedure with
data-driven penalty (Belloni et al. 2012)", the outcome model "with the
distance of interest always included". The checks are T1: an identity with
the OLS path when every column is selected, a high-dimensional confounded
design with known effects, and bookkeeping.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.spatial._treatments_dml import _post_lasso_fit, dml_rings
from statspai.spatial.treatments import _ring_weights


def _confounded(seed=0, n=1000):
    rng = np.random.default_rng(seed)
    k = 10
    g = (np.arange(k) + 0.5) / k
    locs = pd.DataFrame({"x": np.repeat(g, k), "y": np.tile(g, k)})
    locs["p"] = 1 / (1 + np.exp(-(-1.5 + 3 * locs.x)))
    units = pd.DataFrame(rng.uniform(0, 1, (n, 2)), columns=["x", "y"])
    D = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    W = (rng.uniform(size=len(locs)) < locs.p.values).astype(int)
    units["Y"] = (
        10 + 4 * units.x + 2 * (D[:, W == 1] <= 0.04).sum(1) + rng.normal(size=n)
    )
    return units, locs.assign(realized=W), D, W


def _high_dim(seed, k=14, n=1200, pu=500, pz=10):
    """Location x confounds; 500 noise unit covariates, 10 location ones."""
    rng = np.random.default_rng(seed)
    g = (np.arange(k) + 0.5) / k
    locs = pd.DataFrame({"x": np.repeat(g, k), "y": np.tile(g, k)})
    for j in range(pz):
        locs[f"z{j}"] = rng.normal(size=len(locs))
    p = 1 / (1 + np.exp(-(-1.5 + 3 * locs.x.values)))
    units = pd.DataFrame(rng.uniform(0, 1, (n, 2)), columns=["x", "y"])
    noise = pd.DataFrame(rng.normal(size=(n, pu)), columns=[f"u{j}" for j in range(pu)])
    units = pd.concat([units, noise], axis=1)
    D = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    W = (rng.uniform(size=len(locs)) < p).astype(int)
    units["Y"] = (
        10 + 4 * units.x + 2 * (D[:, W == 1] <= 0.04).sum(1) + rng.normal(size=n)
    )
    cov = ["x"] + [f"u{j}" for j in range(pu)]
    lcov = ["x"] + [f"z{j}" for j in range(pz)]
    return units, locs.assign(realized=W), cov, lcov


def _fit(units, L, learner, seed=0, **kw):
    return sp.spatial_treatment(
        units,
        "Y",
        ["x", "y"],
        L,
        bins=[0, 0.04, 0.08],
        max_effect_distance=0.08,
        n_perm=0,
        seed=seed,
        method="dml",
        learner=learner,
        **kw,
    )


def test_default_learner_output_is_unchanged():
    """Pinned before learner= existed (byte-identical to the previous code
    on five configurations, checked when the argument was added)."""
    units, L, _, _ = _confounded()
    r = _fit(units, L, "ols", covariates=["x"], location_covariates=["x"])
    np.testing.assert_allclose(
        r.table.estimate, [2.1121404215498707, 0.04529704209594953], rtol=1e-12
    )
    np.testing.assert_allclose(
        r.table.se, [0.2935457453015944, 0.3788730673529456], rtol=1e-12
    )
    default = sp.spatial_treatment(
        units,
        "Y",
        ["x", "y"],
        L,
        bins=[0, 0.04, 0.08],
        max_effect_distance=0.08,
        n_perm=0,
        method="dml",
        covariates=["x"],
        location_covariates=["x"],
    )
    assert default.table.equals(r.table)
    assert "learner" not in default.model_info


def test_post_lasso_selecting_every_column_is_the_ols_path():
    """A vanishing penalty selects every column; the refitted outcome model
    then has the OLS model's regressors and the post-LASSO logit is the
    unpenalized logit, so m, p and every ring estimate agree."""
    units, L, D, W = _confounded()
    Wf = W.astype(float)
    Y = units.Y.values
    edges = np.array([0, 0.04, 0.08])
    ring_w = [
        _ring_weights(D, edges[j], edges[j + 1], "bin", first=(j == 0))
        for j in range(2)
    ]
    Xu = np.column_stack([units.x, units.x**2, units.y])
    Z = np.column_stack([L.x, L.y])
    a = dml_rings(D, Wf, Y, ring_w, 0.08, None, Z, Xu, 10, np.random.default_rng(3))
    b = dml_rings(
        D,
        Wf,
        Y,
        ring_w,
        0.08,
        None,
        Z,
        Xu,
        10,
        np.random.default_rng(3),
        learner="post_lasso",
        lasso_penalty={"homoscedastic": "none", "lambda.start": 1e-10},
        logit_penalty={"lambda": 1e-12},
    )
    # every penalized column (3 covariates + the other ring) in every fold
    assert all(s == [4, 4] for s in b["n_selected_outcome"])
    assert all(s == 2 for s in b["n_selected_propensity"])
    np.testing.assert_allclose(b["p_hat"], a["p_hat"], rtol=1e-10)
    for k in range(2):
        np.testing.assert_allclose(b["mhat"][k], a["mhat"], atol=1e-10)
        ra, rb = a["rings"][k], b["rings"][k]
        assert rb["est"] == pytest.approx(ra["est"], rel=1e-10, abs=1e-12)
        assert rb["comp"]["var"] == pytest.approx(ra["comp"]["var"], rel=1e-10)


def test_post_lasso_fit_always_keeps_the_forced_columns():
    rng = np.random.default_rng(0)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=(n, 30))])
    y = 1.0 + 0.0 * X[:, 1] + 3.0 * X[:, 2] + rng.normal(size=n)
    forced = np.zeros(X.shape[1], dtype=bool)
    forced[:2] = True  # intercept and a column with no effect
    beta, sel = _post_lasso_fit(X, y, forced, None)
    assert sel[0] and sel.sum() == 1  # the signal column, nothing else
    assert beta[1] != 0.0  # forced in although it has no effect
    ols, *_ = np.linalg.lstsq(X[:, :3], y, rcond=None)
    np.testing.assert_allclose(beta[:3], ols, rtol=1e-12)
    assert np.all(beta[3:] == 0.0)


def test_post_lasso_recovers_the_effect_where_ols_breaks():
    """500 noise unit covariates against 581-762 training units per fold:
    the OLS outcome model overfits and the OLS DML estimate explodes
    (RMSE ~1e4 in 40 draws), post-LASSO recovers the effects (+2 within
    0.04, 0 beyond). Thresholds fixed before the run: bias within 4 MC SE,
    95% coverage at least 0.9, OLS RMSE at least 10 times post-LASSO's.
    Observed (40 draws): mean 1.977 / 0.015, MC SD 0.14 / 0.11, coverage
    1.0 (the design-based SE is conservative, 3-4 times the MC SD)."""
    truth = np.array([2.0, 0.0])
    est, se, ols = [], [], []
    with warnings.catch_warnings():
        # OLS propensities hit the clip; that warning is the point here.
        warnings.simplefilter("ignore")
        for b in range(30):
            units, L, cov, lcov = _high_dim(b)
            r = _fit(
                units, L, "post_lasso", b, covariates=cov, location_covariates=lcov
            )
            est.append(r.table.estimate.values)
            se.append(r.table.se.values)
            if b < 8:
                o = _fit(units, L, "ols", b, covariates=cov, location_covariates=lcov)
                ols.append(o.table.estimate.values)
    est, se, ols = np.array(est), np.array(se), np.array(ols)
    sd = est.std(axis=0)
    np.testing.assert_array_less(
        np.abs(est.mean(axis=0) - truth), 4 * sd / np.sqrt(len(est))
    )
    assert np.all((np.abs(est - truth) <= 1.96 * se).mean(axis=0) >= 0.9)
    rmse_pl = np.sqrt(((est - truth) ** 2).mean(axis=0))
    rmse_ols = np.sqrt(((ols - truth) ** 2).mean(axis=0))
    assert np.all(rmse_ols >= 10 * rmse_pl)
    assert r.model_info["learner"] == "post_lasso"
    assert r.diagnostics["propensity_source"] == "cross-fitted post-LASSO logit"
    # x is the only relevant covariate on either side; selection is sparse
    assert max(max(s) for s in r.diagnostics["n_selected_outcome"]) <= 10
    assert max(r.diagnostics["n_selected_propensity"]) <= 4


def test_post_lasso_is_deterministic_and_reports_selection():
    units, L, cov, lcov = _high_dim(1, pu=50)
    a = _fit(units, L, "post_lasso", 7, covariates=cov, location_covariates=lcov)
    b = _fit(units, L, "post_lasso", 7, covariates=cov, location_covariates=lcov)
    assert a.table.equals(b.table)
    n_folds = len(a.diagnostics["n_train_units"])
    assert len(a.diagnostics["n_selected_outcome"]) == n_folds
    assert all(len(s) == 2 for s in a.diagnostics["n_selected_outcome"])
    assert len(a.diagnostics["n_selected_propensity"]) == n_folds
    t = a.table
    np.testing.assert_allclose(t.estimate, t.mu_treated - t.mu_control, atol=1e-12)
    # Known probabilities: no propensity model, outcome still post-LASSO.
    k = _fit(units, L.assign(p=0.5), "post_lasso", 7, covariates=cov, propensity="p")
    assert k.diagnostics["n_selected_propensity"] == []
    assert np.all(k.diagnostics["propensity_hat"] == 0.5)


def test_learner_input_errors():
    units, L, _, _ = _confounded(n=200)
    with pytest.raises(MethodIncompatibility, match="learner must be"):
        _fit(units, L, "forest")
    with pytest.raises(MethodIncompatibility, match="method='dml' only"):
        sp.spatial_treatment(
            units,
            "Y",
            ["x", "y"],
            L,
            bins=[0, 0.04, 0.08],
            max_effect_distance=0.08,
            n_perm=0,
            learner="post_lasso",
        )
