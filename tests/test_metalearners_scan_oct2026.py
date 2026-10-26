"""Option scan (Oct 2026) of ``statspai.metalearners``.

Each test states the correct behaviour; ground truth is an identity, a
known-truth design or a Monte Carlo sampling distribution.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _linear_models():
    from sklearn.linear_model import LinearRegression, LogisticRegression

    return LinearRegression(), LogisticRegression(max_iter=1000)


def _rct(n: int, seed: int, ate: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.integers(0, 2, size=n)
    y = 0.5 * x1 + 0.3 * x2 + (ate + 0.5 * x1) * d + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


# ---------------------------------------------------------------------------
# 5. Diagnostics must not pair effects with the wrong rows
# ---------------------------------------------------------------------------


def test_cate_by_group_aligns_with_estimation_rows():
    df = _rct(400, 0)
    df["g"] = (df["x1"] > 0).astype(int) + (df["x2"] > 0).astype(int)
    df.loc[:39, "x2"] = np.nan  # metalearner drops these rows
    om, pm = _linear_models()
    res = sp.metalearner(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        learner="t",
        outcome_model=om,
        propensity_model=pm,
    )
    complete = df.dropna(subset=["x2"]).reset_index(drop=True)
    expected = sp.cate_by_group(res, complete, by="g")
    got = sp.cate_by_group(res, df, by="g")
    pd.testing.assert_frame_equal(got, expected)


# ---------------------------------------------------------------------------
# 9. BLP after a clustered / fold-indexed metalearner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("design", ["cluster", "fold_indices"])
def test_blp_test_after_clustered_metalearner(design):
    from sklearn.linear_model import LinearRegression

    df = _rct(600, 1)
    df["c"] = np.repeat(np.arange(60), 10)
    df["p"] = 0.5
    om, pm = _linear_models()
    kw = (
        {"cluster": "c"}
        if design == "cluster"
        else {"fold_indices": np.arange(len(df)) % 5}
    )
    res = sp.metalearner(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        learner="dr",
        outcome_model=om,
        propensity_model=pm,
        cate_model=LinearRegression(),
        **kw,
    )
    out = sp.blp_test(
        res, df, y="y", treat="d", covariates=["x1", "x2"], propensity="p"
    )
    assert np.isfinite(out["beta1"]) and np.isfinite(out["beta2"])
    # ATE is 1; the BLP intercept on D - p is the ATE.
    assert abs(out["beta1"] - 1.0) < 4 * out["beta1_se"]


# ---------------------------------------------------------------------------
# 6. auto_cate: beta1 is the ATE, its p-value tests zero
# ---------------------------------------------------------------------------


def test_auto_cate_blp_beta1_pvalue_tests_zero():
    df = _rct(2000, 2, ate=1.0)
    om, pm = _linear_models()
    from sklearn.linear_model import LinearRegression

    res = sp.auto_cate(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        learners=("t",),
        outcome_model=om,
        propensity_model=pm,
        cate_model=LinearRegression(),
        n_folds=2,
    )
    row = res.leaderboard.iloc[0]
    assert abs(row["blp_beta1"] - 1.0) < 0.2
    # H0: beta1 = 0 is rejected decisively when the ATE is 1.
    assert row["blp_beta1_pvalue"] < 1e-6


# ---------------------------------------------------------------------------
# 7. auto_cate_tuned must choose the propensity by a propensity loss
# ---------------------------------------------------------------------------


def test_auto_cate_tuned_selects_propensity_by_its_own_loss():
    pytest.importorskip("optuna")
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(3)
    n = 600
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.normal(size=n)
    e = 1 / (1 + np.exp(-3 * np.sin(2 * x1)))
    d = (rng.uniform(size=n) < e).astype(int)
    y = x1 + d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    space = {
        "outcome_n_estimators": [50],
        "outcome_max_depth": [2],
        "outcome_learning_rate": [0.1],
        "outcome_subsample": [1.0],
        "propensity_n_estimators": [50],
        "propensity_max_depth": [1, 4],
        "propensity_learning_rate": [0.01, 0.1],
    }
    # Cross-fitted log loss of every propensity candidate, same folds.
    X = df[["x1", "x2"]].to_numpy()
    losses = {}
    for depth in (1, 4):
        for lr in (0.01, 0.1):
            m = GradientBoostingClassifier(
                n_estimators=50, max_depth=depth, learning_rate=lr, random_state=42
            )
            p = np.zeros(n)
            for tr, te in KFold(3, shuffle=True, random_state=42).split(X):
                p[te] = clone(m).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
            p = np.clip(p, 0.01, 0.99)
            losses[(depth, lr)] = -np.mean(d * np.log(p) + (1 - d) * np.log(1 - p))
    best = min(losses, key=losses.get)
    import optuna

    picks = set()
    for seed in (0, 1):
        # A grid visits all four candidates, in a seed-dependent order.
        res = sp.auto_cate_tuned(
            df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            learners=("t",),
            tune="nuisance",
            n_trials=4,
            search_space=space,
            n_folds=3,
            random_state=42,
            sampler=optuna.samplers.GridSampler(space, seed=seed),
        )
        tp = res.best_result.model_info["tuned_params"]
        picks.add((tp["propensity_max_depth"], tp["propensity_learning_rate"]))
    assert picks == {best}


# ---------------------------------------------------------------------------
# 4. focal_cate: se_grid is a standard error of tau(x)
# ---------------------------------------------------------------------------


def test_focal_cate_se_matches_sampling_sd():
    test = pd.DataFrame({"x1": [0.5], "x2": [0.0]})
    est, ses = [], []
    for r in range(60):
        rng = np.random.default_rng(100 + r)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        d = rng.integers(0, 2, size=n)
        y0 = 1.0 + x2 + rng.normal(size=n)
        y1 = y0 + 1.0 + 0.5 * x1
        df = pd.DataFrame(
            {
                "y_t0": np.where(d == 1, y1, y0),
                "y_t1": np.where(d == 1, y1 + 0.3, y0),
                "d": d,
                "x1": x1,
                "x2": x2,
            }
        )
        res = sp.focal_cate(
            df,
            y_columns=["y_t0", "y_t1"],
            treat="d",
            covariates=["x1", "x2"],
            test_data=test,
        )
        est.append(res.cate_grid[0, 0])
        ses.append(res.se_grid[0, 0])
    ratio = np.mean(ses) / np.std(est, ddof=1)
    assert 0.7 < ratio < 1.5, ratio


# ---------------------------------------------------------------------------
# 12 / 13. Validation
# ---------------------------------------------------------------------------


def test_cluster_cate_rejects_non_01_treatment():
    df = _rct(300, 4)
    df["d"] = df["d"] + 1  # coded 1 / 2
    with pytest.raises((MethodIncompatibility, ValueError)):
        sp.cluster_cate(df, y="y", treat="d", covariates=["x1", "x2"], n_clusters=3)


def test_cate_eval_rejects_unknown_target():
    rng = np.random.default_rng(5)
    n = 200
    T = rng.integers(0, 2, n)
    Y = rng.normal(size=n)
    with pytest.raises((MethodIncompatibility, ValueError)):
        sp.cate_eval(
            rng.normal(size=n),
            Y,
            T,
            e_hat=np.full(n, 0.5),
            mu1_hat=np.zeros(n),
            mu0_hat=np.zeros(n),
            target="bogus",
        )


def test_blp_and_gate_align_with_estimation_rows():
    """Identity: diagnostics on the frame the model saw (with the rows it
    dropped) equal the same diagnostics on its complete cases."""
    df = _rct(500, 7)
    df["p"] = 0.5
    df["w"] = 1.0
    df.loc[[3, 50, 51, 400], "x2"] = np.nan
    df.loc[[10, 77], "w"] = np.nan  # a design column with gaps
    om, pm = _linear_models()
    res = sp.metalearner(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        learner="t",
        outcome_model=om,
        propensity_model=pm,
        weights="w",
    )
    complete = df.dropna(subset=["x2", "w"]).reset_index(drop=True)
    assert len(res.model_info["cate"]) == len(complete)
    kw = dict(
        y="y", treat="d", covariates=["x1", "x2"], propensity="p", proxy="in_sample"
    )
    a = sp.blp_test(res, df, **kw)
    b = sp.blp_test(res, complete, **kw)
    assert a["beta1"] == b["beta1"] and a["beta2_se"] == b["beta2_se"]
    ga = sp.gate_test(res, df, by="x1", **kw)
    gb = sp.gate_test(res, complete, by="x1", **kw)
    pd.testing.assert_frame_equal(ga["gate_table"], gb["gate_table"])


def test_cate_by_group_refuses_unpairable_rows():
    df = _rct(100, 8)
    with pytest.raises(MethodIncompatibility, match="cannot be paired"):
        sp.cate_by_group(np.zeros(90), df, by="d")
