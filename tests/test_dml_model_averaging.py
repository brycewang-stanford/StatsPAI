"""Tests for Model Averaging DML (Ahrens et al. JAE 2025)."""

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def _synth_dml_data(n=400, p=8, theta=1.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    D = X[:, :3].sum(1) + rng.normal(0, 1.0, n)
    Y = theta * D + np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "y": Y,
            "d": D,
            **{f"x{j}": X[:, j] for j in range(p)},
        }
    )


def test_averaging_recovers_theta():
    df = _synth_dml_data(seed=1)
    r = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(8)],
        n_folds=5,
        seed=1,
    )
    assert abs(r.estimate - 1.5) < 0.5
    assert r.se >= 0


def test_weight_rules():
    df = _synth_dml_data(seed=2, n=300)
    cov = [f"x{j}" for j in range(8)]
    for rule in ("inverse_risk", "equal", "single_best"):
        r = sp.dml_model_averaging(
            df, y="y", treat="d", covariates=cov, n_folds=5, seed=2, weight_rule=rule
        )
        assert r.estimate == r.estimate  # not NaN
        total_w = sum(r.model_info["weights"].values())
        assert abs(total_w - 1.0) < 1e-6

    # Single-best should put all mass on one candidate
    r_best = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=cov,
        n_folds=5,
        seed=2,
        weight_rule="single_best",
    )
    weights = list(r_best.model_info["weights"].values())
    assert max(weights) == pytest.approx(1.0)
    assert sorted(weights)[-2] == pytest.approx(0.0)


def test_custom_candidates():
    from sklearn.linear_model import LassoCV

    df = _synth_dml_data(seed=3, n=200)
    cand = [
        (LassoCV(cv=3), LassoCV(cv=3), "lasso_a"),
        (LassoCV(cv=3), LassoCV(cv=3), "lasso_b"),
    ]
    r = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(8)],
        candidates=cand,
        n_folds=3,
        seed=3,
    )
    assert set(r.model_info["candidates"]) == {"lasso_a", "lasso_b"}


def test_invalid_weight_rule_raises():
    df = _synth_dml_data(seed=4, n=100)
    with pytest.raises(ValueError):
        sp.dml_model_averaging(
            df, y="y", treat="d", covariates=["x0"], weight_rule="bogus"
        )


def test_registered_in_public_api():
    fns = sp.list_functions()
    assert "dml_model_averaging" in fns
    assert "model_averaging_dml" in fns


def test_se_on_correct_scale():
    """Regression test for the v1.5.1 √n scaling bug.

    Before the fix, the CI width was ~√n × too wide (≈4.2 for n=400 instead
    of ~0.2). After the fix, on the canonical DGP with true θ=1.5 and
    n=400, a single draw should give a CI width well under 1.0.
    """
    df = _synth_dml_data(seed=42, n=400)
    r = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=[f"x{j}" for j in range(8)],
        n_folds=3,
        seed=42,
    )
    width = r.ci[1] - r.ci[0]
    assert (
        width < 1.0
    ), f"CI width {width:.3f} suggests the √n scaling bug has regressed"
    # Point estimate shouldn't have drifted wildly
    assert abs(r.estimate - 1.5) < 0.5


# --- cluster= and n_rep= (the paths sp.dml has and the 1.45 short-stacking
# default needs). The reference is sp.dml itself: with one candidate the two
# functions run the same cross-fitting on the same folds, so they must agree.


def _clustered_data(seed: int = 3, n_clusters: int = 60, size: int = 8):
    rng = np.random.default_rng(seed)
    n = n_clusters * size
    g = np.repeat(np.arange(n_clusters), size)
    u = rng.normal(size=n_clusters)[g]
    X = rng.normal(size=(n, 4))
    d = X[:, 0] + 0.5 * u + rng.normal(size=n)
    y = 1.0 * d + np.sin(X[:, 1]) + u + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["d"], df["y"], df["g"] = d, y, g
    df["w"] = rng.uniform(0.5, 2.0, size=n)
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


_XS = ["x1", "x2", "x3", "x4"]


@pytest.mark.parametrize(
    "extra",
    [
        {"cluster": "g"},
        {"n_rep": 3},
        {"cluster": "g", "n_rep": 3},
        {"cluster": "g", "sample_weight": "w"},
    ],
    ids=["cluster", "n_rep", "cluster+n_rep", "cluster+weights"],
)
def test_one_candidate_equals_sp_dml(extra):
    from sklearn.linear_model import RidgeCV

    df = _clustered_data()
    ref = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=_XS,
        ml_g=RidgeCV(),
        ml_m=RidgeCV(),
        n_folds=4,
        random_state=7,
        **extra,
    )
    got = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=_XS,
        candidates=[(RidgeCV(), RidgeCV(), "ridge")],
        n_folds=4,
        seed=7,
        **extra,
    )
    # same folds, same learner, same formulas: agreement to rounding
    assert got.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert got.se == pytest.approx(ref.se, rel=1e-12)


def test_cluster_equal_weights_of_identical_candidates_is_one_candidate():
    """The cross-candidate cluster covariance: two copies of one candidate
    under equal weights have the variance of that candidate."""
    from sklearn.linear_model import RidgeCV

    df = _clustered_data()
    one = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=_XS,
        cluster="g",
        n_folds=4,
        seed=7,
        candidates=[(RidgeCV(), RidgeCV(), "a")],
    )
    two = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=_XS,
        cluster="g",
        n_folds=4,
        seed=7,
        weight_rule="equal",
        candidates=[(RidgeCV(), RidgeCV(), "a"), (RidgeCV(), RidgeCV(), "b")],
    )
    assert two.estimate == pytest.approx(one.estimate, rel=1e-12)
    assert two.se == pytest.approx(one.se, rel=1e-12)


def test_cluster_folds_keep_clusters_whole_and_are_recorded():
    from sklearn.linear_model import RidgeCV

    df = _clustered_data()
    r = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=_XS,
        cluster="g",
        n_folds=4,
        seed=7,
        candidates=[(RidgeCV(), RidgeCV(), "a")],
    )
    assert r.model_info["cluster"] == "g"
    assert r.model_info["n_clusters"] == 60
    assert r.model_info["fold_source"] == "cluster_kfold"


def test_n_rep_aggregation_is_the_median_rule():
    from sklearn.linear_model import RidgeCV

    df = _clustered_data()
    kw = dict(
        y="y",
        treat="d",
        covariates=_XS,
        n_folds=4,
        candidates=[(RidgeCV(), RidgeCV(), "a")],
    )
    reps = [sp.dml_model_averaging(df, seed=7 + r, **kw) for r in range(3)]
    agg = sp.dml_model_averaging(df, seed=7, n_rep=3, **kw)
    thetas = np.array([r.estimate for r in reps])
    ses = np.array([r.se for r in reps])
    theta = np.median(thetas)
    assert agg.estimate == pytest.approx(theta, rel=1e-12)
    assert agg.se == pytest.approx(
        np.sqrt(np.median(ses**2 + (thetas - theta) ** 2)), rel=1e-12
    )
    assert agg.model_info["n_rep"] == 3
    assert agg.model_info["theta_reps"] == pytest.approx(thetas.tolist())
    assert "_y_resid" not in agg.model_info


def test_cluster_and_n_rep_refuse_what_they_cannot_do():
    from statspai.exceptions import DataInsufficient, MethodIncompatibility

    df = _clustered_data()
    kw = dict(y="y", treat="d", covariates=_XS)
    with pytest.raises(MethodIncompatibility, match="two-way"):
        sp.dml_model_averaging(df, cluster=["g", "x1"], **kw)
    with pytest.raises(MethodIncompatibility, match="n_rep=1"):
        sp.dml_model_averaging(df, n_rep=2, fold_indices=np.arange(len(df)) % 3, **kw)
    with pytest.raises(MethodIncompatibility, match="split a cluster"):
        sp.dml_model_averaging(
            df, cluster="g", fold_indices=np.arange(len(df)) % 3, **kw
        )
    few = df[df["g"] < 3]
    with pytest.raises(DataInsufficient, match="3 clusters"):
        sp.dml_model_averaging(few, cluster="g", n_folds=5, **kw)
