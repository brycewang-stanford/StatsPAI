"""``sp.metalearner(weights=, cluster=)``: the ATE behind ``estimate`` / ``se``.

No reference implementation takes survey weights and clusters for a
meta-learner's AIPW average, so the evidence here is analytic (T1):

* unit weights reproduce the unweighted fit exactly, and weight scale is
  irrelevant;
* singleton clusters reproduce the unclustered SE exactly;
* the weighted estimate equals a hand-built weighted cross-fit AIPW on the
  same folds with the same (weighted) nuisance fits;
* default cluster folds never split a cluster.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    G, m = 50, 10
    n = G * m
    g = np.repeat(np.arange(G), m)
    u = rng.normal(size=G)[g]
    x1 = rng.normal(size=n) + 0.5 * u
    x2 = rng.normal(size=n)
    d = (0.5 * x1 - 0.4 * x2 + rng.normal(size=n) > 0).astype(int)
    y = 1 + (1 + 0.8 * x2) * d + x1 + u + rng.normal(size=n)
    w = np.exp(0.5 * x2 + 0.3 * rng.normal(size=n))
    return pd.DataFrame({"g": g, "x1": x1, "x2": x2, "d": d, "y": y, "w": w})


_DF = _data()


def _fit(learner: str = "t", data: pd.DataFrame = _DF, **extra: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.metalearner(
            data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            learner=learner,
            outcome_model=LinearRegression(),
            propensity_model=LogisticRegression(penalty=None, max_iter=1000),
            n_folds=5,
            **extra,
        )


@pytest.mark.parametrize("learner", ["t", "dr"])
def test_unit_weights_and_weight_scale(learner: str) -> None:
    base = _fit(learner)
    one = _fit(learner, data=_DF.assign(one=1.0), weights="one")
    np.testing.assert_allclose(
        [one.estimate, one.se], [base.estimate, base.se], rtol=1e-10
    )
    a = _fit(learner, weights="w")
    b = _fit(learner, data=_DF.assign(w7=7 * _DF["w"]), weights="w7")
    np.testing.assert_allclose([a.estimate, a.se], [b.estimate, b.se], rtol=1e-10)
    assert abs(a.estimate - base.estimate) > 1e-3


def test_singleton_clusters_reproduce_the_unclustered_se() -> None:
    df = _DF.assign(row=np.arange(len(_DF)))
    folds = np.arange(len(df)) % 5
    a = _fit("t", data=df, fold_indices=folds)
    b = _fit("t", data=df, fold_indices=folds, cluster="row")
    assert a.estimate == b.estimate
    np.testing.assert_allclose(b.se, a.se, rtol=1e-12)


def test_weighted_estimate_is_a_weighted_cross_fit_aipw() -> None:
    folds = np.arange(len(_DF)) % 5
    res = _fit("s", fold_indices=folds, weights="w")
    X = _DF[["x1", "x2"]].to_numpy()
    Y, D = _DF["y"].to_numpy(float), _DF["d"].to_numpy(float)
    w = _DF["w"].to_numpy() * len(_DF) / _DF["w"].sum()
    mu1, mu0, e = np.empty(len(Y)), np.empty(len(Y)), np.empty(len(Y))
    for k in range(5):
        tr, te = folds != k, folds == k
        t1, t0 = tr & (D == 1), tr & (D == 0)
        mu1[te] = (
            LinearRegression().fit(X[t1], Y[t1], sample_weight=w[t1]).predict(X[te])
        )
        mu0[te] = (
            LinearRegression().fit(X[t0], Y[t0], sample_weight=w[t0]).predict(X[te])
        )
        e[te] = (
            LogisticRegression(penalty=None, max_iter=1000)
            .fit(X[tr], D[tr], sample_weight=w[tr])
            .predict_proba(X[te])[:, 1]
        )
    e = np.clip(e, 0.01, 0.99)
    phi = mu1 - mu0 + D * (Y - mu1) / e - (1 - D) * (Y - mu0) / (1 - e)
    np.testing.assert_allclose(res.estimate, np.mean(w * phi), rtol=1e-10)


def test_cluster_folds_and_misuse() -> None:
    res = _fit("dr", cluster="g")
    assert res.model_info["n_clusters"] == 50
    assert res.model_info["se_method"] == "cluster_aipw_influence_function"
    with pytest.raises(MethodIncompatibility, match="split a cluster"):
        _fit("dr", cluster="g", fold_indices=np.arange(len(_DF)) % 5)
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        _fit("t", data=_DF.assign(w=0.0), weights="w")
