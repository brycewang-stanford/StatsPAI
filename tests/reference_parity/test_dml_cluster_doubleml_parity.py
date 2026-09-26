"""Cross-implementation parity: ``sp.dml(cluster=...)`` against DoubleML.

Fixture: ``_fixtures/_generate_dml_cluster_doubleml.py`` (Python DoubleML on
``DoubleMLClusterData``, one cluster variable). Both sides read the same CSV,
use deterministic learners (OLS / unpenalised logit) and the committed
cluster-level folds, so the estimate (the fold-weighted solution of the
linear score) and the Chiang-Kato-Ma-Sasaki (2022) cluster variance are
compared directly for all four models ``sp.dml`` dispatches to.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import DataInsufficient, MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6  # CLAUDE.md §5.1 default budget; observed ~1e-15.
K = 4


def _logit() -> LogisticRegression:
    return LogisticRegression(penalty=None, max_iter=1000)


MODELS: Dict[str, Dict[str, Any]] = {
    "plr": dict(
        model="plr",
        y="y",
        treat="d",
        learners=lambda: dict(ml_g=LinearRegression(), ml_m=LinearRegression()),
    ),
    "irm": dict(
        model="irm",
        y="yb",
        treat="db",
        learners=lambda: dict(ml_g=LinearRegression(), ml_m=_logit()),
    ),
    "pliv": dict(
        model="pliv",
        y="y",
        treat="d",
        instrument="z",
        learners=lambda: dict(
            ml_g=LinearRegression(), ml_m=LinearRegression(), ml_r=LinearRegression()
        ),
    ),
    "iivm": dict(
        model="iivm",
        y="yi",
        treat="di",
        instrument="z",
        learners=lambda: dict(ml_g=LinearRegression(), ml_m=_logit(), ml_r=_logit()),
    ),
}


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "dml_cluster_doubleml.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_dml_cluster_doubleml.py first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "dml_cluster_data.csv")


def _fit(name: str, **extra: Any):
    spec = dict(MODELS[name])
    learners = spec.pop("learners")()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.dml(
            _data(), covariates=["x1", "x2"], n_folds=K, **spec, **learners, **extra
        )


@pytest.mark.parametrize("name", list(MODELS))
def test_cluster_dml_matches_doubleml(name: str) -> None:
    res = _fit(name, cluster="g", fold_indices="fold")
    ref = _ref()[name]
    np.testing.assert_allclose(res.estimate, ref["theta"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["se"], rtol=RTOL)
    assert res.model_info["cluster"] == "g"
    assert res.model_info["n_clusters"] == 60


@pytest.mark.parametrize("name", list(MODELS))
def test_unclustered_path_is_unchanged(name: str) -> None:
    """Without cluster=, the fit is the pre-1.32 i.i.d. one (same folds)."""
    base = _fit(name, fold_indices="fold")
    assert "cluster" not in base.model_info
    clus = _fit(name, cluster="g", fold_indices="fold")
    assert clus.se != pytest.approx(base.se, rel=1e-3)


def test_default_folds_never_split_a_cluster() -> None:
    df = _data()
    est = sp.DoubleML(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        model="plr",
        n_folds=K,
        ml_g=LinearRegression(),
        ml_m=LinearRegression(),
        cluster="g",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = est.fit()
    assert res.model_info["fold_source"] == "cluster_kfold"
    impl = est._impl
    splits = impl._make_splits(df[["x1", "x2"]].to_numpy(), rng_seed=42)
    codes = impl._cluster_codes
    for train, test in splits:
        assert not set(codes[train]) & set(codes[test])


def test_unit_weights_equal_unweighted_and_scale_invariance() -> None:
    df = _data().assign(
        one=1.0, w=lambda d: 1.0 + (d["x2"] > 0), w3=lambda d: 3 * d["w"]
    )
    a = _fit("plr", cluster="g", fold_indices="fold")
    b = _fit("plr", cluster="g", fold_indices="fold", weights=df["one"].to_numpy())
    np.testing.assert_allclose([a.estimate, a.se], [b.estimate, b.se], rtol=1e-12)
    c = _fit("plr", cluster="g", fold_indices="fold", weights=df["w"].to_numpy())
    d = _fit("plr", cluster="g", fold_indices="fold", weights=df["w3"].to_numpy())
    np.testing.assert_allclose([c.estimate, c.se], [d.estimate, d.se], rtol=1e-10)
    assert abs(c.estimate - a.estimate) > 1e-4


def test_cluster_misuse_fails_loudly() -> None:
    df = _data()
    with pytest.raises(MethodIncompatibility, match="split a cluster"):
        _fit("plr", cluster="g", fold_indices=np.arange(len(df)) % K)
    with pytest.raises(MethodIncompatibility, match="not in data"):
        _fit("plr", cluster="nope")
    with pytest.raises(MethodIncompatibility, match="two-way"):
        _fit("plr", cluster=["g", "fold"])
    with pytest.raises(DataInsufficient, match="clusters"):
        _fit("plr", cluster="z")  # two clusters for four folds
