"""``sp.dml(cluster=[c1, c2])`` (two-way cluster DML) against DoubleML.

Fixture: ``_fixtures/_generate_dml_twoway_doubleml.py`` (Python DoubleML
0.11.3, BSD-3, on ``DoubleMLClusterData`` with two cluster variables). Both
sides read the same CSV, use deterministic learners and the committed
per-dimension cluster folds (``fold1`` / ``fold2``), so the cell-weighted
score solution and the two-way Chiang-Kato-Ma-Sasaki (2022) variance are
compared directly for the four models ``sp.dml`` dispatches to.
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

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6  # CLAUDE.md §5.1 default budget; observed ~1e-15.
K = 3


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
    return json.loads((_FIX / "dml_twoway_doubleml.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "dml_twoway_data.csv")


def _fit(name: str, **extra: Any):
    spec = dict(MODELS[name])
    learners = spec.pop("learners")()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.dml(
            _data(), covariates=["x1", "x2"], n_folds=K, **spec, **learners, **extra
        )


def _folds() -> np.ndarray:
    return _data()[["fold1", "fold2"]].to_numpy()


@pytest.mark.parametrize("name", list(MODELS))
def test_twoway_cluster_dml_matches_doubleml(name: str) -> None:
    res = _fit(name, cluster=["c1", "c2"], fold_indices=_folds())
    ref = _ref()[name]
    np.testing.assert_allclose(res.estimate, ref["theta"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["se"], rtol=RTOL)
    assert res.model_info["cluster"] == ["c1", "c2"]
    assert res.model_info["n_clusters"] == [30, 24]
    assert res.model_info["n_cells"] == K * K


def test_default_cells_never_share_a_cluster_between_train_and_test() -> None:
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
        cluster=["c1", "c2"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = est.fit()
    assert res.model_info["fold_source"] == "cluster_kfold_twoway"
    impl = est._impl
    splits = impl._make_splits(df[["x1", "x2"]].to_numpy(), rng_seed=42)
    codes = impl._cluster_codes
    tested = np.zeros(len(df), dtype=int)
    for train, test in splits:
        tested[test] += 1
        for j in range(2):
            assert not set(codes[train, j]) & set(codes[test, j])
    assert (tested == 1).all()  # the K x K cells partition the sample


def test_twoway_misuse_fails_loudly() -> None:
    with pytest.raises(MethodIncompatibility, match="more than two"):
        _fit("plr", cluster=["c1", "c2", "z"])
    with pytest.raises(MethodIncompatibility, match="n x 2"):
        _fit("plr", cluster=["c1", "c2"], fold_indices=_folds()[:, 0])
    with pytest.raises(MethodIncompatibility, match="splits"):
        bad = _folds().copy()
        bad[0, 0] = (bad[0, 0] + 1) % K
        _fit("plr", cluster=["c1", "c2"], fold_indices=bad)
    with pytest.raises(DataInsufficient, match="clusters"):
        _fit("plr", cluster=["c1", "z"])  # two z-clusters for three folds
