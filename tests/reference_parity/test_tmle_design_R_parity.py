"""Cross-language parity: ``sp.tmle(weights=, cluster=)`` against R ``tmle``.

Fixture: ``_generate_tmle_design_R.R`` (tmle::tmle with ``obsWeights=`` and
``id=``, SL.glm nuisances, ``gbound = 0.025``, ``cvQinit = FALSE``); both
sides read ``_fixtures/tmle_design_data.csv``. StatsPAI runs the same glm
learners with ``fluctuation='per_arm'`` and ``q_bound=5e-4``, the settings
under which the unweighted rows of ``test_teffects_R_parity.py`` already
agree. Clusters are all of size 12: tmle averages the influence function
within ``id``, which equals StatsPAI's centred cluster sum with ``G/(G-1)``
exactly when sizes are equal (see ``sp.tmle``'s ``cluster`` docs).
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

from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
# glm convergence leaves ~1e-9; CLAUDE.md §5.1 default budget.
RTOL = 1e-6
CASES: Dict[str, Dict[str, Any]] = {
    "plain": {},
    "cluster": {"cluster": "g"},
    "weights": {"weights": "w"},
    "weights_cluster": {"weights": "w", "cluster": "g"},
}


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "tmle_design_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_tmle_design_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "tmle_design_data.csv")


def _lr() -> LogisticRegression:
    return LogisticRegression(penalty=None, tol=1e-12, max_iter=100000)


def _fit(family: str, data: pd.DataFrame = None, **extra: Any):
    y = "y" if family == "gaussian" else "yb"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.tmle(
            _data() if data is None else data,
            y=y,
            treat="d",
            covariates=["x1", "x2", "x3"],
            outcome_library=[LinearRegression() if family == "gaussian" else _lr()],
            propensity_library=[_lr()],
            fluctuation="per_arm",
            q_bound=5e-4,
            **extra,
        )


@pytest.mark.parametrize("family", ["gaussian", "binomial"])
@pytest.mark.parametrize("case", list(CASES))
def test_tmle_design_matches_r(family: str, case: str) -> None:
    ref = _ref()[f"{family}_{case}"]
    res = _fit(family, **CASES[case])
    np.testing.assert_allclose(res.estimate, ref["psi"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["se"], rtol=RTOL)


def test_weights_move_the_estimate() -> None:
    ref = _ref()
    assert abs(ref["gaussian_weights"]["psi"] - ref["gaussian_plain"]["psi"]) > 0.01


def test_singleton_clusters_reproduce_the_unclustered_se() -> None:
    df = _data().assign(row=np.arange(len(_data())))
    a = _fit("gaussian", data=df)
    b = _fit("gaussian", data=df, cluster="row")
    assert a.estimate == b.estimate
    np.testing.assert_allclose(b.se, a.se, rtol=1e-12)
    assert b.model_info["se_method"] == "cluster_efficient_influence_function"


def test_unsupported_combinations_fail_loudly() -> None:
    with pytest.raises(MethodIncompatibility, match="estimand='ATE'"):
        _fit("gaussian", weights="w", estimand="ATT")
    with pytest.raises(MethodIncompatibility, match="cross-fitted"):
        _fit("gaussian", weights="w", fold_indices=np.arange(len(_data())) % 3)
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        _fit("gaussian", data=_data().assign(w=0.0), weights="w")
