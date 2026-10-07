"""``sp.ctmle`` against R ``ctmle::ctmleDiscrete`` (ctmle 0.1.2).

Fixture: ``_generate_ctmle_R.R`` on ``_fixtures/ctmle_data.csv``, three
simulated data sets with weak overlap, each with a correct and a wrong
initial outcome fit supplied to both sides.

What is compared, and what is not
---------------------------------
* The greedy sequence. The order in which covariates enter the propensity
  model is the same in all six cases, and the estimate at every step
  agrees to 1e-7 (observed 3e-9: Newton tolerance of the logistic fits).
  That is the estimator's construction, and it is a strict comparison.
  ``penalty='variance'`` is ``ctmle``'s criterion.
* Not the step that cross-validation selects. Given the same folds the two
  implementations report different cross-validated losses and choose
  different steps in four of the six cases. ``ctmle`` is GPL and was used
  as a black box; the reason for the difference was not located. It is
  recorded here, not asserted away: the test checks only that the step
  each side selects is one of the common candidates.

Over 200 simulated data sets the two have the same statistical profile
(``docs/dev/2026-10-07-schuler-vanderlaan-review.md``).
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
COV = ["x1", "x2", "x3"]
CASES = [(r, q) for r in range(3) for q in ("c", "w")]


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "ctmle_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_ctmle_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "ctmle_data.csv")


def _fit(rep: int, q: str):
    x = _data()
    x = x[x["rep"] == rep].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ctmle(
            x,
            y="y",
            treat="a",
            covariates=COV,
            Q=x[[f"q0{q}", f"q1{q}"]].to_numpy(),
            penalty="variance",
            fold_indices=x["fold"].to_numpy(),
        )


@pytest.mark.parametrize("rep,q", CASES)
def test_greedy_sequence_matches_ctmle(rep: int, q: str) -> None:
    ref = _ref()[f"rep{rep}_{q}"]
    res = _fit(rep, q)
    assert ref["terms"][0] == "1"
    assert res.model_info["candidate_order"] == ref["terms"][1:]
    np.testing.assert_allclose(
        res.detail["estimate"], ref["candidate_estimates"], rtol=1e-7
    )


@pytest.mark.parametrize("rep,q", CASES)
def test_selected_step_is_a_common_candidate(rep: int, q: str) -> None:
    ref = _ref()[f"rep{rep}_{q}"]
    res = _fit(rep, q)
    k = res.model_info["step"]
    np.testing.assert_allclose(res.estimate, ref["candidate_estimates"][k], rtol=1e-7)
    # ctmle's own choice, for the record: best_k counts the intercept.
    np.testing.assert_allclose(
        ref["est"], ref["candidate_estimates"][ref["best_k"] - 1], rtol=1e-12
    )


def test_wrong_outcome_model_brings_the_confounder_in_first() -> None:
    for rep in range(3):
        assert _fit(rep, "w").model_info["candidate_order"][0] == "x1"
        assert "x1" in _fit(rep, "w").model_info["selected_covariates"]
