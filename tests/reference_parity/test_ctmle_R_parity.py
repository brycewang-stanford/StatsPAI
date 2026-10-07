"""``sp.ctmle`` against R ``ctmle::ctmleDiscrete`` (ctmle 0.1.2).

Fixture: ``_generate_ctmle_R.R`` on ``_fixtures/ctmle_data.csv``, three
simulated data sets with weak overlap, each with a correct and a wrong
initial outcome fit supplied to both sides.

What is compared, and what is not
---------------------------------
* The greedy sequence. The order in which covariates enter the propensity
  model is the same in all six cases, and the estimate at every step
  agrees to 1e-7 (observed 5e-9: Newton tolerance of the logistic fits).
  That is the estimator's construction, and it is a strict comparison.
  ``penalty='search'`` is ``ctmleDiscrete``'s default: the variance of the
  influence function is added in the greedy search, and the
  cross-validation compares residual sums of squares.
* The pre-ordered sequence (``order=``, R ``preOrder = TRUE``), six more
  cases. Four agree to 1e-9. In the two where the instrument is offered
  first, the steps after it differ by up to 5.4e-6 in relative terms. The
  cause was not located (the logistic fits agree to 1e-13 and the rule for
  restarting from the current fit was varied without closing it), so
  those steps are held to 1e-5 and the row is not called strict.
* Not the step that cross-validation selects, for a reason that was
  located: ``ctmleDiscrete`` ignores its ``folds`` argument. Its results
  depend on the random seed and are identical whatever partition is
  passed, so the two sides never score the same partition. Over 60 random
  partitions of one data set the cross-validated sums of squares of the
  two have the same mean at every step (differences of 0.1 to 1.1
  standard errors), and over 200 data sets the estimates have the same
  bias and spread and the same average selected step
  (``docs/dev/2026-10-07-schuler-vanderlaan-review.md``). That is a
  statistical screen, not a parity row.

``ctmle`` is GPL and was used as a black box.
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


def _fit(rep: int, q: str, **extra: Any):
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
            penalty="search",
            fold_indices=x["fold"].to_numpy(),
            **extra,
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


_ORDERS = [("x1", "x2", "x3"), ("x2", "x3", "x1"), ("x3", "x1", "x2")]


@pytest.mark.parametrize("q", ["c", "w"])
@pytest.mark.parametrize("order", _ORDERS)
def test_pre_ordered_sequence_matches_ctmle(q: str, order: tuple) -> None:
    res = _fit(0, q, order=list(order))
    assert res.model_info["search"] == "pre-ordered"
    assert res.model_info["candidate_order"] == list(order)
    # 1e-9 where the instrument x2 is not offered first; see the module
    # docstring for the two sequences where it is.
    rtol = 1e-5 if order[0] == "x2" else 1e-9
    ref = _ref()["preordered"][q + "_" + "_".join(order)]
    # ctmle may stop extending a pre-ordered sequence early; compare the
    # steps it reports.
    m = len(ref["candidate_estimates"])
    assert m >= 2 and ref["terms"][1:] == list(order)[: m - 1]
    np.testing.assert_allclose(
        res.detail["estimate"].iloc[:m], ref["candidate_estimates"], rtol=rtol
    )
