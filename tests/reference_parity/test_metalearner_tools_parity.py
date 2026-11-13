"""``sp.predict_cate`` and ``sp.compare_metalearners`` against ``econml`` and ``DoubleML``.

Same data (``_fixtures/metalearner_train.csv``, 1,200 rows;
``metalearner_new.csv``, 200 rows for prediction), the same base learners
and the same five cross-fitting folds on every side.

References:

* ``econml`` 0.16.0 (MIT) ``SLearner`` / ``TLearner`` / ``XLearner`` with
  ridge outcome models (``alpha = 50``, so that the T- and X-learner are
  different functions), least-squares effect models and a Newton logistic
  propensity (``_generate_metalearner_econml.py`` ->
  ``metalearner_econml.json``). Compared: the effect predicted at the 200
  new rows (``sp.predict_cate``) and the spread of the in-sample effects
  that ``sp.compare_metalearners`` reports (``cate_std``, ``cate_iqr``).
  Tolerance 1e-10 absolute (observed 1e-15).
* R ``DoubleML`` 1.0.2 ``DoubleMLIRM(score = "ATE")`` with ``regr.lm`` and
  ``classif.log_reg`` on the stored folds, trimming at 0.01
  (``_generate_metalearner_doubleml_R.R`` ->
  ``metalearner_doubleml_R.json``). Compared: the average effect and
  standard error ``sp.compare_metalearners`` reports for every learner,
  which is the cross-fitted AIPW estimate. Tolerance 1e-9 relative
  (observed 1e-15).

Two conventions, each pinned by a computation in this file:

* Standard error divisor. DoubleML reports ``sqrt(mean(psi^2) / n)``,
  StatsPAI ``sd(psi, ddof = 1) / sqrt(n)``; they differ by
  ``sqrt(n / (n - 1))``.
* S-learner with a penalised base learner. econml appends both treatment
  indicators ``[1 - D, D]`` to the covariates, StatsPAI appends ``D``.
  With least squares the two are the same function; under a ridge penalty
  they are not (0.8817 against 0.8129 here), and neither is wrong.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
ECONML = json.loads((_FIX / "metalearner_econml.json").read_text(encoding="utf-8"))
DML = json.loads((_FIX / "metalearner_doubleml_R.json").read_text(encoding="utf-8"))
TRAIN = pd.read_csv(_FIX / "metalearner_train.csv")
NEW = pd.read_csv(_FIX / "metalearner_new.csv")
COV = ["x1", "x2", "x3"]
_spec = importlib.util.spec_from_file_location(
    "_metalearner_learners", _FIX / "_metalearner_learners.py"
)
_learners = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_learners)


def _ridge():
    return Ridge(alpha=50.0, solver="cholesky")


def _common(outcome_model):
    return dict(
        outcome_model=outcome_model,
        propensity_model=_learners.NewtonLogit(),
        cate_model=LinearRegression(),
        fold_indices=TRAIN["fold"].to_numpy(),
    )


def _fit(learner, outcome_model):
    return sp.metalearner(
        TRAIN,
        y="y",
        treat="d",
        covariates=COV,
        learner=learner,
        **_common(outcome_model),
    )


def test_fixtures_are_the_reference_outputs():
    assert ECONML["versions"]["econml"] == "0.16.0"
    assert DML["versions"]["DoubleML"] == "1.0.2"
    assert DML["n"] == len(TRAIN) == 1200
    # the T- and X-learner references are different functions
    t = np.asarray(ECONML["cases"]["t"]["effect_new"])
    x = np.asarray(ECONML["cases"]["x"]["effect_new"])
    assert np.abs(t - x).max() > 0.05


@pytest.mark.parametrize(
    "case,learner,outcome",
    [("s_ols", "s", "ols"), ("t", "t", "ridge"), ("x", "x", "ridge")],
)
def test_predict_cate_matches_econml_effect_on_new_rows(case, learner, outcome):
    model = LinearRegression() if outcome == "ols" else _ridge()
    pred = sp.predict_cate(_fit(learner, model), NEW)
    ref = np.asarray(ECONML["cases"][case]["effect_new"])
    assert pred.shape == (200,)
    np.testing.assert_allclose(pred, ref, rtol=0, atol=1e-10)


def test_compare_metalearners_cate_spread_matches_econml():
    table = sp.compare_metalearners(
        TRAIN, "y", "d", COV, learners=["t", "x"], **_common(_ridge())
    ).set_index("learner")
    for name, row in (("t", "T-Learner"), ("x", "X-Learner")):
        ref = ECONML["cases"][name]
        assert table.loc[row, "cate_std"] == pytest.approx(ref["cate_std"], abs=1e-10)
        assert table.loc[row, "cate_iqr"] == pytest.approx(ref["cate_iqr"], abs=1e-10)


def test_compare_metalearners_average_effect_matches_doubleml_irm():
    table = sp.compare_metalearners(
        TRAIN,
        "y",
        "d",
        COV,
        learners=["s", "t", "x", "r", "dr"],
        **_common(LinearRegression()),
    )
    n = len(TRAIN)
    for _, row in table.iterrows():
        assert row["se_method"] == "aipw_influence_function"
        assert row["ate"] == pytest.approx(DML["ate"], rel=1e-9)
        # divisor convention: n - 1 here, n in DoubleML
        assert row["se"] * np.sqrt((n - 1) / n) == pytest.approx(DML["se"], rel=1e-9)
        assert row["ci_lower"] == pytest.approx(
            row["ate"] - 1.959963984540054 * row["se"], rel=1e-9
        )


def test_penalised_s_learner_differs_from_econml_by_the_treatment_encoding():
    X = TRAIN[COV].to_numpy()
    d = TRAIN["d"].to_numpy()
    y = TRAIN["y"].to_numpy()
    both = _ridge().fit(np.column_stack([X, 1 - d, d]), y)
    one = _ridge().fit(np.column_stack([X, d]), y)
    # econml: both indicators enter the ridge fit
    assert both.coef_[4] - both.coef_[3] == pytest.approx(
        ECONML["cases"]["s"]["effect_new"][0], abs=1e-10
    )
    # StatsPAI: the treatment enters once
    pred = sp.predict_cate(_fit("s", _ridge()), NEW)
    np.testing.assert_allclose(pred, one.coef_[3], rtol=0, atol=1e-10)
