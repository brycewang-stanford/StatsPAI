"""Cross-implementation parity: ``sp.dml`` options against DoubleML.

Track A modules 08 and 71 pin ``sp.dml`` against R DoubleML at five
folds, one repetition and the default score and propensity processing of
each model. This file covers the options around those cells, against
doubleml-for-py 0.11.3 on the same CSV and the same folds
(``_fixtures/_generate_dml_options_doubleml.py``):

* ``n_folds`` 2, 3 and 10 for all four models;
* ``score='IV-type'`` for the partially linear model, and ``'ATTE'`` for
  the interactive model;
* propensity processing other than the registered defaults: trimming at
  0.1 (it binds on this design: some propensities are outside
  (0.1, 0.9)) and ``normalize_ipw=True``, for IRM (ATE, ATTE) and IIVM;
* ``n_rep=3`` with the folds ``sp.dml`` draws itself.

Learners are OLS and unpenalised logit, so with the folds shared the
estimate and standard error agree to floating point (observed 7e-16).

Repeated cross-fitting
----------------------
Every repetition's estimate and standard error agree with both DoubleML
ports (Python, and R DoubleML 1.0.2 via ``_generate_dml_options_R.R``), and
so does the aggregated estimate, the median over repetitions. The
aggregated standard error is a convention on which the two DoubleML ports
themselves disagree, by 1.8e-3 relative on this sample:

* R DoubleML:      ``se^2 = median(n se_r^2 + (theta_r - theta_med)^2) / n``
* Python DoubleML: ``se = (median(theta_r + 1.96 se_r) - theta_med) / 1.96``
* ``sp.dml``:      ``se^2 = median(se_r^2 + (theta_r - theta_med)^2)``

The R rule adds the squared deviation to the *asymptotic* variance
``n se_r^2`` without scaling it, so the sample-splitting term enters with
weight 1/n; ``sp.dml`` adds it to the variance of the estimate, where
both terms are of the same order. Each reference's number is
reconstructed here from the shared repetitions to 1e-12, so the gap is
fully accounted for; it is registered as a disclosure (T4), not parity.
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

_FIX = pathlib.Path(__file__).parent / "_fixtures"
#: CLAUDE.md section 5.1 budget; observed 7e-16 on every comparison.
RTOL = 1e-6
COVARIATES = ["x1", "x2", "x3", "x4"]


def _logit() -> LogisticRegression:
    return LogisticRegression(penalty=None, max_iter=1000)


MODELS: Dict[str, Dict[str, Any]] = {
    "plr": dict(
        y="y",
        treat="d",
        learners=lambda: dict(ml_g=LinearRegression(), ml_m=LinearRegression()),
    ),
    "irm": dict(
        y="yb",
        treat="db",
        learners=lambda: dict(ml_g=LinearRegression(), ml_m=_logit()),
    ),
    "pliv": dict(
        y="yc",
        treat="dc",
        instrument="z",
        learners=lambda: dict(
            ml_g=LinearRegression(), ml_m=LinearRegression(), ml_r=LinearRegression()
        ),
    ),
    "iivm": dict(
        y="yi",
        treat="di",
        instrument="zb",
        learners=lambda: dict(ml_g=LinearRegression(), ml_m=_logit(), ml_r=_logit()),
    ),
}


@lru_cache(maxsize=None)
def _py() -> Dict[str, Any]:
    return json.loads((_FIX / "dml_options_doubleml.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _r() -> Dict[str, Any]:
    return json.loads((_FIX / "dml_options_R.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "dml_options_data.csv")


@lru_cache(maxsize=None)
def _fit(key: str):
    ref = _py()[key]
    spec = dict(MODELS[ref["model"]])
    learners = spec.pop("learners")()
    kw: Dict[str, Any] = dict(model=ref["model"], covariates=COVARIATES, **spec)
    if ref["model"] in ("plr", "irm"):
        kw["score"] = ref["score"]
    for option in ("trimming_threshold", "normalize_ipw"):
        if option in ref:
            kw[option] = ref[option]
    df = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ref["n_rep"] > 1:
            return sp.dml(df, n_folds=5, n_rep=ref["n_rep"], **kw, **learners)
        folds = df[ref["folds"]].to_numpy()
        return sp.dml(
            df,
            n_folds=len(np.unique(folds)),
            fold_indices=folds,
            **kw,
            **learners,
        )


SINGLE = sorted(
    k for k, v in _py().items() if not k.startswith("_") and v["n_rep"] == 1
)
REPEATED = sorted(
    k for k, v in _py().items() if not k.startswith("_") and v["n_rep"] > 1
)


def test_fixture_covers_the_intended_cells():
    cells = {(_py()[k]["model"], _py()[k]["folds"]) for k in SINGLE}
    for model in MODELS:
        for k in (2, 3, 10):
            assert (model, f"f{k}") in cells, (model, k)
    assert {"plr_po_rep3", "plr_ivtype_rep3", "irm_ate_rep3"} <= set(REPEATED)
    assert _py()["_meta"]["doubleml_version"] == "0.11.3"


def test_trimming_and_normalisation_move_the_estimate():
    """Otherwise the 'other' propensity cells would test the default again."""
    ref = _py()
    base_irm, base_iivm = ref["irm_ate_k3"]["theta"], ref["iivm_k3"]["theta"]
    for key in ("irm_ate_trim0.1_k3", "irm_ate_normalized_k3"):
        assert abs(ref[key]["theta"] / base_irm - 1) > 1e-3, key
    for key in ("iivm_trim0.1_k3", "iivm_normalized_k3"):
        assert abs(ref[key]["theta"] / base_iivm - 1) > 1e-3, key


@pytest.mark.parametrize("key", SINGLE)
def test_estimate_and_se_match_doubleml(key):
    ref = _py()[key]
    res = _fit(key)
    np.testing.assert_allclose(res.estimate, ref["theta"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["se"], rtol=RTOL)
    mi = res.model_info
    assert mi["fold_source"] == "user"
    assert mi["n_folds"] == int(ref["folds"][1:])


@pytest.mark.parametrize("key", REPEATED)
def test_internal_folds_are_the_committed_ones(key):
    """The repetitions use sp.dml's own folds; the fixture recorded them."""
    from sklearn.model_selection import KFold, StratifiedKFold

    df = _data()
    ref = _py()[key]
    X = df[COVARIATES].to_numpy()
    for r in range(ref["n_rep"]):
        if ref["folds"] == "kfold":
            splits = KFold(5, shuffle=True, random_state=42 + r).split(X)
        else:
            splits = StratifiedKFold(5, shuffle=True, random_state=42 + r).split(
                X, df["db"].to_numpy()
            )
        labels = np.empty(len(df), dtype=int)
        for fold, (_, test) in enumerate(splits):
            labels[test] = fold
        np.testing.assert_array_equal(labels, df[f"rep{r}_{ref['folds']}"])
    assert _fit(key).model_info["fold_source"] != "user"


@pytest.mark.parametrize("key", REPEATED)
def test_every_repetition_and_the_median_match_doubleml(key):
    ref = _py()[key]
    mi = _fit(key).model_info
    np.testing.assert_allclose(mi["theta_all_reps"], ref["all_theta"], rtol=RTOL)
    np.testing.assert_allclose(mi["se_all_reps"], ref["all_se"], rtol=RTOL)
    np.testing.assert_allclose(_fit(key).estimate, ref["theta"], rtol=RTOL)
    if key in _r():
        rr = _r()[key]
        np.testing.assert_allclose(mi["theta_all_reps"], rr["all_theta"], rtol=RTOL)
        np.testing.assert_allclose(mi["se_all_reps"], rr["all_se"], rtol=RTOL)
        np.testing.assert_allclose(_fit(key).estimate, rr["theta"], rtol=RTOL)


def _rules(theta: np.ndarray, se: np.ndarray, n: int) -> Dict[str, float]:
    med = float(np.median(theta))
    return {
        "statspai": float(np.sqrt(np.median(se**2 + (theta - med) ** 2))),
        "r": float(np.sqrt(np.median(n * se**2 + (theta - med) ** 2) / n)),
        "python": float((np.median(theta + 1.96 * se) - med) / 1.96),
    }


@pytest.mark.parametrize("key", REPEATED)
def test_aggregated_se_is_accounted_for_under_each_rule(key):
    """Each implementation's SE is its own rule applied to the same repetitions."""
    res = _fit(key)
    theta = np.asarray(res.model_info["theta_all_reps"])
    se = np.asarray(res.model_info["se_all_reps"])
    rules = _rules(theta, se, len(_data()))
    assert res.se == pytest.approx(rules["statspai"], rel=1e-12)
    assert _py()[key]["se"] == pytest.approx(rules["python"], rel=1e-12)
    if key in _r():
        assert _r()[key]["se"] == pytest.approx(rules["r"], rel=1e-12)
        # the two DoubleML ports disagree with each other beyond the budget
        assert abs(_r()[key]["se"] / _py()[key]["se"] - 1) > 1e-4


def test_repetitions_widen_the_interval_only_through_split_dispersion():
    """sp.dml's SE is at least the median per-repetition SE (equal if no spread)."""
    res = _fit("plr_po_rep3")
    se = np.asarray(res.model_info["se_all_reps"])
    assert res.se >= float(np.median(se))
    theta = np.asarray(res.model_info["theta_all_reps"])
    assert np.ptp(theta) > 0  # the repetitions really differ
