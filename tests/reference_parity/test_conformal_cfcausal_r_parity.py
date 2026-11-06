"""Weighted split conformal inference against R ``cfcausal``.

``cfcausal`` (Lei and Candes; MIT licence, commit e4ca670) is the authors'
implementation of Lei & Candes (2021) [@lei2021conformal]. The reference
numbers are in ``_fixtures/conformal_cfcausal_R.json``, written by
``_fixtures/_generate_conformal_cfcausal_R.R`` on the data and splits of
``_fixtures/_generate_conformal_cfcausal_data.py``. ``conformalInference``
(GPL-2, run as a black box) gives a second opinion on the generic weighted
procedure of Tibshirani et al. (2019) [@tibshirani2019conformal].

Everything is deterministic: the splits are the rows StatsPAI draws for the
seeds used here (supplied to R as ``trainid``), the outcome learner is least
squares on both sides (``LinearRegression`` / ``lm``), and the propensity is
the known ``e(x)`` of the data, so the weights are the same numbers. R's
``predict.conformalSplit`` is called with ``wthigh = Inf, wtlow = 0`` (its
defaults truncate the normalised weights to [0.05, 20], which StatsPAI does
not do) and ``useInf = TRUE`` where StatsPAI returns an infinite band.

What is compared, all to 1e-12 (observed <= 3e-15):

* ``sp.weighted_conformal_prediction``: ``cfcausal::conformal`` +
  ``predict``, unweighted and with density-ratio weights including the test
  point's own weight (one test point gets an infinite band);
  ``conformalInference::conformal.pred.split`` agrees.
* ``sp.conformal_counterfactual``: per-arm bands with weights ``1/e(x)`` and
  ``1/(1-e(x))``, which is ``conformalCf(estimand = "unconditional")`` run on
  the supplied split (the steps of ``conformalCf_split``).
* ``sp.conformal_ite_interval``: both bands at ``alpha/2`` combined as
  ``[L1 - U0, U1 - L0]``, the rule of ``cfcausal``'s naive ITE method. That
  method weights for the population whose outcome is missing
  (``(1-e)/e``); StatsPAI targets the whole population (``1/e``), so the
  reference is run with the unconditional weights.
* ``sp.conformal_regression(method="split", newdata=...)``.
* ``sp.conformal_cate``: in a randomized design the weights are constant,
  so its unweighted per-arm bands equal the reference's. Two of the four
  cases put the conformal rank below the calibration size, where the
  quantile used to be interpolated (fixed for 1.43).
* ``sp.conformal_continuous``: plain split conformal on (dose, covariates).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression

import statspai as sp

FX = Path(__file__).parent / "_fixtures"
REF = json.loads((FX / "conformal_cfcausal_R.json").read_text(encoding="utf-8"))
SPEC = json.loads((FX / "conformal_cfcausal_spec.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(FX / "conformal_cfcausal_data.csv")
COLS = ["x1", "x2"]
TOL = 1e-12


def _dec(values) -> np.ndarray:
    return np.array([float(v) for v in values])


def _same(got, ref) -> None:
    """Equal to 1e-12 relative (absolute below 1); infinities in the same places."""
    got = np.asarray(got, dtype=float)
    ref = _dec(ref)
    assert got.shape == ref.shape
    np.testing.assert_array_equal(np.isposinf(got), np.isposinf(ref))
    np.testing.assert_array_equal(np.isneginf(got), np.isneginf(ref))
    fin = np.isfinite(ref)
    np.testing.assert_allclose(got[fin], ref[fin], rtol=TOL, atol=TOL)


class KnownPropensity(ClassifierMixin, BaseEstimator):
    """The data's propensity e(x) = plogis(a + b x1); fit is a no-op."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        x1 = np.asarray(X, dtype=float)[:, 0]
        e = 1.0 / (1.0 + np.exp(-(SPEC["ps_a"] + SPEC["ps_b"] * x1)))
        return np.column_stack([1.0 - e, e])


# ---------------------------------------------------------------------------
# weighted_conformal_prediction
# ---------------------------------------------------------------------------

_W = DATA[DATA.block == "wcp"]
_TR, _CA, _TE = (_W[_W.role == r] for r in ("train", "calib", "test"))


def _wt(d: pd.DataFrame) -> np.ndarray:
    return np.exp(SPEC["shift"] * d["x1"].to_numpy())


def _wcp(alpha: float, weighted: bool):
    return sp.weighted_conformal_prediction(
        _TR[COLS].to_numpy(),
        _TR["y"].to_numpy(),
        _CA[COLS].to_numpy(),
        _CA["y"].to_numpy(),
        _TE[COLS].to_numpy(),
        weights_calib=_wt(_CA) if weighted else None,
        weights_test=_wt(_TE) if weighted else None,
        model=LinearRegression(),
        alpha=alpha,
    )


@pytest.mark.parametrize(
    "key,alpha,weighted",
    [
        ("plain_a10", 0.10, False),
        ("plain_a20", 0.20, False),
        ("weighted_a10", 0.10, True),
        ("weighted_a05", 0.05, True),
    ],
)
def test_weighted_conformal_prediction_equals_cfcausal(key, alpha, weighted):
    lower, upper, point = _wcp(alpha, weighted)
    ref = REF["wcp"][key]
    _same(lower, ref["lower"])
    _same(upper, ref["upper"])
    _same(point, ref["point"])


def test_weighted_cases_include_an_infinite_band():
    """The last test point (x1 = 4) carries more than alpha of the weight."""
    assert _dec(REF["wcp"]["weighted_a10"]["upper"])[-1] == np.inf
    lower, upper, _ = _wcp(0.10, True)
    assert np.isinf(upper[-1]) and np.isinf(lower[-1])
    assert np.isfinite(upper[:-1]).all()


@pytest.mark.parametrize("key,weighted", [("plain_a10", False), ("weighted_a10", True)])
def test_weighted_conformal_prediction_equals_conformal_inference(key, weighted):
    lower, upper, _ = _wcp(0.10, weighted)
    ref = REF["wcp_conformalInference"][key]
    _same(lower, ref["lower"])
    _same(upper, ref["upper"])


# ---------------------------------------------------------------------------
# conformal_counterfactual / conformal_ite_interval
# ---------------------------------------------------------------------------

_CF = DATA[(DATA.block == "cf") & (DATA.role == "data")].copy()
_CF["t"] = _CF["t"].astype(int)
_CFT = DATA[(DATA.block == "cf") & (DATA.role == "test")][COLS].to_numpy()


def _cf_args(key: str) -> dict:
    seed, frac = key.split("_")
    return dict(
        y="y",
        treat="t",
        covariates=COLS,
        X_test=_CFT,
        alpha=0.10,
        calib_frac=float(frac),
        model=LinearRegression(),
        propensity_model=KnownPropensity(),
        random_state=int(seed),
    )


@pytest.mark.parametrize("key", sorted(SPEC["cf"]))
def test_conformal_counterfactual_equals_cfcausal(key):
    res = sp.conformal_counterfactual(_CF, **_cf_args(key))
    ref = REF["cf"][key]
    _same(res.lower_Y1, ref["Y1"]["lower"])
    _same(res.upper_Y1, ref["Y1"]["upper"])
    _same(res.point_Y1, ref["Y1"]["point"])
    _same(res.lower_Y0, ref["Y0"]["lower"])
    _same(res.upper_Y0, ref["Y0"]["upper"])
    _same(res.point_Y0, ref["Y0"]["point"])
    # the last test point has e(x) = 0.024: no finite band for Y(1)
    assert np.isinf(res.upper_Y1[-1]) and np.isfinite(res.upper_Y0[-1])


@pytest.mark.parametrize("key", sorted(SPEC["cf"]))
def test_conformal_ite_interval_equals_cfcausal_naive_rule(key):
    res = sp.conformal_ite_interval(_CF, **_cf_args(key))
    ref = REF["cf"][key]["ite"]
    _same(res.lower, ref["lower"])
    _same(res.upper, ref["upper"])
    _same(res.point, ref["point"])


# ---------------------------------------------------------------------------
# conformal_regression(method="split")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(SPEC["reg"]))
def test_conformal_regression_split_equals_cfcausal(key):
    seed, frac = key.split("_")
    out = sp.conformal_regression(
        "y ~ x1 + x2",
        pd.concat([_TR, _CA]),
        method="split",
        newdata=_TE,
        alpha=0.10,
        train_frac=float(frac),
        seed=int(seed),
    )
    ref = REF["reg"][key]
    _same(out["yhat"], ref["point"])
    _same(out["lower"], ref["lower"])
    _same(out["upper"], ref["upper"])


# ---------------------------------------------------------------------------
# conformal_cate (randomized design)
# ---------------------------------------------------------------------------

_RCT = DATA[DATA.block == "rct"].copy()
_RCT["t"] = _RCT["t"].astype(int)


@pytest.mark.parametrize("key", sorted(SPEC["cate"]))
def test_conformal_cate_bands_equal_cfcausal(key):
    seed, frac, alpha = key.split("_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.conformal_cate(
            _RCT,
            y="y",
            treat="t",
            covariates=COLS,
            model=LinearRegression(),
            alpha=float(alpha),
            calib_fraction=float(frac),
            random_state=int(seed),
        )
    ref = REF["cate"][key]
    _same(res.model_info["cate"], ref["cate"])
    _same(res.model_info["cate_lower"], ref["lower"])
    _same(res.model_info["cate_upper"], ref["upper"])


def test_some_cate_cases_have_the_rank_below_the_calibration_size():
    """Guards the coverage of the fixture: the quantile fix is exercised."""
    below = 0
    for key, s in SPEC["cate"].items():
        alpha = float(key.split("_")[2]) / 2
        for arm in ("0", "1"):
            n = len(s[arm]["calib"])
            below += int(np.ceil((1 - alpha) * (n + 1))) < n
    assert below >= 2


# ---------------------------------------------------------------------------
# conformal_continuous
# ---------------------------------------------------------------------------

_DOSE = DATA[(DATA.block == "dose") & (DATA.role == "data")].reset_index(drop=True)
_DOSE_T = DATA[(DATA.block == "dose") & (DATA.role == "test")].reset_index(drop=True)


@pytest.mark.parametrize("key", sorted(SPEC["dose"]))
def test_conformal_continuous_bands_equal_cfcausal(key):
    seed, frac, alpha = key.split("_")
    res = sp.conformal_continuous(
        _DOSE,
        y="y",
        treatment="t",
        covariates=["x1"],
        test_data=_DOSE_T,
        alpha=float(alpha),
        estimator=LinearRegression(),
        calibration_frac=float(frac),
        random_state=int(seed),
    )
    ref = REF["dose"][key]
    _same(res.predictions["prediction"], ref["point"])
    _same(res.predictions["lo"], ref["lower"])
    _same(res.predictions["hi"], ref["upper"])


# ---------------------------------------------------------------------------
# conformal_interference: the same order statistic (no R reference)
# ---------------------------------------------------------------------------


def test_conformal_interference_quantile_is_the_conformal_order_statistic():
    """``quantile`` is the ceil((n + 1)(1 - alpha))-th smallest cluster score.

    No package implements the cluster-mean score, so this is the defining
    identity, not a cross-language comparison.
    """
    rng = np.random.default_rng(3)
    rows = []
    for c in range(40):
        shock = rng.normal(0, 0.5)
        for _ in range(6):
            d = int(rng.integers(0, 2))
            x = rng.normal()
            rows.append({"g": c, "d": d, "x": x, "y": 1 + 0.6 * d + 0.4 * x + shock})
    df = pd.DataFrame(rows)
    res = sp.conformal_interference(
        df,
        y="y",
        treatment="d",
        cluster="g",
        covariates=["x"],
        test_clusters=[0, 1],
        alpha=0.2,
        estimator=LinearRegression(),
        random_state=0,
    )
    s = np.sort(res.cluster_scores.to_numpy())
    rank = int(np.ceil(0.8 * (len(s) + 1)))
    assert rank < len(s)
    assert res.quantile == s[rank - 1]
