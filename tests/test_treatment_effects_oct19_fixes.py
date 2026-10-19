"""Treatment-effect fixes of the 2026-10-19 audit follow-up.

Each test asserts a true proposition -- an identity, a known optimum found by
exhaustive search, or the typed refusal of an input the estimator cannot
handle. The reference-package checks of the new options live in
``tests/reference_parity/test_matchit_constraints_parity.py`` and
``tests/reference_parity/test_ebalance_estimand_parity.py``.
"""

from __future__ import annotations

import itertools
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility


def _design(n=600, seed=0, strength=0.5):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-strength * (x1 - x2))))
    y = 1 + 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


# --------------------------------------------------------------------------
# non-binary treatments are refused by name, not turned into numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda df: sp.cbps(df, "y", "d", ["x1"], n_bootstrap=2),
        lambda df: sp.ipw(df, "y", "d", ["x1"]),
        lambda df: sp.aipw(df, "y", "d", ["x1"]),
        lambda df: sp.tmle(df, "y", "d", ["x1"]),
        lambda df: sp.ebalance(df, "y", "d", ["x1"]),
        lambda df: sp.optimal_match(df, treatment="d", outcome="y", covariates=["x1"]),
    ],
)
@pytest.mark.parametrize("recode", ["two", "three_arms", "dose"])
def test_non_binary_treatment_is_refused(call, recode):
    df = _design(n=200)
    if recode == "two":
        df["d"] = 2 * df["d"]  # 0/2: was "need 2 treated" or a number
    elif recode == "three_arms":
        df["d"] = np.arange(len(df)) % 3
    else:
        df["d"] = df["x2"]  # cbps returned -122.9 on a continuous column
    with pytest.raises(MethodIncompatibility, match="coded 0/1"):
        call(df)


# --------------------------------------------------------------------------
# sp.tmle(weights=) with the default Super Learner library
# --------------------------------------------------------------------------


def test_tmle_weights_default_library_drops_knn_and_says_so():
    """Identity: the weighted fit with the default library equals the fit
    with that library minus the learners that cannot take weights."""
    from statspai.tmle.super_learner import SuperLearner

    df = _design(n=400, seed=3)
    df["w"] = 1 + (np.arange(len(df)) % 5) / 4
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        auto = sp.tmle(df, "y", "d", ["x1", "x2"], weights="w", random_state=1)
    msgs = [str(w.message) for w in rec]
    assert any("KNeighborsRegressor" in m for m in msgs)
    assert any("KNeighborsClassifier" in m for m in msgs)

    reg = [
        m
        for m in SuperLearner(task="regression", random_state=1)._default_library()
        if "KNeighbors" not in type(m).__name__
    ]
    clf = [
        m
        for m in SuperLearner(task="classification", random_state=1)._default_library()
        if "KNeighbors" not in type(m).__name__
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explicit = sp.tmle(
            df,
            "y",
            "d",
            ["x1", "x2"],
            weights="w",
            outcome_library=reg,
            propensity_library=clf,
            random_state=1,
        )
    assert auto.estimate == explicit.estimate
    assert auto.se == explicit.se


# --------------------------------------------------------------------------
# sp.dml(model='irm') says how many propensities it clipped
# --------------------------------------------------------------------------


def test_dml_irm_clipping_is_counted_and_warned():
    rng = np.random.default_rng(5)
    n = 2000
    x = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-4 * x)))
    df = pd.DataFrame({"y": 1 + 2 * d + x + rng.normal(size=n), "d": d, "x": x})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = sp.dml(df, "y", "d", ["x"], model="irm", model_y="lasso", model_d="lasso")
    diag = r.model_info["diagnostics"]
    ps = np.asarray(r.model_info["_pscore"])
    thr = r.model_info["trimming_threshold"]
    # identity: the count is the number of cross-fitted scores outside the bounds
    assert diag["n_clipped"] == int(np.sum((ps < thr) | (ps > 1 - thr)))
    assert diag["n_clipped"] == diag["n_clipped_below"] + diag["n_clipped_above"]
    assert diag["clip_share"] == pytest.approx(diag["n_clipped"] / n, rel=1e-12)
    hits = [w for w in rec if issubclass(w.category, AssumptionWarning)]
    assert hits and f"{diag['n_clipped']} cross-fitted" in str(hits[0].message)
    assert any(v["test"] == "dml_overlap" for v in r.violations())


def test_dml_irm_without_clipping_does_not_warn():
    df = _design(n=800, seed=2, strength=0.3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = sp.dml(
            df, "y", "d", ["x1", "x2"], model="irm", model_y="lasso", model_d="lasso"
        )
    assert r.model_info["diagnostics"]["n_clipped"] == 0
    assert not [w for w in rec if "cross-fitted propensity" in str(w.message)]


# --------------------------------------------------------------------------
# sp.rosenbaum_bounds(pair_id=) with malformed pairs
# --------------------------------------------------------------------------


def _pairs(n_pairs=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "pair": np.repeat(np.arange(n_pairs), 2),
            "d": np.tile([1, 0], n_pairs),
            "y": rng.normal(size=2 * n_pairs) + np.tile([0.8, 0.0], n_pairs),
        }
    )


def test_rosenbaum_two_treated_in_a_pair_is_a_typed_error():
    p = _pairs()
    p.loc[1, "d"] = 1
    with pytest.raises(MethodIncompatibility, match="exactly one treated"):
        sp.rosenbaum_bounds(data=p, y="y", treat="d", pair_id="pair")


def test_rosenbaum_incomplete_pair_is_dropped_with_a_warning():
    """Identity: dropping a pair that lost a row gives the bounds of the
    remaining pairs."""
    p = _pairs()
    full = p[p["pair"] != 3]
    p.loc[7, "y"] = np.nan  # pair 3 loses its control
    with pytest.warns(UserWarning, match="1 pair"):
        got = sp.rosenbaum_bounds(data=p, y="y", treat="d", pair_id="pair")
    want = sp.rosenbaum_bounds(data=full, y="y", treat="d", pair_id="pair")
    assert got.n_pairs == want.n_pairs == 39
    np.testing.assert_array_equal(got.pvalue_upper, want.pvalue_upper)
    assert got.statistic == want.statistic


# --------------------------------------------------------------------------
# sp.optimal_match with fewer controls than treated
# --------------------------------------------------------------------------


def test_optimal_match_more_treated_than_controls_is_the_exhaustive_optimum():
    rng = np.random.default_rng(11)
    n_t, n_c = 6, 4
    df = pd.DataFrame(
        {
            "d": [1] * n_t + [0] * n_c,
            "x": rng.normal(size=n_t + n_c),
            "y": rng.normal(size=n_t + n_c),
        }
    )
    with pytest.warns(AssumptionWarning, match="2 treated units stay unmatched"):
        r = sp.optimal_match(
            df, treatment="d", outcome="y", covariates=["x"], metric="euclidean"
        )
    x = df["x"].to_numpy()
    cost = np.abs(x[:n_t, None] - x[None, n_t:])
    best = min(
        sum(cost[t, c] for c, t in enumerate(perm))
        for perm in itertools.permutations(range(n_t), n_c)
    )
    assert r.n_matched == n_c and r.n_unmatched_treated == 2
    assert abs(r.distances.sum() - best) < 1e-12
    assert "not the ATT" in r.summary()


# --------------------------------------------------------------------------
# sp.bcf says what it is
# --------------------------------------------------------------------------


def test_bcf_warns_that_it_is_an_approximation():
    from statspai.bcf.bcf import BCFApproximationWarning

    df = _design(n=200, seed=4)
    with pytest.warns(BCFApproximationWarning, match="not Bayesian Causal Forest"):
        r = sp.bcf(
            df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            n_bootstrap=3,
            n_trees_mu=20,
            n_trees_tau=10,
        )
    assert r.model_info["posterior"] is False
    assert "not BART" in r.method


# --------------------------------------------------------------------------
# sp.dose_response default models
# --------------------------------------------------------------------------


def test_dose_response_default_models_warn_and_report_monotonicity():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    t = 1 + 0.8 * x + rng.normal(size=n)
    df = pd.DataFrame({"y": 0.8 * t + x + rng.normal(size=n), "t": t, "x": x})
    with pytest.warns(sp.StatsPAIWarning, match="non-monotone"):
        r = sp.dose_response(df, "y", "t", ["x"], n_dose_points=8, n_bootstrap=2)
    curve = r.detail["response"].to_numpy()
    mono = bool(np.all(np.diff(curve) > 0) or np.all(np.diff(curve) < 0))
    assert r.model_info["curve_monotone"] is mono
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.StatsPAIWarning)
        sp.dose_response(
            df,
            "y",
            "t",
            ["x"],
            n_dose_points=8,
            n_bootstrap=2,
            treatment_model=LinearRegression(),
            outcome_model=LinearRegression(),
        )


# --------------------------------------------------------------------------
# sp.causal_forest does not depend on the thread count
# --------------------------------------------------------------------------


def test_causal_forest_is_thread_count_independent_end_to_end():
    df = _design(n=600, seed=8)
    fits = [
        sp.causal_forest(
            data=df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            n_estimators=100,
            n_jobs=nj,
            random_state=3,
        )
        for nj in (1, 3)
    ]
    a, b = (
        np.asarray(f.oob_effect() if callable(f.oob_effect) else f.oob_effect)
        for f in fits
    )
    np.testing.assert_array_equal(a, b)
    assert (
        fits[0].average_treatment_effect()["se"]
        == fits[1].average_treatment_effect()["se"]
    )


# --------------------------------------------------------------------------
# sp.sensemakr returns a result object that is still a dict
# --------------------------------------------------------------------------


def test_sensemakr_result_is_a_dict_with_attributes():
    df = _design(n=500, seed=6)
    r = sp.sensemakr(df, y="y", treat="d", controls=["x1", "x2"])
    assert isinstance(r, dict)
    for key in ("rv_q", "rv_qa", "beta_treat", "se_treat", "benchmark_table"):
        assert getattr(r, key) is r[key]
    assert r.estimate == r["beta_treat"] and r.se == r["se_treat"]
    assert r["n_obs"] == 500 and r["df_resid"] == 500 - 4
    text = r.summary()
    assert "RV_q" in text and "x1" in text
    json.dumps(r.to_dict())
    assert "cinelli2020making" in r.cite()
