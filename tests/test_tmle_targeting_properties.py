"""What makes a TMLE a TMLE, checked on ``sp.tmle``.

Schuler and van der Laan's *Introduction to Modern Causal Inference*
(ch. 4.4) characterises the estimator by two properties, both testable
without a reference implementation:

* it is a plug-in, so it inherits the range of the parameter;
* the targeted fit solves the efficient-influence-function equation,
  ``mean(EIF) = 0``, which is what removes the plug-in bias.

A third check uses a saturated model, where every valid estimator has to
return the stratification estimator exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

COV = ["x1", "x2"]


def _data(n: int = 600, seed: int = 0, binary: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.7 * x1 - 0.4 * x2))))
    if binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 0.8 * a + 0.6 * x1)))).astype(
            float
        )
    else:
        y = a * (1 + 0.5 * x1) + x1 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "a": a, "y": y})


def _fit(df: pd.DataFrame, **kw):
    binary = set(df["y"].unique()) <= {0.0, 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.tmle(
            df,
            y="y",
            treat="a",
            covariates=COV,
            outcome_library=[LogisticRegression() if binary else LinearRegression()],
            propensity_library=[LogisticRegression()],
            n_folds=2,
            **kw,
        )


@pytest.mark.parametrize("binary", [True, False])
@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC", "EY1", "EY0", "RR", "OR"])
def test_targeted_fit_solves_the_eif_equation(binary: bool, estimand: str) -> None:
    df = _data(binary=binary)
    if not binary and estimand in ("RR", "OR"):
        with pytest.raises(MethodIncompatibility, match="not defined"):
            _fit(df, estimand=estimand)
        return
    res = _fit(df, estimand=estimand)
    ic = res.model_info["influence_function"]
    assert ic.shape == (len(df),)
    scale = max(1.0, float(np.ptp(df["y"])))
    assert abs(ic.mean()) < 1e-8 * scale
    # The reported standard error is the one this influence function gives.
    se_ic = ic.std(ddof=1) / np.sqrt(len(df))
    if res.model_info["influence_function_scale"] == "log":
        se_ic *= res.estimate
    np.testing.assert_allclose(res.se, se_ic, rtol=1e-12)


def test_plug_in_respects_the_range_of_a_binary_outcome() -> None:
    # Rare outcome, poor overlap: the regime where a bias-corrected or
    # estimating-equation estimator can leave [0, 1] and a plug-in cannot.
    rng = np.random.default_rng(3)
    n = 250
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-2.5 * x1)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-3.0 + 1.0 * a + 1.5 * x1)))).astype(float)
    df = pd.DataFrame({"x1": x1, "x2": x2, "a": a, "y": y})
    table = _fit(df, estimand="EY1").detail.set_index("parameter")
    assert 0.0 <= table.loc["EY1", "estimate"] <= 1.0
    assert 0.0 <= table.loc["EY0", "estimate"] <= 1.0
    for estimand in ("ATT", "ATC"):
        assert -1.0 <= _fit(df, estimand=estimand).estimate <= 1.0


def test_contrasts_are_functions_of_the_two_targeted_means() -> None:
    table = _fit(_data(), estimand="OR").detail.set_index("parameter")["estimate"]
    e1, e0 = table["EY1"], table["EY0"]
    np.testing.assert_allclose(table["ATE"], e1 - e0, rtol=1e-14)
    np.testing.assert_allclose(table["RR"], e1 / e0, rtol=1e-14)
    np.testing.assert_allclose(table["OR"], e1 / (1 - e1) / (e0 / (1 - e0)), rtol=1e-14)


def test_saturated_fits_return_the_stratification_estimator() -> None:
    # One binary covariate, cell means as initial fits: the nonparametric
    # maximum likelihood estimate. Nothing is left to target, and every
    # estimand has a closed form.
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.binomial(1, 0.4, n)
    a = rng.binomial(1, np.where(x == 1, 0.7, 0.3))
    y = rng.binomial(1, 0.2 + 0.3 * a + 0.25 * x).astype(float)
    df = pd.DataFrame({"x1": x, "x2": 0.0, "a": a, "y": y})
    cell = df.groupby(["x1", "a"])["y"].mean()
    q0 = np.array([cell[(v, 0)] for v in x])
    q1 = np.array([cell[(v, 1)] for v in x])
    g = df.groupby("x1")["a"].transform("mean").to_numpy()
    diff = q1 - q0
    truth = {
        "ATE": diff.mean(),
        "ATT": diff[a == 1].mean(),
        "ATC": diff[a == 0].mean(),
        "EY1": q1.mean(),
        "RR": q1.mean() / q0.mean(),
    }
    for estimand, value in truth.items():
        res = sp.tmle(
            df,
            y="y",
            treat="a",
            covariates=COV,
            Q=np.column_stack([q0, q1]),
            g1W=g,
            estimand=estimand,
        )
        np.testing.assert_allclose(res.estimate, value, rtol=0, atol=1e-10)


def test_atc_is_the_att_of_the_relabelled_treatment() -> None:
    df = _data(binary=False)
    atc = _fit(df, estimand="ATC")
    att_flipped = _fit(df.assign(a=1 - df["a"]), estimand="ATT")
    np.testing.assert_allclose(atc.estimate, -att_flipped.estimate, rtol=1e-9)
    np.testing.assert_allclose(atc.se, att_flipped.se, rtol=1e-9)
    assert atc.model_info["n_treated"] == int(df["a"].sum())


def test_att_is_the_plug_in_over_the_treated() -> None:
    # Outcome-only targeting with the propensity held fixed: the score
    # equation makes the estimating-equation form equal to the plug-in mean
    # over the treated, which a saturated fit shows directly (see
    # test_saturated_fits_return_the_stratification_estimator) and the
    # zero-mean influence function shows in general.
    df = _data(binary=False)
    single = _fit(df, estimand="ATT")
    per_arm = _fit(df, estimand="ATT", fluctuation="per_arm")
    assert abs(single.estimate - per_arm.estimate) < 0.1 * single.se
    assert per_arm.detail is None


def test_estimand_is_validated_and_case_insensitive() -> None:
    df = _data()
    # Before 1.39 any string other than 'ATE' silently returned the ATT.
    assert _fit(df, estimand="ate").estimate == _fit(df, estimand="ATE").estimate
    assert _fit(df, estimand="ate").estimand == "ATE"
    for bad in ("LATE", "x", ""):
        with pytest.raises(MethodIncompatibility, match="unknown estimand"):
            _fit(df, estimand=bad)


def test_incompatible_targeting_choices_are_refused() -> None:
    df = _data()
    with pytest.raises(MethodIncompatibility, match="per_arm"):
        _fit(df, estimand="RR", fluctuation="single")


def test_default_ate_has_no_table_and_per_arm_has_one() -> None:
    df = _data()
    assert _fit(df).detail is None
    assert list(_fit(df, fluctuation="per_arm").detail["parameter"]) == [
        "EY1",
        "EY0",
        "ATE",
        "RR",
        "OR",
    ]


def test_scalar_propensity_is_a_known_randomisation_probability() -> None:
    rng = np.random.default_rng(5)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, 0.5, n)
    y = a + x1 + rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "a": a, "y": y})
    q = np.column_stack([x1, 1 + x1])
    one = sp.tmle(df, y="y", treat="a", covariates=COV, Q=q, g1W=0.5)
    vec = sp.tmle(df, y="y", treat="a", covariates=COV, Q=q, g1W=np.full(n, 0.5))
    assert one.estimate == vec.estimate and one.se == vec.se


# ---------------------------------------------------------------------------
# Observation weights for the effect on the treated / the controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimand", ["ATT", "ATC"])
def test_integer_weights_equal_row_replication(estimand: str) -> None:
    # With the initial fits supplied, weighting a row by k and stacking k
    # copies of it are the same estimating equations.
    rng = np.random.default_rng(2)
    df = _data(n=300, seed=2, binary=False)
    k = rng.integers(1, 4, len(df))
    q = np.column_stack([df["x1"], 1 + 1.4 * df["x1"]])
    g = (1 / (1 + np.exp(-(0.7 * df["x1"] - 0.4 * df["x2"])))).to_numpy()
    weighted = sp.tmle(
        df.assign(w=k),
        y="y",
        treat="a",
        covariates=COV,
        Q=q,
        g1W=g,
        estimand=estimand,
        weights="w",
    )
    rows = np.repeat(np.arange(len(df)), k)
    stacked = sp.tmle(
        df.iloc[rows].reset_index(drop=True),
        y="y",
        treat="a",
        covariates=COV,
        Q=q[rows],
        g1W=g[rows],
        estimand=estimand,
    )
    np.testing.assert_allclose(weighted.estimate, stacked.estimate, rtol=1e-9)
    assert abs(weighted.model_info["influence_function"].mean()) < 1e-8


def test_weighted_saturated_att_is_the_weighted_stratification_formula() -> None:
    rng = np.random.default_rng(4)
    n = 1500
    x = rng.binomial(1, 0.5, n)
    a = rng.binomial(1, np.where(x == 1, 0.65, 0.3))
    y = rng.binomial(1, 0.2 + 0.25 * a + 0.3 * x).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame({"x1": x, "x2": 0.0, "a": a, "y": y, "w": w})

    def wmean(v, m):
        return np.sum(w[m] * v[m]) / np.sum(w[m])

    q0 = np.array([wmean(y, (x == v) & (a == 0)) for v in x])
    q1 = np.array([wmean(y, (x == v) & (a == 1)) for v in x])
    g = np.array([wmean(a.astype(float), x == v) for v in x])
    res = sp.tmle(
        df,
        y="y",
        treat="a",
        covariates=COV,
        Q=np.column_stack([q0, q1]),
        g1W=g,
        estimand="ATT",
        weights="w",
    )
    np.testing.assert_allclose(res.estimate, wmean(q1 - q0, a == 1), atol=1e-10)


# ---------------------------------------------------------------------------
# Bootstrap inference
# ---------------------------------------------------------------------------


def test_bootstrap_replaces_inference_and_keeps_the_estimate() -> None:
    df = _data(n=400, seed=6, binary=False)
    plain = _fit(df, estimand="ATT")
    boot = _fit(df, estimand="ATT", se_method="bootstrap", n_boot=60)
    again = _fit(df, estimand="ATT", se_method="bootstrap", n_boot=60)
    assert boot.estimate == plain.estimate
    assert boot.se == again.se and boot.ci == again.ci
    info = boot.model_info
    assert info["se_method"] == "bootstrap" and info["n_boot_failed"] == 0
    assert info["se_influence"] == plain.se
    draws = info["bootstrap_estimates"]
    assert draws.shape == (60,)
    np.testing.assert_allclose(boot.se, draws.std(ddof=1), rtol=1e-12)
    np.testing.assert_allclose(boot.ci, np.quantile(draws, [0.025, 0.975]), rtol=1e-12)
    # Same order of magnitude as the analytic one where overlap is good.
    assert 0.6 < boot.se / plain.se < 1.6


def test_bootstrap_covers_the_whole_table() -> None:
    df = _data(n=400, seed=7)
    plain = _fit(df, estimand="RR")
    boot = _fit(df, estimand="RR", se_method="bootstrap", n_boot=40)
    assert list(boot.detail["parameter"]) == list(plain.detail["parameter"])
    np.testing.assert_array_equal(boot.detail["estimate"], plain.detail["estimate"])
    assert not np.allclose(boot.detail["se"], plain.detail["se"])
    row = boot.detail.set_index("parameter").loc["RR"]
    assert boot.ci == (row["ci_lower"], row["ci_upper"]) and boot.se == row["se"]
    assert row["ci_lower"] < boot.estimate < row["ci_upper"]


def test_cluster_bootstrap_resamples_clusters() -> None:
    df = _data(n=400, seed=8, binary=False)
    df["g"] = np.arange(len(df)) // 8
    boot = _fit(df, se_method="bootstrap", n_boot=40, cluster="g")
    assert boot.model_info["se_method"] == "cluster_bootstrap"
    assert boot.se > 0


def test_bootstrap_refuses_what_it_cannot_resample() -> None:
    df = _data(n=200)
    q = np.column_stack([np.full(200, 0.4), np.full(200, 0.6)])
    with pytest.raises(MethodIncompatibility, match="nothing to"):
        sp.tmle(
            df,
            y="y",
            treat="a",
            covariates=COV,
            Q=q,
            g1W=0.5,
            se_method="bootstrap",
        )
    with pytest.raises(MethodIncompatibility, match="fold_indices"):
        _fit(df, se_method="bootstrap", fold_indices=np.arange(200) % 4)
    with pytest.raises(MethodIncompatibility, match="at least 20"):
        _fit(df, se_method="bootstrap", n_boot=5)
    with pytest.raises(MethodIncompatibility, match="se_method"):
        _fit(df, se_method="jackknife")
