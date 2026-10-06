"""``sp.threshold``: known-truth recovery, inference and refusals.

Stata parity (``threshold`` and ``nl``) is in
``tests/reference_parity/test_hansen_methods_stata_parity.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _data(n=800, seed=0, jump=1.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "q": rng.uniform(0, 1, n),
            "x": rng.normal(size=n),
            "g": rng.integers(0, 40, n),
        }
    )
    effect = rng.normal(size=40)[df.g]
    df["y"] = 1 + 0.5 * df.x + jump * (df.q > 0.4) * (1 + df.x) + rng.normal(size=n) / 2
    df["y_fe"] = df.y + effect
    df["y_kink"] = (
        1
        + 0.5 * df.x
        - np.minimum(df.q - 0.4, 0)
        + 2 * np.maximum(df.q - 0.4, 0)
        + rng.normal(size=n) / 5
    )
    return df


def test_recovers_the_threshold_and_both_regimes():
    df = _data()
    fit = sp.threshold("y ~ x", df, "q")
    info = fit.model_info
    assert abs(info["threshold"] - 0.4) < 0.01
    lo, hi = info["threshold_ci"]
    assert lo <= info["threshold"] <= hi and hi - lo < 0.05
    regimes = info["regimes"]
    assert abs(regimes.loc["x", "below"] - 0.5) < 0.1
    assert abs(regimes.loc["x", "above"] - 1.5) < 0.1
    assert abs(regimes.loc["Intercept", "change"] - 1.0) < 0.15
    assert info["n_below"] + info["n_above"] == len(df)
    # the criterion is zero at the estimate and the interval is its
    # sub-level set
    crit = info["criterion"]
    assert crit["lr"].min() == 0.0
    inside = crit.loc[crit["lr"] <= info["lr_critical"], "threshold"]
    assert (inside.min(), inside.max()) == (lo, hi)
    assert np.isclose(info["lr_critical"], -2 * np.log(1 - np.sqrt(0.95)))


def test_the_fit_is_ols_given_the_threshold():
    df = _data()
    fit = sp.threshold("y ~ x", df, "q", vce="robust")
    df["above"] = (df.q > fit.model_info["threshold"]).astype(float)
    df["above_x"] = df["above"] * df.x
    ols = sp.regress("y ~ x + above + above_x", data=df, vce="robust")
    np.testing.assert_allclose(fit.params.to_numpy(), ols.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        fit.std_errors.to_numpy(), ols.std_errors.to_numpy(), rtol=1e-10
    )
    # and no other candidate does better
    for gamma in np.quantile(df.q, [0.2, 0.35, 0.5, 0.8]):
        df["a"] = (df.q > gamma).astype(float)
        df["ax"] = df["a"] * df.x
        other = sp.regress("y ~ x + a + ax", data=df)
        rss = float((np.asarray(other.data_info["residuals"]) ** 2).sum())
        assert rss >= fit.diagnostics["Residual SS"] - 1e-9


def test_fixed_effects_are_removed_at_every_candidate():
    df = _data()
    fit = sp.threshold("y_fe ~ x", df, "q", absorb="g", cluster="g")
    assert abs(fit.model_info["threshold"] - 0.4) < 0.01
    df["above"] = (df.q > fit.model_info["threshold"]).astype(float)
    df["above_x"] = df["above"] * df.x
    dummies = sp.regress("y_fe ~ x + above + above_x + C(g)", data=df)
    for name in ("x", "above"):
        assert np.isclose(fit.params[name], dummies.params[name], rtol=1e-9)
    assert np.isclose(fit.params["above:x"], dummies.params["above_x"], rtol=1e-9)
    assert "Intercept" not in fit.params.index
    assert fit.model_info["n_clusters"] == 40


def test_kink_model_is_nonlinear_least_squares():
    df = _data()
    fit = sp.threshold("y_kink ~ x", df, "q", kink=True)
    assert abs(fit.params["threshold"] - 0.4) < 3 * fit.std_errors["threshold"]
    assert abs(fit.params["q:below"] + 1) < 0.3 and abs(fit.params["q:above"] - 2) < 0.2
    nls = sp.nls(
        "y_kink ~ {b1}*(q-{c})*(q<{c}) + {b2}*(q-{c})*(q>{c}) + {b3}*x + {b4}",
        df,
        start={"b1": -1, "b2": 2, "b3": 0.5, "b4": 1, "c": fit.params["threshold"]},
        vce="robust",
    )
    assert fit.diagnostics["Residual SS"] <= nls.diagnostics["Residual SS"] * (
        1 + 1e-12
    )
    for ours, theirs in (
        ("q:below", "b1"),
        ("q:above", "b2"),
        ("x", "b3"),
        ("Intercept", "b4"),
        ("threshold", "c"),
    ):
        assert np.isclose(fit.params[ours], nls.params[theirs], rtol=1e-6)
        assert np.isclose(fit.std_errors[ours], nls.std_errors[theirs], rtol=1e-5)
    lo, hi = fit.model_info["threshold_ci"]
    assert fit.model_info["threshold_ci_method"] == "normal"
    assert np.isclose(hi - lo, 2 * 1.959963984540054 * fit.std_errors["threshold"])


def test_linearity_test_rejects_a_threshold_and_not_a_line():
    df = _data(n=500, jump=0.5)
    with_threshold = sp.threshold("y ~ x", df, "q", n_boot=300, seed=1)
    test = with_threshold.model_info["linearity_test"]
    assert test["pvalue"] < 0.01
    assert test["sse_linear"] > test["sse_threshold"]
    # same seed, same answer
    again = sp.threshold("y ~ x", df, "q", n_boot=300, seed=1)
    assert again.model_info["linearity_test"]["pvalue"] == test["pvalue"]


def test_linearity_test_has_the_right_size_under_the_null():
    """The naive F test would compare the largest of many F statistics with
    one F quantile. Rejections at 10% over 200 linear samples: near 20."""
    rejections = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.StatsPAIWarning)
        for seed in range(200):
            df = _data(n=150, seed=seed, jump=0.0)
            fit = sp.threshold("y ~ x", df, "q", n_boot=99, seed=seed)
            rejections += fit.model_info["linearity_test"]["pvalue"] <= 0.10
    assert 8 <= rejections <= 36


def test_candidate_grids():
    df = _data(n=200)
    every = sp.threshold("y ~ x", df, "q", trim=0.15)
    # floor(n * trim) observations are kept out on each side
    assert every.model_info["n_candidates"] == 200 - 2 * 30 + 1
    spaced = sp.threshold("y ~ x", df, "q", trim=0.15, grid=50)
    points = spaced.model_info["criterion"]["threshold"].to_numpy()
    assert len(points) == 50
    np.testing.assert_allclose(
        [points[0], points[-1]], np.quantile(df.q, [0.15, 0.85]), rtol=1e-12
    )
    given = sp.threshold("y ~ x", df, "q", grid=[0.3, 0.4, 0.5])
    assert given.model_info["threshold"] in (0.3, 0.4, 0.5)


def test_refusals_and_warnings():
    df = _data(n=200)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not a column"):
        sp.threshold("y ~ x", df, "nope")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="trim"):
        sp.threshold("y ~ x", df, "q", trim=0.6)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="must not be in"):
        sp.threshold("y_kink ~ x + q", df, "q", kink=True)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="regime"):
        sp.threshold("y_kink ~ x", df, "q", kink=True, regime=["x"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="vce"):
        sp.threshold("y ~ x", df, "q", vce="hac")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="regime variable"):
        sp.threshold("y ~ x", df, "q", regime=["zzz"])
    # a criterion that keeps falling to the edge of the range is reported
    # the jump is at 0.97, outside the trimmed range
    df["late"] = 3 * (df.q > 0.97) + np.random.default_rng(3).normal(size=len(df)) / 5
    with pytest.warns(sp.StatsPAIWarning, match="edge of the search range"):
        sp.threshold("late ~ x", df, "q", regime=[])


def test_rows_with_a_missing_threshold_are_dropped():
    df = _data(n=300)
    full = sp.threshold("y ~ x", df.iloc[10:], "q")
    df.loc[df.index[:10], "q"] = np.nan
    fit = sp.threshold("y ~ x", df, "q")
    assert fit.data_info["nobs"] == 290
    np.testing.assert_allclose(fit.params.to_numpy(), full.params.to_numpy())
