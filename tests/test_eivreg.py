"""``sp.eivreg``: identities, known-truth recovery and refusals.

Numerical agreement with Stata 18 ``eivreg`` is in
``tests/reference_parity/test_eivreg_stata_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import (
    IdentificationFailure,
    MethodIncompatibility,
    StatsPAIError,
)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=n),
            "x": x + rng.normal(scale=0.5, size=n),
            "z": z,
            "w": rng.uniform(0.5, 2.0, size=n),
        }
    )


def test_reliability_one_is_ols_with_the_hc0_sandwich():
    df = _data()
    fit = sp.eivreg("y ~ x + z", df, reliability={"x": 1.0})
    ols = sp.regress("y ~ x + z", data=df, robust="hc0")
    np.testing.assert_allclose(fit.params.values, ols.params.values, rtol=1e-12)
    # With r = 1 the H rows are e_i x_i: the sandwich is White's HC0.
    np.testing.assert_allclose(fit.std_errors.values, ols.std_errors.values, rtol=1e-10)


def test_recovers_the_true_slope_when_the_reliability_is_right():
    # var(x*) = 1, var(u) = 0.25: reliability 0.8. OLS converges to 1.6.
    df = _data(n=40000, seed=1)
    fit = sp.eivreg("y ~ x + z", df, reliability={"x": 0.8})
    naive = sp.regress("y ~ x + z", data=df).params["x"]
    assert naive == pytest.approx(1.6, abs=0.03)
    # Sampling SE of the corrected slope is about 0.011 at this n.
    assert fit.params["x"] == pytest.approx(2.0, abs=0.04)
    assert fit.params["z"] == pytest.approx(0.5, abs=0.03)


def test_analytic_weights_are_scale_free():
    df = _data()
    a = sp.eivreg("y ~ x + z", df, reliability={"x": 0.8}, weights="w")
    b = sp.eivreg(
        "y ~ x + z", df.assign(w=df.w * 7.3), reliability={"x": 0.8}, weights="w"
    )
    np.testing.assert_allclose(a.params.values, b.params.values, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors.values, b.std_errors.values, rtol=1e-12)


def test_frequency_weights_equal_expanded_rows():
    df = _data(n=60)
    df["k"] = np.arange(60) % 3 + 1
    long = df.loc[df.index.repeat(df["k"])].reset_index(drop=True)
    a = sp.eivreg(
        "y ~ x + z", df, reliability={"x": 0.8}, weights="k", weights_as="frequency"
    )
    b = sp.eivreg("y ~ x + z", long, reliability={"x": 0.8})
    np.testing.assert_allclose(a.params.values, b.params.values, rtol=1e-10)
    np.testing.assert_allclose(a.std_errors.values, b.std_errors.values, rtol=1e-10)
    assert a.nobs == b.nobs == int(df["k"].sum())


def test_duplicate_index_labels_do_not_duplicate_rows():
    df = _data(n=100)
    dup = pd.concat([df.iloc[:50], df.iloc[50:]])
    dup.index = list(range(50)) * 2
    a = sp.eivreg("y ~ x + z", df, reliability={"x": 0.8}, weights="w")
    b = sp.eivreg("y ~ x + z", dup, reliability={"x": 0.8}, weights="w")
    assert b.nobs == 100
    np.testing.assert_allclose(a.params.values, b.params.values, rtol=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.2, 1.2, "high", float("nan")])
def test_reliability_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(MethodIncompatibility, match="reliability"):
        sp.eivreg("y ~ x + z", _data(), reliability={"x": bad})


def test_too_small_a_reliability_is_an_identification_failure():
    with pytest.raises(IdentificationFailure) as err:
        sp.eivreg("y ~ x + z", _data(), reliability={"x": 0.05})
    assert err.value.recovery_hint


def test_refusals():
    df = _data()
    with pytest.raises(MethodIncompatibility, match="not terms"):
        sp.eivreg("y ~ x + z", df, reliability={"q": 0.8})
    with pytest.raises(MethodIncompatibility, match="intercept"):
        sp.eivreg("y ~ x + z - 1", df, reliability={"x": 0.8})
    with pytest.raises(MethodIncompatibility, match="intercept"):
        sp.eivreg("y ~ x + z", df, reliability={"Intercept": 0.8})
    with pytest.raises(MethodIncompatibility, match="sampling"):
        sp.eivreg("y ~ x + z", df, weights="w", weights_as="sampling")
    with pytest.raises(MethodIncompatibility, match="integers"):
        sp.eivreg("y ~ x + z", df, weights="w", weights_as="frequency")
    with pytest.raises(MethodIncompatibility, match="negative"):
        sp.eivreg("y ~ x + z", df.assign(w=-df.w), weights="w")
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.eivreg("y ~ x + z + x2", df.assign(x2=2 * df.x), reliability={"x": 0.8})
    with pytest.raises(StatsPAIError):
        sp.eivreg("y ~ x + nope", df)


def test_result_carries_reliabilities_and_summary():
    fit = sp.eivreg("y ~ x + z", _data(), reliability={"x": 0.8})
    assert fit.model_info["reliability"]["x"] == 0.8
    assert fit.model_info["reliability"]["z"] == 1.0
    assert fit.diagnostics["error_prone_terms"] == ["x"]
    assert "x" in str(fit.summary())
