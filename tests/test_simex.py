"""``sp.simex``: known-truth recovery, identities and refusals.

The screen against R ``simex`` is in
``tests/reference_parity/test_simex_r_screen.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import ConvergenceWarning, MethodIncompatibility


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + x + 0.5 * z)))
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=n),
            "d": (rng.uniform(size=n) < p).astype(float),
            "c": rng.poisson(np.exp(0.2 + 0.5 * x + 0.3 * z)).astype(float),
            "w": x + rng.normal(scale=0.5, size=n),
            "z": z,
        }
    )


@pytest.fixture(scope="module")
def big():
    return _data(n=20000, seed=1)


def test_rational_extrapolant_recovers_the_linear_slope(big):
    """In a linear model with normal error the path is exactly
    a + b / (c + lambda), so the nonlinear extrapolant is consistent.
    Naive slope -> 2 * 1 / 1.25 = 1.6."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # flat path of z
        fit = sp.simex(
            "y ~ w + z", big, error_sd={"w": 0.5}, extrapolant="nonlinear", B=50
        )
    assert fit.model_info["naive_params"]["w"] == pytest.approx(1.6, abs=0.03)
    # Sampling SE about 0.010; simulation error at B=50 about 0.003.
    assert fit.params["w"] == pytest.approx(2.0, abs=0.04)


@pytest.mark.parametrize(
    "family, formula, truth",
    [("logit", "d ~ w + z", 1.0), ("poisson", "c ~ w + z", 0.5)],
)
def test_quadratic_extrapolant_removes_most_of_the_attenuation(
    big, family, formula, truth
):
    fit = sp.simex(formula, big, error_sd={"w": 0.5}, family=family, B=30)
    naive = fit.model_info["naive_params"]["w"]
    assert abs(fit.params["w"] - truth) < 0.4 * abs(naive - truth)


@pytest.mark.parametrize(
    "family, formula",
    [("gaussian", "y ~ w + z"), ("logit", "d ~ w + z"), ("poisson", "c ~ w + z")],
)
def test_lambda_zero_is_the_naive_fit(family, formula):
    df = _data(n=500)
    fit = sp.simex(formula, df, error_sd={"w": 0.5}, family=family, B=5)
    ref = {
        "gaussian": lambda: sp.regress(formula, data=df),
        "logit": lambda: sp.logit(formula, data=df),
        "poisson": lambda: sp.poisson(formula, data=df),
    }[family]()
    names = list(fit.params.index)
    # sp.poisson labels the intercept ``_cons``.
    ref_names = [
        "_cons" if n == "Intercept" and "_cons" in ref.params else n for n in names
    ]
    np.testing.assert_allclose(
        fit.model_info["path"].loc[0.0, names].values,
        ref.params[ref_names].values,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        fit.model_info["naive_std_errors"][names].values,
        ref.std_errors[ref_names].values,
        rtol=1e-6,
    )


def test_zero_error_variance_returns_the_naive_estimator_exactly():
    df = _data(n=400)
    fit = sp.simex("d ~ w + z", df, error_sd={"w": 0.0}, family="logit")
    np.testing.assert_array_equal(
        fit.params.values, fit.model_info["naive_params"].values
    )
    np.testing.assert_array_equal(
        fit.std_errors.values, fit.model_info["naive_std_errors"].values
    )
    assert fit.diagnostics["no_measurement_error"] is True


def test_seed_reproducibility_and_column_of_sds():
    df = _data(n=500)
    a = sp.simex("y ~ w + z", df, error_sd={"w": 0.5}, B=20, seed=7)
    b = sp.simex("y ~ w + z", df, error_sd={"w": 0.5}, B=20, seed=7)
    c = sp.simex("y ~ w + z", df.assign(s=0.5), error_sd={"w": "s"}, B=20, seed=7)
    np.testing.assert_array_equal(a.params.values, b.params.values)
    np.testing.assert_allclose(a.params.values, c.params.values, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors.values, c.std_errors.values, rtol=1e-12)


def test_linear_extrapolant_on_a_straight_path_is_exact():
    """lambda-path linear in lambda: the linear extrapolant reads the
    straight line at -1, which the quadratic matches too."""
    from statspai.measurement_error.simex import _extrapolate

    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    path = np.column_stack([3.0 - 0.4 * grid, 1.0 + 0.0 * grid])
    for method in ("linear", "quadratic"):
        out, failed = _extrapolate(grid, path, method)
        np.testing.assert_allclose(out, [3.4, 1.0], rtol=1e-12)
        assert failed == []
    rat = 1.0 + 2.0 / (3.0 + grid)
    out, failed = _extrapolate(grid, rat[:, None], "nonlinear")
    assert out[0] == pytest.approx(1.0 + 2.0 / 2.0, rel=1e-8)


def test_flat_path_under_the_nonlinear_extrapolant_falls_back_with_a_warning():
    df = _data(n=500)
    with pytest.warns(ConvergenceWarning, match="quadratic was used"):
        fit = sp.simex(
            "y ~ w + z", df, error_sd={"w": 0.5}, extrapolant="nonlinear", B=10
        )
    # The paths of the intercept and z are flat up to simulation noise.
    fallback = fit.diagnostics["quadratic_fallback_terms"]
    assert fallback and "w" not in fallback


def test_duplicate_index_labels():
    df = _data(n=300)
    dup = df.copy()
    dup.index = [0] * len(dup)
    a = sp.simex("y ~ w + z", df, error_sd={"w": 0.5}, B=5)
    b = sp.simex("y ~ w + z", dup, error_sd={"w": 0.5}, B=5)
    assert b.nobs == 300
    np.testing.assert_array_equal(a.params.values, b.params.values)


def test_refusals():
    df = _data(n=200)
    bad = [
        (dict(error_sd={"q": 0.5}), "not a term"),
        (dict(error_sd={}), "names no"),
        (dict(error_sd={"w": -1.0}), "non-negative"),
        (dict(error_sd={"w": 0.5}, lambdas=(0.5, -1)), "positive"),
        (dict(error_sd={"w": 0.5}, lambdas=(1.0,)), "cannot"),
        (dict(error_sd={"w": 0.5}, B=1), "at least 2"),
        (dict(error_sd={"w": 0.5}, family="cauchit"), "family"),
        (dict(error_sd={"w": 0.5}, extrapolant="cubic"), "extrapolant"),
        (dict(error_sd={"w": 0.5}, vce="hc3"), "vce"),
    ]
    for kwargs, msg in bad:
        with pytest.raises(MethodIncompatibility, match=msg):
            sp.simex("y ~ w + z", df, **kwargs)
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.simex("y ~ w + z", df, error_sd={"w": 0.5}, family="logit")
    with pytest.raises(MethodIncompatibility, match="plain column"):
        sp.simex("y ~ w + z + w:z", df, error_sd={"w": 0.5})


def test_missing_columns_are_named():
    from statspai.exceptions import ColumnNotFound

    df = _data(n=100)
    with pytest.raises(ColumnNotFound):
        sp.simex("y ~ w + z", df, error_sd={"w": "no_such_sd"})
    with pytest.raises(ColumnNotFound):
        sp.simex("y ~ w + z", df, error_sd={"w": 0.5}, weights="no_such_w")
    with pytest.raises(ColumnNotFound):
        sp.eivreg("y ~ w + z", df, reliability={"w": 0.8}, weights="no_such_w")
