"""Classic SCM without predictors: which problem each setting solves.

By default every pre-treatment period is a predictor, rescaled by its range
across units, with an equal V; that convention is pinned against R ``Synth``
in ``tests/reference_parity/test_synth_rest_R_parity.py``.
``standardize_predictors=False`` fits the raw pre-treatment path, the
simplex-constrained least squares that ``augsynth(progfunc = "None",
scm = TRUE)`` returns. On the suicide-rate panel of the Remix workshop
project those weights agree with augsynth's to between 3e-8 and 1e-5, and
the default differs from them (for Ireland: pre-treatment sum of squares
51.55 against 45.14). The docstring used to call the default the
least-squares estimator.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import statspai as sp


@pytest.fixture(scope="module")
def prop99() -> pd.DataFrame:
    return sp.california_prop99()


def _simplex_least_squares(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Independent solver for min ||y - Xw||^2 on the simplex."""
    k = X.shape[1]
    out = minimize(
        lambda w: float(np.sum((y - X @ w) ** 2)),
        np.full(k, 1.0 / k),
        jac=lambda w: -2.0 * X.T @ (y - X @ w),
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 5000},
    )
    return out.x


def _fit(df: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.synth(
            df,
            outcome="packspercapita",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            method="classic",
            placebo=False,
            **kwargs,
        )


def _pre(df: pd.DataFrame):
    wide = df.pivot(index="year", columns="state", values="packspercapita")
    pre = wide[wide.index < 1989]
    donors = [c for c in pre.columns if c != "California"]
    return pre["California"].to_numpy(), pre[donors].to_numpy(), donors


def _weights(result, donors) -> np.ndarray:
    table = result.model_info["weights"].set_index("unit")["weight"]
    return table.reindex(donors).fillna(0.0).to_numpy()


def test_unstandardised_fit_is_simplex_least_squares(prop99):
    y, X, donors = _pre(prop99)
    w = _weights(_fit(prop99, standardize_predictors=False), donors)
    best = _simplex_least_squares(y, X)
    sse = lambda v: float(np.sum((y - X @ v) ** 2))  # noqa: E731
    assert sse(w) <= sse(best) * (1 + 1e-6)
    # the minimiser is unique here, so the weights themselves agree
    np.testing.assert_allclose(w, best, atol=2e-4)


def test_default_solves_the_range_scaled_problem(prop99):
    """The default minimises the squared error of the range-scaled periods,
    so it does at least as well on that criterion and no better on the raw
    one."""
    y, X, donors = _pre(prop99)
    default = _weights(_fit(prop99), donors)
    raw = _weights(_fit(prop99, standardize_predictors=False), donors)
    block = np.column_stack([y, X])
    scale = block.max(axis=1) - block.min(axis=1)
    scaled = lambda v: float(np.sum(((y - X @ v) / scale) ** 2))  # noqa: E731
    plain = lambda v: float(np.sum((y - X @ v) ** 2))  # noqa: E731
    assert scaled(default) <= scaled(raw) * (1 + 1e-6)
    assert plain(raw) <= plain(default) * (1 + 1e-6)


def test_nested_v_on_the_outcome_lags_reaches_the_least_squares_weights(prop99):
    _, _, donors = _pre(prop99)
    raw = _weights(_fit(prop99, standardize_predictors=False), donors)
    nested = _weights(_fit(prop99, v_method="nested"), donors)
    np.testing.assert_allclose(raw, nested, atol=5e-3)


def test_predictors_are_still_standardised(prop99):
    """With covariates the predictors are in different units and the
    rescaling is what makes an equal V meaningful: the switch still bites."""
    _, _, donors = _pre(prop99)
    cols = [c for c in ("lnincome", "retprice", "age15to24", "beer") if c in prop99]
    if not cols:
        pytest.skip("the bundled panel has no covariate columns")
    on = _weights(_fit(prop99, covariates=cols, v_method="equal"), donors)
    off = _weights(
        _fit(prop99, covariates=cols, v_method="equal", standardize_predictors=False),
        donors,
    )
    assert np.max(np.abs(on - off)) > 1e-3
