"""``sp.regress(collinear='pinv')``: minimum-norm OLS on a rank-deficient
design (GitHub #55).

The coefficients are ``pinv(X) @ y``. The reference is ``numpy.linalg.pinv``
on the same design matrix (what statsmodels ``OLS.fit(method='pinv')``
computes), and, for the issue's example, the numbers written in the issue.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture()
def issue_frames():
    train = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "z": [0.0, 1.0, 2.0, 3.0],
            "y": [2.0, 2.0, 4.0, 8.0],
        }
    )
    holdout = pd.DataFrame({"x": [1.0], "z": [0.0]})
    return train, holdout


@pytest.fixture()
def wide():
    rng = np.random.default_rng(55)
    n = 200
    d = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "w": rng.uniform(0.5, 2.0, n),
        }
    )
    d["c"] = d["a"] + 2.0 * d["b"]
    d["y"] = 1.0 + d["a"] - d["b"] + rng.normal(size=n)
    return d


def test_issue_example_coefficients_and_forecast(issue_frames):
    train, holdout = issue_frames
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an opt-in: nothing to warn about
        fit = sp.regress("y ~ x + z", data=train, collinear="pinv")
    # y = 1 + 2 t on x = z = t; the minimum-norm split of the slope is 1 + 1
    np.testing.assert_allclose(fit.params.to_numpy(), [1.0, 1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(fit.predict(holdout), [2.0], atol=1e-12)


def test_rank_deficiency_is_flagged_and_inference_withheld(issue_frames):
    train, _ = issue_frames
    fit = sp.regress("y ~ x + z", data=train, collinear="pinv")
    info = fit.model_info
    assert info["rank_deficient"] is True
    assert info["rank"] == 2
    assert [a["variable"] for a in info["aliased"]] == ["z"]
    assert info["inference"] == "unavailable"
    assert "omitted" not in info
    assert np.isnan(fit.std_errors.to_numpy()).all()
    # the column space is identified: n - rank residual df, and the R-squared
    # of y on t (RSS 4 / TSS 24)
    assert fit.data_info["df_resid"] == 2
    assert fit.diagnostics["R-squared"] == pytest.approx(1.0 - 4.0 / 24.0, abs=1e-12)


@pytest.mark.parametrize("weights", [None, "w"])
@pytest.mark.parametrize("formula", ["y ~ a + b + c", "y ~ a + b + c - 1"])
def test_matches_numpy_pinv(wide, formula, weights):
    kw = {"weights": weights} if weights else {}
    fit = sp.regress(formula, data=wide, collinear="pinv", **kw)
    cols = [wide[v].to_numpy() for v in ("a", "b", "c")]
    if "- 1" not in formula:
        cols = [np.ones(len(wide))] + cols
    X = np.column_stack(cols)
    y = wide["y"].to_numpy()
    sw = np.sqrt(wide["w"].to_numpy()) if weights else np.ones(len(wide))
    expected = np.linalg.pinv(X * sw[:, None], rcond=1e-10) @ (y * sw)
    np.testing.assert_allclose(fit.params.to_numpy(), expected, rtol=0, atol=1e-10)


def test_fitted_values_equal_the_omit_fit(wide):
    pinv = sp.regress("y ~ a + b + c", data=wide, collinear="pinv")
    with pytest.warns(UserWarning, match="omitted because of collinearity"):
        omit = sp.regress("y ~ a + b + c", data=wide)
    np.testing.assert_allclose(
        pinv.data_info["fitted_values"], omit.data_info["fitted_values"], atol=1e-10
    )
    assert pinv.data_info["df_resid"] == omit.data_info["df_resid"]


def test_full_rank_design_is_the_ordinary_fit(wide):
    pinv = sp.regress("y ~ a + b", data=wide, collinear="pinv")
    base = sp.regress("y ~ a + b", data=wide)
    pd.testing.assert_series_equal(pinv.params, base.params)
    pd.testing.assert_series_equal(pinv.std_errors, base.std_errors)
    assert "rank_deficient" not in pinv.model_info
    assert pinv.model_info.get("inference") != "unavailable"


def test_default_and_raise_are_unchanged(issue_frames):
    train, _ = issue_frames
    with pytest.warns(UserWarning, match="omitted because of collinearity"):
        omit = sp.regress("y ~ x + z", data=train)
    assert list(omit.params.index) == ["Intercept", "x"]
    with pytest.raises(sp.exceptions.NumericalInstability):
        sp.regress("y ~ x + z", data=train, collinear="raise")


@pytest.mark.parametrize(
    "option", [{"robust": "hc1"}, {"cluster": "g"}, {"weights_as": "sampling"}]
)
def test_variance_options_are_refused(wide, option):
    wide = wide.assign(g=np.arange(len(wide)) % 10)
    kw = dict(option)
    if "weights_as" in kw:
        kw["weights"] = "w"
    with pytest.raises(MethodIncompatibility, match="collinear='pinv'"):
        sp.regress("y ~ a + b + c", data=wide, collinear="pinv", **kw)


def test_unknown_value_is_refused(wide):
    with pytest.raises(MethodIncompatibility, match="'omit', 'raise' or 'pinv'"):
        sp.regress("y ~ a + b", data=wide, collinear="svd")
