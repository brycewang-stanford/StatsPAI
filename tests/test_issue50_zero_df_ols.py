"""GitHub #50: coefficient-only OLS at zero residual degrees of freedom.

A full-rank design with as many complete observations as parameters
identifies the coefficients exactly (the fit interpolates the data) but no
residual variance exists. ``coef_only=True`` returns those coefficients with
every inferential field NaN; without it the fit refuses and names the
option.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pl = pytest.importorskip("polars")

# the issue's three-observation, three-coefficient design: y = 2 + 3 x1 - x2
ISSUE = {"y": [2.0, 5.0, 1.0], "x1": [0.0, 1.0, 0.0], "x2": [0.0, 0.0, 1.0]}
EXACT = {"Intercept": 2.0, "x1": 3.0, "x2": -1.0}


def _fe_data() -> pd.DataFrame:
    # three groups of two rows, three slopes: 6 rows, 3 + 3 parameters
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "g": ["a", "a", "b", "b", "c", "c"],
            "x1": rng.normal(size=6),
            "x2": rng.normal(size=6),
            "x3": rng.normal(size=6),
            "y": rng.normal(size=6),
        }
    )


def _assert_inference_unavailable(res) -> None:
    for field in (res.std_errors, res.tvalues, res.pvalues):
        assert np.isnan(np.asarray(field, dtype=float)).all()
    assert np.isnan(res.conf_int().to_numpy(dtype=float)).all()
    assert res.data_info["df_resid"] == 0
    assert res.model_info["inference"] == "unavailable"
    assert "zero residual degrees of freedom" in res.model_info["inference_reason"]
    assert res.inference_rule()["distribution"] == "none"


@pytest.mark.parametrize(
    "fit",
    [
        lambda d, **kw: sp.regress("y ~ x1 + x2", data=d, **kw),
        lambda d, **kw: sp.feols("y ~ x1 + x2", data=d, vcov="iid", **kw),
    ],
    ids=["regress", "feols"],
)
def test_zero_df_refuses_by_default_and_names_the_option(fit):
    with pytest.raises(sp.DataInsufficient, match="residual df=0") as info:
        fit(pl.DataFrame(ISSUE))
    assert "coef_only=True" in info.value.recovery_hint


@pytest.mark.parametrize(
    "fit",
    [
        lambda d: sp.regress("y ~ x1 + x2", data=d, robust="nonrobust", coef_only=True),
        lambda d: sp.feols("y ~ x1 + x2", data=d, vcov="iid", coef_only=True),
    ],
    ids=["regress", "feols"],
)
def test_coef_only_returns_the_exact_solution(fit):
    res = fit(pl.DataFrame(ISSUE))
    # the exact solve of the 3 x 3 system (the issue's statsmodels numbers)
    X = np.column_stack([np.ones(3), ISSUE["x1"], ISSUE["x2"]])
    b = np.linalg.solve(X, np.asarray(ISSUE["y"]))
    np.testing.assert_allclose(res.params.to_numpy(), b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        res.params.reindex(list(EXACT)).to_numpy(), list(EXACT.values()), atol=1e-12
    )
    _assert_inference_unavailable(res)
    assert res.diagnostics["R-squared"] == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(res.diagnostics["Adj. R-squared"])
    # the table and the summary still render the coefficients
    assert "Inference: unavailable" in res.summary()
    assert "-1.0" in str(sp.regtable(res))


def test_feols_coef_only_counts_the_fixed_effects():
    d = _fe_data()
    # 6 rows, 3 slopes + 3 group effects: zero residual df only with the
    # fixed effects counted
    with pytest.raises(sp.DataInsufficient, match="fixed effects included") as info:
        sp.feols("y ~ x1 + x2 + x3 | g", data=d, vcov="iid", backend="native")
    assert "coef_only=True" in info.value.recovery_hint
    res = sp.feols("y ~ x1 + x2 + x3 | g", data=d, coef_only=True)
    # the slopes of the dummy-variable regression, solved exactly
    X = np.column_stack(
        [
            pd.get_dummies(d["g"]).to_numpy(dtype=float),
            d[["x1", "x2", "x3"]].to_numpy(),
        ]
    )
    b = np.linalg.solve(X, d["y"].to_numpy())
    np.testing.assert_allclose(res.params.to_numpy(), b[3:], rtol=1e-9, atol=1e-12)
    _assert_inference_unavailable(res)


def test_regress_coef_only_with_residual_df_still_withholds_inference():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=20)})
    d["y"] = 1 + 2 * d["x"] + rng.normal(size=20)
    full = sp.regress("y ~ x", data=d)
    res = sp.regress("y ~ x", data=d, coef_only=True)
    np.testing.assert_allclose(res.params, full.params, rtol=0, atol=1e-14)
    assert np.isnan(res.std_errors.to_numpy()).all()
    assert res.model_info["inference"] == "unavailable"
    assert res.model_info["inference_reason"].startswith("coef_only=True")


def test_rank_deficient_design_still_refuses():
    # x2 duplicates x1: three rows identify only two coefficients
    d = pd.DataFrame({"y": [2.0, 5.0, 1.0], "x1": [0.0, 1.0, 3.0]})
    d["x2"] = d["x1"]
    with pytest.raises(sp.StatsPAIError):
        sp.regress("y ~ x1 + x2", data=d, coef_only=True, collinear="raise")
    # an underdetermined design (fewer rows than coefficients) is never fit:
    # collinear='omit' first drops what the rows cannot identify ...
    two = pd.DataFrame(ISSUE).iloc[:2]
    with pytest.warns(UserWarning, match="x2 omitted"):
        res = sp.regress("y ~ x1 + x2", data=two, coef_only=True)
    assert list(res.params.index) == ["Intercept", "x1"]
    # ... and collinear='raise' refuses it
    with pytest.raises(sp.StatsPAIError):
        sp.regress("y ~ x1 + x2", data=two, coef_only=True, collinear="raise")


@pytest.mark.parametrize(
    "kwargs",
    [{"robust": "hc1"}, {"cluster": "x1"}, {"vce": "hc3"}],
)
def test_regress_coef_only_refuses_a_variance_option(kwargs):
    with pytest.raises(sp.MethodIncompatibility, match="coef_only=True"):
        sp.regress("y ~ x1 + x2", data=pd.DataFrame(ISSUE), coef_only=True, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"vcov": "hetero"}, {"cluster": "x1"}, {"backend": "pyfixest"}],
)
def test_feols_coef_only_refuses_a_variance_option(kwargs):
    with pytest.raises(sp.MethodIncompatibility, match="coef_only=True"):
        sp.feols("y ~ x1 + x2", data=pd.DataFrame(ISSUE), coef_only=True, **kwargs)
