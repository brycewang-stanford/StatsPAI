"""sp.predictions / comparisons / slopes / hypotheses: identities and edges.

The R parity lives in ``tests/reference_parity/test_marginaleffects_r_parity.py``;
here every assertion is an identity that holds without a reference: the
Stata-convention ``sp.margins`` reproduced exactly, closed forms of linear
models, and the delta method written out by hand.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    n = 500
    d = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": rng.integers(0, 3, n),
            "b": rng.integers(0, 2, n),
            "w": rng.uniform(0.5, 2.0, n),
        }
    )
    eta = 0.2 + d.x - 0.4 * d.z + 0.5 * d.b + 0.3 * d.x * d.b
    d["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    d["yc"] = 1 + d.x + 0.5 * d.x**2 + 2 * d.b + 0.3 * d.g + rng.normal(size=n)
    return d


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


@pytest.mark.parametrize("weights", [None, "w"])
def test_weighted_model_scale_ame_is_stata_margins(data, weights):
    fit = _quiet(sp.logit, "y ~ x*b + z + C(g)", data=data, weights=weights)
    ref = sp.margins(fit, data, variables=["x", "z", "g"])
    me = sp.avg_slopes(fit, ["x", "z", "g"], wts=weights is not None, df="model")
    np.testing.assert_allclose(me.estimate, ref["dy/dx"], rtol=1e-10)
    np.testing.assert_allclose(me.std_error, ref["se"], rtol=1e-10)
    np.testing.assert_allclose(me.p_value, ref["pvalue"], rtol=1e-8)


def test_linear_model_contrast_is_the_coefficient(data):
    fit = _quiet(sp.regress, "yc ~ x + b + z", data=data)
    me = sp.avg_comparisons(fit, "b")
    assert me.contrast.iloc[0] == "1 - 0"
    assert me.estimate.iloc[0] == pytest.approx(fit.params["b"], rel=1e-12)
    assert me.std_error.iloc[0] == pytest.approx(fit.std_errors["b"], rel=1e-10)


def test_slope_differentiates_through_a_square(data):
    fit = _quiet(sp.regress, "yc ~ x + I(x**2) + b", data=data)
    b1, b2 = fit.params["x"], fit.params["I(x ** 2)"]
    s = sp.slopes(fit, "x")
    # I(x**2) is differentiated by a central difference (exact for a
    # quadratic up to rounding), hence the absolute tolerance.
    np.testing.assert_allclose(s.estimate, b1 + 2 * b2 * data.x, rtol=1e-8, atol=1e-9)
    # The SE of b1 + 2 b2 x is the delta method on (1, 2x).
    V = pd.DataFrame(
        np.asarray(fit.cov_params()), index=fit.params.index, columns=fit.params.index
    )
    sub = V.loc[["x", "I(x ** 2)"], ["x", "I(x ** 2)"]].to_numpy()
    g = np.column_stack([np.ones(len(data)), 2 * data.x])
    se = np.sqrt(np.einsum("ij,jk,ik->i", g, sub, g))
    np.testing.assert_allclose(s.std_error, se, rtol=1e-8)


def test_elasticities_are_rescaled_slopes(data):
    fit = _quiet(sp.logit, "y ~ x + z", data=data)
    p = sp.predictions(fit).estimate.to_numpy()
    s = sp.slopes(fit, "x").estimate.to_numpy()
    x = data.x.to_numpy()
    for kind, want in (("eyex", s * x / p), ("eydx", s / p), ("dyex", s * x)):
        np.testing.assert_allclose(sp.slopes(fit, "x", slope=kind).estimate, want)


def test_by_groups_average_the_unit_rows(data):
    fit = _quiet(sp.logit, "y ~ x*b + z + C(g)", data=data)
    unit = sp.predictions(fit)
    by = sp.avg_predictions(fit, by="g")
    for _, row in by.iterrows():
        want = unit.loc[unit.g == row.g, "estimate"].mean()
        assert row.estimate == pytest.approx(want, rel=1e-12)


def test_newdata_predictions_are_the_inverse_link(data):
    fit = _quiet(sp.logit, "y ~ x + z", data=data)
    new = pd.DataFrame({"x": [-1.0, 0.0, 2.0], "z": [0.5, 0.0, -1.0]})
    p = sp.predictions(fit, new)
    eta = fit.params["Intercept"] + fit.params["x"] * new.x + fit.params["z"] * new.z
    np.testing.assert_allclose(p.estimate, 1 / (1 + np.exp(-eta)), rtol=1e-12)
    np.testing.assert_allclose(sp.predictions(fit, new, type="link").estimate, eta)


def test_missing_rows_are_outside_the_estimation_sample(data):
    d = data.copy()
    d.loc[:9, "x"] = np.nan
    fit = _quiet(sp.logit, "y ~ x + z", data=d)
    assert len(sp.predictions(fit)) == len(d) - 10


def test_change_specifications(data):
    fit = _quiet(sp.logit, "y ~ x + z", data=data)
    x = data.x
    mean_sd = sp.avg_comparisons(fit, {"x": "sd"}).estimate.iloc[0]
    lo = x.mean() - x.std() / 2
    hi = x.mean() + x.std() / 2
    p = sp.avg_predictions
    want = (
        p(fit, data.assign(x=hi)).estimate.iloc[0]
        - p(fit, data.assign(x=lo)).estimate.iloc[0]
    )
    assert mean_sd == pytest.approx(want, rel=1e-12)
    pair = sp.avg_comparisons(fit, {"x": (0, 1)})
    assert pair.contrast.iloc[0] == "1 - 0"
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = sp.avg_comparisons(fit, {"x": "iqr"}).estimate.iloc[0]
    want = (
        p(fit, data.assign(x=q3)).estimate.iloc[0]
        - p(fit, data.assign(x=q1)).estimate.iloc[0]
    )
    assert iqr == pytest.approx(want, rel=1e-12)


def test_ratio_of_averages_when_averaging(data):
    fit = _quiet(sp.logit, "y ~ x + b", data=data)
    avg = sp.avg_comparisons(fit, "b", comparison="ratio")
    p1 = sp.avg_predictions(fit, data.assign(b=1)).estimate.iloc[0]
    p0 = sp.avg_predictions(fit, data.assign(b=0)).estimate.iloc[0]
    assert avg.contrast.iloc[0] == "mean(1) / mean(0)"
    assert avg.estimate.iloc[0] == pytest.approx(p1 / p0, rel=1e-12)
    unit = sp.comparisons(fit, "b", comparison="ratio")
    assert len(unit) == len(data)
    assert unit.contrast.iloc[0] == "1 / 0"


def test_hypotheses_on_coefficients_is_the_delta_method(data):
    fit = _quiet(sp.regress, "yc ~ x + z", data=data)
    names = list(fit.params.index)
    i, j = names.index("x"), names.index("z")
    h = sp.hypotheses(fit, f"b{i + 1} = b{j + 1}")
    V = np.asarray(fit.cov_params())
    se = np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])
    assert h.estimate.iloc[0] == pytest.approx(
        fit.params["x"] - fit.params["z"], rel=1e-12
    )
    assert h.std_error.iloc[0] == pytest.approx(se, rel=1e-10)
    # Names work too, and "lhs = rhs" equals "lhs - rhs".
    by_name = sp.hypotheses(fit, "x - z")
    assert by_name.estimate.iloc[0] == pytest.approx(h.estimate.iloc[0])
    # Normal reference by default, the fit's t with df='model'.
    t = sp.hypotheses(fit, "x - z", df="model")
    assert t.conf_high.iloc[0] > by_name.conf_high.iloc[0]
    assert by_name.p_value.iloc[0] == pytest.approx(
        2 * stats.norm.sf(abs(by_name.statistic.iloc[0]))
    )


def test_hypotheses_on_a_table_uses_the_joint_covariance(data):
    fit = _quiet(sp.logit, "y ~ x + C(g)", data=data)
    tab = sp.avg_comparisons(fit, "g")  # 1 - 0, 2 - 0
    h = sp.hypotheses(tab, "b2 - b1 = 0")
    direct = (
        sp.avg_predictions(fit, data.assign(g=2)).estimate.iloc[0]
        - sp.avg_predictions(fit, data.assign(g=1)).estimate.iloc[0]
    )
    assert h.estimate.iloc[0] == pytest.approx(direct, rel=1e-10)
    J = tab.attrs["jacobian"]
    V = tab.attrs["vcov_coefficients"]
    g = J[1] - J[0]
    assert h.std_error.iloc[0] == pytest.approx(np.sqrt(g @ V @ g), rel=1e-12)


# --------------------------------------------------------------------------
# Edges: every misuse raises, nothing returns NaN silently
# --------------------------------------------------------------------------


def test_unknown_variable(data):
    fit = _quiet(sp.logit, "y ~ x + z", data=data)
    with pytest.raises(MethodIncompatibility, match="not variables of the model"):
        sp.avg_slopes(fit, "nope")


def test_bad_options(data):
    fit = _quiet(sp.logit, "y ~ x + z", data=data)
    with pytest.raises(MethodIncompatibility, match="comparison must be"):
        sp.avg_comparisons(fit, "x", comparison="odds")
    with pytest.raises(MethodIncompatibility, match="slope must be"):
        sp.avg_slopes(fit, "x", slope="elasticity")
    with pytest.raises(MethodIncompatibility, match="change for 'x'"):
        sp.avg_comparisons(fit, {"x": "huge"})
    with pytest.raises(MethodIncompatibility, match="by= column"):
        sp.avg_predictions(fit, by="nope")
    with pytest.raises(MethodIncompatibility, match="fit had no weight"):
        sp.avg_predictions(fit, wts=True)


def test_bad_hypotheses(data):
    fit = _quiet(sp.logit, "y ~ x + z", data=data)
    with pytest.raises(MethodIncompatibility, match="out of range"):
        sp.hypotheses(fit, "b9 = 0")
    with pytest.raises(MethodIncompatibility, match="single '='"):
        sp.hypotheses(fit, "x == z")
    with pytest.raises(MethodIncompatibility, match="Jacobian"):
        sp.hypotheses(pd.DataFrame({"estimate": [1.0]}), "b1 = 0")


def test_result_without_coefficients():
    with pytest.raises(MethodIncompatibility, match="params"):
        sp.avg_predictions(object())
