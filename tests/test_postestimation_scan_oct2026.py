"""Option scan (Oct 2026) of postestimation: test / lincom / nlcom /
hypotheses, margins, predictions with new data.

Every test asserts the correct behaviour, from an identity (the irr=False
fit, the stored fitted values) or the section 11 convention that a factor
level the fit never saw is refused. All failed before the fixes except the
three marked as suspects that were not defects. The irr=True predict test
is a strict xfail until core/results.py applies the coefficient scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _count_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    d["y"] = rng.poisson(np.exp(0.3 + 0.4 * d["x1"] - 0.2 * d["x2"]))
    return d


def _factor_frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {"x": rng.normal(size=n), "g": rng.integers(1, 4, size=n).astype(int)}
    )
    d["y"] = 1.0 + 0.5 * d["x"] + 0.7 * (d["g"] == 2) - 0.4 * (d["g"] == 3)
    d["y"] = d["y"] + rng.normal(size=n)
    return d


# --------------------------------------------------------------------- #
#  S1  irr=True: test / lincom / nlcom / hypotheses read exp(b) as b
# --------------------------------------------------------------------- #
# sp.poisson(irr=True) reports exp(b) and exp(b)*se(b) but keeps the b-scale
# covariance. Stata's test / lincom / nlcom after `poisson, irr` work on
# e(b) = b, so each answer must equal the one after the irr=False fit.


def test_test_after_irr_equals_test_on_coefficients():
    d = _count_frame()
    b = sp.poisson("y ~ x1 + x2", data=d)
    e = sp.poisson("y ~ x1 + x2", data=d, irr=True)
    for h in ("x1 = 0", "x1 = x2", "x1 = x2 = 0"):
        tb, te = sp.test(b, h), sp.test(e, h)
        assert te["statistic"] == pytest.approx(tb["statistic"], rel=1e-10)
        assert te["pvalue"] == pytest.approx(tb["pvalue"], rel=1e-10)


def test_lincom_and_nlcom_after_irr_equal_coefficient_scale():
    d = _count_frame()
    b = sp.poisson("y ~ x1 + x2", data=d)
    e = sp.poisson("y ~ x1 + x2", data=d, irr=True)
    lb, le = sp.lincom(b, "x1 + x2"), sp.lincom(e, "x1 + x2")
    assert le["estimate"] == pytest.approx(lb["estimate"], rel=1e-10)
    assert le["se"] == pytest.approx(lb["se"], rel=1e-10)
    nb, ne = sp.nlcom(b, "exp(x1) - 1"), sp.nlcom(e, "exp(x1) - 1")
    assert ne["estimate"] == pytest.approx(nb["estimate"], rel=1e-10)
    assert ne["se"] == pytest.approx(nb["se"], rel=1e-10)


def test_hypotheses_after_irr_equal_coefficient_scale():
    d = _count_frame()
    b = sp.poisson("y ~ x1 + x2", data=d)
    e = sp.poisson("y ~ x1 + x2", data=d, irr=True)
    hb, he = sp.hypotheses(b, "x1 - x2 = 0"), sp.hypotheses(e, "x1 - x2 = 0")
    assert he["estimate"].iloc[0] == pytest.approx(hb["estimate"].iloc[0], rel=1e-10)
    assert he["std_error"].iloc[0] == pytest.approx(hb["std_error"].iloc[0], rel=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason="S1b: EconometricResults.predict(newdata) uses exp(b) as b after "
    "irr=True; fix lives in core/results.py (Track C list) -> coordinator",
)
def test_predict_newdata_after_irr_equals_fitted_values():
    d = _count_frame()
    e = sp.poisson("y ~ x1 + x2", data=d, irr=True)
    np.testing.assert_allclose(
        e.predict(d.iloc[:10]),
        np.asarray(e.data_info["fitted_values"])[:10],
        rtol=1e-10,
    )


# --------------------------------------------------------------------- #
#  S2  a factor level the model never saw is predicted as the base level
# --------------------------------------------------------------------- #
# CLAUDE.md section 11: unseen levels must be rejected. core predict() does
# (reject_unseen_levels); margins_at / margins(at=) / predictions(newdata=)
# use the term-name design, where an unseen level gets zeros in every dummy.


def test_margins_at_rejects_unseen_factor_level():
    d = _factor_frame()
    fit = sp.regress("y ~ x + C(g)", data=d)
    with pytest.raises(MethodIncompatibility):
        sp.margins_at(fit, data=d, at={"g": [1, 4]})


def test_margins_at_option_rejects_unseen_factor_level():
    d = _factor_frame()
    fit = sp.logit(
        "yb ~ x + C(g)", data=d.assign(yb=(d["y"] > d["y"].median()).astype(int))
    )
    with pytest.raises(MethodIncompatibility):
        sp.margins(
            fit,
            data=d.assign(yb=(d["y"] > d["y"].median()).astype(int)),
            variables=["x"],
            at={"g": 7},
        )


def test_predictions_newdata_rejects_unseen_factor_level():
    d = _factor_frame()
    fit = sp.regress("y ~ x + C(g)", data=d)
    new = d.head(5).assign(g=9)
    with pytest.raises(MethodIncompatibility):
        sp.predictions(fit, newdata=new)


# --------------------------------------------------------------------- #
#  S3  new data rebuilt with patsy from scratch
# --------------------------------------------------------------------- #
# _FormulaDesign rebuilds the design_info on whatever frame it is handed, so
# (a) a stateful transform (center / standardize / bs) is re-estimated on
# the new rows, and (b) new rows that do not show every factor level are
# refused. In-sample rows must reproduce the stored fitted values.


def test_predictions_newdata_with_stateful_transform_match_fitted():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"x": rng.normal(5.0, 2.0, size=200)})
    d["y"] = 1.0 + 2.0 * d["x"] + rng.normal(size=200)
    fit = sp.regress("y ~ center(x)", data=d)
    fitted = np.asarray(fit.data_info["fitted_values"])
    p = sp.predictions(fit, newdata=d.iloc[:4])
    np.testing.assert_allclose(p["estimate"].to_numpy(), fitted[:4], rtol=1e-10)


def test_margins_standardize_slope_uses_the_fit_scale():
    # Suspected and not a defect: the fit and margins both memorise
    # standardize() on every row of data (patsy), so they agree. Kept as a
    # guard on the design rebuilt from the fitted frame.
    rng = np.random.default_rng(3)
    n = 300
    d = pd.DataFrame({"x": rng.normal(2.0, 3.0, size=n)})
    d["y"] = 0.5 + 1.5 * d["x"] + rng.normal(size=n)
    # rows outside the estimation sample whose x is far from the rest
    d.loc[:39, "y"] = np.nan
    d.loc[:39, "x"] = d.loc[:39, "x"] + 25.0
    fit = sp.regress("y ~ standardize(x)", data=d)
    # the scale the fit used, read off its own design column
    X = np.asarray(fit.data_info["X"], dtype=float)
    xs = d.loc[d["y"].notna(), "x"].to_numpy()
    scale = (xs[1] - xs[0]) / (X[1, 1] - X[0, 1])
    me = sp.margins(fit, data=d, variables=["x"])
    assert me["dy/dx"].iloc[0] == pytest.approx(fit.params.iloc[1] / scale, rel=1e-6)


def test_predictions_newdata_without_every_level_formula_backend():
    d = _factor_frame()
    fit = sp.regress("y ~ x + I(x ** 2) + C(g)", data=d)
    fitted = np.asarray(fit.data_info["fitted_values"])
    rows = np.flatnonzero(d["g"].to_numpy() == 2)[:3]
    p = sp.predictions(fit, newdata=d.iloc[rows])
    np.testing.assert_allclose(p["estimate"].to_numpy(), fitted[rows], rtol=1e-10)


# --------------------------------------------------------------------- #
#  S4  margins(subset=) with duplicated index labels
# --------------------------------------------------------------------- #


def test_margins_subset_with_duplicate_index_labels():
    # Suspected and not a defect (it passed before any change); kept as a
    # guard.
    d = _factor_frame()
    fit = sp.regress("y ~ x + C(g) + x:C(g)", data=d)
    ref = sp.margins(fit, data=d, subset="x > 0")
    dup = d.copy()
    dup.index = np.repeat(np.arange(len(d) // 2), 2)
    fit_dup = sp.regress("y ~ x + C(g) + x:C(g)", data=dup)
    out = sp.margins(fit_dup, data=dup, subset="x > 0")
    np.testing.assert_allclose(out["dy/dx"], ref["dy/dx"], rtol=1e-12)
    np.testing.assert_allclose(out["se"], ref["se"], rtol=1e-10)


# --------------------------------------------------------------------- #
#  S5  LDV margins with at= on a factor (probe: crash on valid input?)
# --------------------------------------------------------------------- #


def _ordered_frame(n: int = 600, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {"x": rng.normal(size=n), "g": rng.integers(1, 4, size=n).astype(int)}
    )
    latent = 0.8 * d["x"] + 0.5 * (d["g"] == 2) + rng.logistic(size=n)
    d["y"] = np.digitize(latent, [-0.5, 0.7])
    return d


def test_ologit_margins_at_factor_equals_margins_on_that_subpopulation():
    # Suspected crash, not a defect (it passed before any change); kept as
    # an identity check.
    d = _ordered_frame()
    fit = sp.ologit("y ~ x + C(g)", data=d)
    at = sp.margins(fit, data=d, variables=["x"], at={"g": 2})
    # identity: the average dy/dx with g set to 2 for every row equals the
    # derivative of the predictive margin at g = 2 (central difference)
    h = 1e-5
    up = sp.margins_at(fit, data=d.assign(x=d["x"] + h), at={"g": [2]})
    dn = sp.margins_at(fit, data=d.assign(x=d["x"] - h), at={"g": [2]})
    slope = (up["margin"].to_numpy() - dn["margin"].to_numpy()) / (2 * h)
    np.testing.assert_allclose(at["dy/dx"].to_numpy(), slope, rtol=1e-6)


# --------------------------------------------------------------------- #
#  S6  margins_table after a multiple-outcome model: one label per row
# --------------------------------------------------------------------- #


def test_margins_table_labels_are_unique_after_ologit():
    d = _ordered_frame()
    fit = sp.ologit("y ~ x + C(g)", data=d)
    tab = sp.margins_table(fit, data=d)
    assert tab.params.index.is_unique
    m = sp.margins(fit, data=d)
    assert len(tab.params) == len(m)


# --------------------------------------------------------------------- #
#  S7  hypotheses(): positional b1 and a coefficient named b1
# --------------------------------------------------------------------- #


def test_hypotheses_refuses_ambiguous_positional_name():
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"b1": rng.normal(size=200), "b2": rng.normal(size=200)})
    d["y"] = 1 + 2 * d["b1"] + rng.normal(size=200)
    fit = sp.regress("y ~ b1 + b2", data=d)
    with pytest.raises(MethodIncompatibility):
        sp.hypotheses(fit, "b1 = 0")


# --------------------------------------------------------------------- #
#  S9  nlcom edge cases raise the taxonomy error, not TypeError
# --------------------------------------------------------------------- #


def test_nlcom_fractional_power_of_negative_coefficient_is_refused():
    rng = np.random.default_rng(6)
    d = pd.DataFrame({"x": rng.normal(size=200)})
    d["y"] = 1 - 2 * d["x"] + rng.normal(size=200)
    fit = sp.regress("y ~ x", data=d)
    assert fit.params["x"] < 0
    with pytest.raises(MethodIncompatibility):
        sp.nlcom(fit, "x ** 0.5")


def test_lincom_validates_alpha():
    d = _factor_frame()
    fit = sp.regress("y ~ x", data=d)
    with pytest.raises(MethodIncompatibility):
        sp.lincom(fit, "x", alpha=5)


# --------------------------------------------------------------------- #
#  S10  margins(method='mem') after a weighted fit
# --------------------------------------------------------------------- #
# Found by the before/after snapshot: the at-means quantity is one row,
# averaged again with the n weights -> "shape-mismatch for sum".


def test_mem_after_weighted_logit_is_the_slope_at_the_weighted_means():
    rng = np.random.default_rng(8)
    n = 500
    d = pd.DataFrame({"x": rng.normal(size=n), "w": rng.uniform(0.5, 2.0, size=n)})
    d["yb"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 0.8 * d["x"])))).astype(
        int
    )
    fit = sp.logit("yb ~ x", data=d, weights="w")
    me = sp.margins(fit, data=d, method="mem")
    b = fit.params.to_numpy()
    xbar = np.average(d["x"], weights=d["w"])
    p = 1 / (1 + np.exp(-(b[0] + b[1] * xbar)))
    assert me["dy/dx"].iloc[0] == pytest.approx(p * (1 - p) * b[1], rel=1e-10)


def test_mem_after_weighted_regress_with_factor_runs_and_is_linear():
    d = _factor_frame().assign(w=lambda f: 1.0 + (f["x"] > 0))
    fit = sp.regress("y ~ x + C(g)", data=d, weights="w")
    me = sp.margins(fit, data=d, method="mem")
    # linear model: dy/dx and the discrete changes are the coefficients
    np.testing.assert_allclose(
        me.set_index("variable")["dy/dx"].loc[["x", "2.g", "3.g"]].to_numpy(),
        fit.params[["x", "C(g)[T.2]", "C(g)[T.3]"]].to_numpy(),
        rtol=1e-10,
    )
