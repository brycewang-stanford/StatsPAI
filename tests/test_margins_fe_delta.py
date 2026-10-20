"""Margins after ``sp.fepois`` count the absorbed effects as parameters.

Reference: Stata 18 MP ``margins`` after ``poisson y x i.g`` (one-way,
``fixtures/margins_fe/fep.csv``) and ``poisson y x1 x2 d i.g i.h`` (two-way,
``fep2.csv``, from ``_make_fep2.py``), with ``vce(oim)``, ``vce(robust)``,
``vce(cluster g)`` and ``[pw=w], vce(cluster g)``
(``stata_margins_fe_delta.do``). ``ppmlhdfe ..., a(g h) d`` + ``margins``
gives the same numbers to 1e-8 (also in the JSON).

``sp.fepois`` keeps fixest's small-sample factor (``(n-1)/(n-K)`` on top of
Stata's ``G/(G-1)``), a scalar on the whole covariance, so the comparison
rescales by the ratio of the coefficient SEs (Stata over ours, read off the
same fits) and then asks for 1e-8 on the one-way fixture and 1e-6 on the
two-way one (the fit's own two-way SEs agree with Stata's to 5e-7 across
coefficients; that is the fit's projection tolerance, not the margin's).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

F = Path(__file__).parent / "fixtures" / "margins_fe"
R = json.loads((F / "stata_margins_fe_delta.json").read_text(encoding="utf-8"))
D1 = pd.read_csv(F / "fep.csv")
D2 = pd.read_csv(F / "fep2.csv")
VCE = {"oim": "iid", "robust": "hetero", "cluster": {"CRV1": "g"}}


def _se(rows):
    return np.array([r[1] for r in rows])


def _est(rows):
    return np.array([r[0] for r in rows])


@pytest.mark.parametrize("vce", list(VCE))
def test_one_way_margins_match_stata(vce):
    r = sp.fepois("y ~ x | g", data=D1, vcov=VCE[vce])
    k = R[f"fep_{vce}_bx"][1] / r.std_errors["x"]
    m = sp.margins(r, data=D1)
    np.testing.assert_allclose(m["dy/dx"], _est(R[f"fep_{vce}_dydx"]), rtol=1e-9)
    np.testing.assert_allclose(m["se"] * k, _se(R[f"fep_{vce}_dydx"]), rtol=1e-8)
    at = sp.margins_at(r, data=D1, at={"x": [0, 1]})
    np.testing.assert_allclose(at["margin"], _est(R[f"fep_{vce}_at"]), rtol=1e-9)
    np.testing.assert_allclose(at["se"] * k, _se(R[f"fep_{vce}_at"]), rtol=1e-8)
    c = sp.margins_at(r, data=D1, at={"x": [0, 1]}, contrast="first")
    np.testing.assert_allclose(c["se"] * k, _se(R[f"fep_{vce}_atcontrast"]), rtol=1e-8)
    # the marginaleffects functions give the same numbers
    s = sp.avg_slopes(r)
    np.testing.assert_allclose(s["std_error"] * k, _se(R[f"fep_{vce}_dydx"]), rtol=1e-8)
    p = sp.avg_predictions(r, newdata=D1.assign(x=1.0))
    np.testing.assert_allclose(
        p["std_error"] * k, _se(R[f"fep_{vce}_at"])[1:], rtol=1e-8
    )
    cmp = sp.avg_comparisons(r, variables={"x": (0, 1)})
    np.testing.assert_allclose(
        cmp["std_error"] * k, _se(R[f"fep_{vce}_atcontrast"]), rtol=1e-8
    )


@pytest.mark.parametrize("vce", list(VCE))
def test_two_way_margins_match_stata(vce):
    r = sp.fepois("y ~ x1 + x2 + d | g + h", data=D2, vcov=VCE[vce])
    k = R[f"fep2_{vce}_b"][0][1] / r.std_errors["x1"]
    m = sp.margins(r, data=D2)
    np.testing.assert_allclose(m["dy/dx"], _est(R[f"fep2_{vce}_dydx"]), rtol=1e-8)
    np.testing.assert_allclose(m["se"] * k, _se(R[f"fep2_{vce}_dydx"]), rtol=1e-6)
    at = sp.margins_at(r, data=D2, at={"d": [0, 1]})
    np.testing.assert_allclose(at["se"] * k, _se(R[f"fep2_{vce}_at"]), rtol=1e-6)
    c = sp.margins_at(r, data=D2, at={"d": [0, 1]}, contrast="first")
    np.testing.assert_allclose(c["se"] * k, _se(R[f"fep2_{vce}_atcontrast"]), rtol=1e-6)


def test_weighted_clustered_margins_match_stata():
    r = sp.fepois("y ~ x1 + x2 + d | g + h", data=D2, vcov={"CRV1": "g"}, weights="w")
    k = R["fep2_w_b"][0][1] / r.std_errors["x1"]
    m = sp.margins(r, data=D2)
    np.testing.assert_allclose(m["dy/dx"], _est(R["fep2_w_dydx"]), rtol=1e-8)
    np.testing.assert_allclose(m["se"] * k, _se(R["fep2_w_dydx"]), rtol=1e-6)
    # marginaleffects reports a 0/1 regressor as the 1 - 0 contrast, which
    # Stata gives as margins, at(d=(0 1)) contrast(atcontrast(r))
    s = sp.avg_slopes(r, wts=True)
    np.testing.assert_allclose(
        s["std_error"][:2] * k, _se(R["fep2_w_dydx"])[:2], rtol=1e-6
    )
    np.testing.assert_allclose(
        s["estimate"][2], _est(R["fep2_w_atcontrast"])[0], rtol=1e-8
    )
    np.testing.assert_allclose(
        s["std_error"][2] * k, _se(R["fep2_w_atcontrast"])[0], rtol=1e-6
    )


def test_ppmlhdfe_agrees_with_dummy_poisson():
    """The reference itself: Stata's margins after ppmlhdfe equal those after
    poisson with the effects as dummies (so the target is unambiguous)."""
    np.testing.assert_allclose(
        _se(R["fep2_ppml_dydx"]), _se(R["fep2_cluster_dydx"]), rtol=1e-7
    )


def test_one_way_oim_closed_form():
    """Identity on the one-way fixture: the Poisson score equations make the
    AME b * mean(y), and the effects add b^2 mean(y) / n to the variance:
    Var = c (mean(y)^2 / S + b^2 mean(y) / n), with c S^-1 the fit's Var(b)
    and S = sum mu x~^2 (x~ demeaned within g with weights mu)."""
    r = sp.fepois("y ~ x | g", data=D1, vcov="iid")
    m = sp.margins(r, data=D1)
    mu = r.data_info["fitted_values"]
    x = D1["x"].to_numpy()
    g = D1["g"].to_numpy()
    xbar = pd.Series(mu * x).groupby(g).transform("sum") / pd.Series(mu).groupby(
        g
    ).transform("sum")
    S = float(np.sum(mu * (x - xbar.to_numpy()) ** 2))
    c = r.std_errors["x"] ** 2 * S
    b, ybar, n = r.params["x"], D1["y"].mean(), len(D1)
    want = np.sqrt(c * (ybar**2 / S + b**2 * ybar / n))
    np.testing.assert_allclose(m["se"], [want], rtol=1e-8)


def test_hypotheses_on_the_table_use_the_same_covariance():
    r = sp.fepois("y ~ x1 + x2 + d | g + h", data=D2, vcov={"CRV1": "g"})
    s = sp.avg_slopes(r, variables=["x1", "x2"])
    h = sp.hypotheses(s, "b1 = 0")
    np.testing.assert_allclose(h["std_error"], s["std_error"][:1], rtol=1e-10)
    m = sp.margins_at(r, data=D2, at={"x1": [0, 1]}, contrast="first")
    d = sp.avg_comparisons(r, variables={"x1": (0, 1)})
    np.testing.assert_allclose(d["std_error"], m["se"], rtol=1e-10)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: sp.slopes(r),  # unit-level quantities
        lambda r: sp.margins(r, data=D2, method="mem"),
    ],
)
def test_still_refused(call):
    r = sp.fepois("y ~ x1 + x2 + d | g + h", data=D2)
    with pytest.raises(MethodIncompatibility, match="vcov=False"):
        call(r)


def test_logit_with_absorbed_effects_still_refused():
    d = D2.assign(yb=(D2["y"] > 0).astype(int))
    r = sp.feglm("yb ~ x1 + x2 | h", data=d, family="logit")
    with pytest.raises(MethodIncompatibility, match="vcov=False"):
        sp.margins(r, data=d)


def test_two_way_clustering_is_refused():
    r = sp.fepois("y ~ x1 + x2 + d | g + h", data=D2, vcov={"CRV1": "g + h"})
    with pytest.raises(MethodIncompatibility, match="scalar multiple|one-way"):
        sp.margins(r, data=D2)
