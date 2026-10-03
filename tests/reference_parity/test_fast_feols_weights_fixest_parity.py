"""``sp.fast.feols`` against R ``fixest``: weights, variance kinds, FE layouts.

Fixture: ``_fixtures/fast_feols_weights.csv`` (unbalanced two-way panel,
1,042 rows, 150 firms, 8 years, 25 clusters nesting the firms) and
``_fixtures/fast_feols_weights_R.json`` from
``_generate_fast_feols_weights_R.R`` (fixest 0.14.0, default ``ssc()``).

What is pinned, all at the default ``ssc='fixest'``:

* one and two absorbed dimensions x no weights / weights x ``iid`` /
  ``hc1`` / ``cr1``: coefficients and standard errors;
* nine clustered layouts that differ in which absorbed dimensions the
  cluster key nests, because fixest drops the nested ones from ``K``.

The defect this caught (2026-10-03): with **one** absorbed dimension the
degrees of freedom were ``n - p - (G - 1)``; fixest and reghdfe use
``n - p - G``, since the effects span the intercept. IID and HC1 standard
errors were ``sqrt((n - p - G)/(n - p - G + 1))`` of fixest's (5.6e-4
low here), and clustered ones were ``sqrt((n - p - 1)/(n - p))`` of
fixest's when the single dimension was nested in the clusters. With two
absorbed dimensions neither nested in the clusters the count was one too
many. Two-way models clustered on a nesting key, which is what the Track A
modules run, were exact before and are unchanged.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
#: Closed form on both sides; the two-way demeaning is iterative, and its
#: worst measured gap is 6.3e-10 (weighted, clustered).
RTOL = 1e-8
FORMULAS = {"one": "y ~ x1 + x2 | firm", "two": "y ~ x1 + x2 | firm + year"}
VCOV = {
    "iid": dict(vcov="iid"),
    "hetero": dict(vcov="hc1"),
    "cluster": dict(vcov="cr1", cluster="g"),
}


@pytest.fixture(scope="module")
def ref():
    path = _FIX / "fast_feols_weights_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_fast_feols_weights_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(_FIX / "fast_feols_weights.csv")
    df["c2"] = (df["firm"] * 7 + df["year"]) % 20
    return df


def _values(fit):
    names = list(fit.coef_names)
    se = np.sqrt(np.diag(fit.vcov_matrix))
    return (
        {n: float(b) for n, b in zip(names, fit.coef_vec)},
        {n: float(s) for n, s in zip(names, se)},
    )


def _assert_matches(fit, cell):
    b, se = _values(fit)
    for name in ("x1", "x2"):
        assert b[name] == pytest.approx(cell[f"b_{name}"], rel=RTOL)
        assert se[name] == pytest.approx(cell[f"se_{name}"], rel=RTOL)


@pytest.mark.parametrize("fe", sorted(FORMULAS))
@pytest.mark.parametrize("weights", ["none", "set"])
@pytest.mark.parametrize("vcov", sorted(VCOV))
def test_matches_fixest(ref, data, fe, weights, vcov):
    fit = sp.fast.feols(
        FORMULAS[fe],
        data,
        weights="w" if weights == "set" else None,
        **VCOV[vcov],
    )
    _assert_matches(fit, ref[f"{fe}_{weights}_{vcov}"])


@pytest.mark.parametrize("fe", ["firm + year", "firm", "year"])
@pytest.mark.parametrize("cluster", ["c2", "year", "g"])
def test_clustered_layouts_match_fixest(ref, data, fe, cluster):
    fit = sp.fast.feols(f"y ~ x1 + x2 | {fe}", data, vcov="cr1", cluster=cluster)
    key = "layout_" + fe.replace(" + ", "_") + "_" + cluster
    _assert_matches(fit, ref[key])


def test_one_dimension_costs_its_number_of_levels(data):
    fit = sp.fast.feols(FORMULAS["one"], data, vcov="iid")
    n, g = len(data), data["firm"].nunique()
    assert fit.df_resid == n - 2 - g
    two = sp.fast.feols(FORMULAS["two"], data, vcov="iid")
    assert two.df_resid == n - 2 - (g + data["year"].nunique() - 1)


def test_hdfe_ols_and_feols_agree_with_the_same_reference(ref, data):
    """The other two fixed-effects entry points, on the one-dimension cells."""
    for fn in (sp.feols, sp.hdfe_ols):
        for weights in (None, "w"):
            fit = fn(FORMULAS["one"], data, weights=weights, cluster="g")
            cell = ref[f"one_{'set' if weights else 'none'}_cluster"]
            for name in ("x1", "x2"):
                assert float(fit.std_errors[name]) == pytest.approx(
                    cell[f"se_{name}"], rel=RTOL
                )
