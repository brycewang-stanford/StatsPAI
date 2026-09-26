"""sp.mi_test against Stata 18 ``mi test`` and ``mi test, nosmall`` (T2).

Fixture: ``_fixtures/mi_test_flong.csv`` holds the original data and eight
Stata ``mi impute chained`` completions (flong, ``%21.0g``); the reference
numbers come from ``_fixtures/_generate_mi_test_stata.do`` run on that same
file. The ``m8_*`` / ``m3_*`` cells are the original large-sample checks.
The ``grid`` block crosses sample size (complete-data df 396 / 26 / 11 / 3),
imputations (2, 3, 4, 5, 8), test (equal FMI / ``ufmitest``), k = 1..4
tested terms and small-sample (Stata's default) / ``nosmall`` df: every df
branch, including Reiter (2007) outside its range (df 3), where Stata's
number is reproduced and a warning raised.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility
from statspai.imputation.mice import _rubins_rules

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "mi_test_stata.json").read_text(encoding="utf-8"))
FLONG = pd.read_csv(FIX / "mi_test_flong.csv")


_POOLED: dict = {}


def _pooled(m: int, n: int = 400):
    if (m, n) in _POOLED:
        return _POOLED[m, n]
    ests, names = [], None
    for j in range(1, m + 1):
        rows = (FLONG["_mi_m"] == j) & (FLONG["_mi_id"] <= n)
        fit = sp.regress("y ~ x1 + x2 + x3", data=FLONG[rows])
        ests.append(
            {
                "params": fit.params.to_numpy(),
                "var_cov": np.asarray(fit.data_info["var_cov"]),
                "df_resid": fit.data_info["df_resid"],
            }
        )
        names = list(fit.params.index)
    out = _rubins_rules(ests)
    out["var_names"] = names
    _POOLED[m, n] = out
    return out


CASES = [
    (tag, m, kind, terms)
    for tag, m in (("m8", 8), ("m3", 3))
    for kind, terms in (
        ("equal", ["x2", "x3"]),
        ("unres", ["x2", "x3"]),
        ("equal3", ["x1", "x2", "x3"]),
    )
]


@pytest.mark.parametrize("tag,m,kind,terms", CASES)
def test_mi_test_matches_stata_nosmall(tag, m, kind, terms):
    ref = REF[f"{tag}_{kind}"]
    method = "unrestricted" if kind == "unres" else "equal_fmi"
    res = sp.mi_test(_pooled(m), terms, method=method, small=False)
    assert res["df1"] == ref["df1"]
    np.testing.assert_allclose(res["F"], ref["F"], rtol=1e-9)
    np.testing.assert_allclose(res["df2"], ref["df2"], rtol=1e-9)
    np.testing.assert_allclose(res["pvalue"], ref["p"], rtol=1e-9)


GRID = REF["grid"]


def _grid_id(c):
    df = "small" if c["small"] else "nosmall"
    return f"n{c['n']}-m{c['m']}-{c['kind']}-k{c['df1']}-{df}"


@pytest.mark.parametrize("cell", GRID, ids=[_grid_id(c) for c in GRID])
def test_mi_test_matches_stata_grid(cell):
    pooled = _pooled(cell["m"], cell["n"])
    assert pooled["dfcom"] == cell["dfcom"]
    terms = ["Intercept" if t == "_cons" else t for t in cell["terms"].split()]
    method = "unrestricted" if cell["kind"] == "unres" else "equal_fmi"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)
        res = sp.mi_test(pooled, terms, method=method, small=bool(cell["small"]))
    assert res["df1"] == cell["df1"]
    assert res["df_adjustment"] == ("small" if cell["small"] else "large")
    # F does not depend on the df; df2 and p are closed forms of the same
    # pooled U, B -- 1e-9 leaves room only for the 17-digit export.
    np.testing.assert_allclose(res["F"], cell["F"], rtol=1e-9)
    np.testing.assert_allclose(res["df2"], cell["df2"], rtol=1e-9)
    np.testing.assert_allclose(res["pvalue"], cell["p"], rtol=1e-9)


def test_grid_covers_every_df_branch():
    branches = set()
    for c in GRID:
        k, t = c["df1"], c["df1"] * (c["m"] - 1)
        branches.add(
            (
                c["kind"],
                bool(c["small"]),
                "k1" if k == 1 else ("t>4" if t > 4 else "t<=4"),
            )
        )
    assert len(branches) == 2 * 2 * 3
    # The switch is on k(m-1) (Stata manual), not m(k-1) (Marchenko-Reiter
    # 2009): cells where the two disagree must be present in both directions.
    assert any(c["df1"] * (c["m"] - 1) > 4 >= c["m"] * (c["df1"] - 1) for c in GRID)
    assert any(c["df1"] * (c["m"] - 1) <= 4 < c["m"] * (c["df1"] - 1) for c in GRID)


def test_small_is_the_default_and_matches_stata_default():
    ref = next(
        c
        for c in GRID
        if (c["n"], c["m"], c["kind"], c["terms"], c["small"])
        == (30, 5, "equal", "x2 x3", 1)
    )
    res = sp.mi_test(_pooled(5, 30), ["x2", "x3"])
    assert res["df_adjustment"] == "small"
    np.testing.assert_allclose(res["df2"], ref["df2"], rtol=1e-9)


def test_reiter_out_of_range_warns():
    # Complete-data df 3: Reiter's denominator nu_c* - 4(1 + a) is negative.
    with pytest.warns(AssumptionWarning, match="outside"):
        res = sp.mi_test(_pooled(8, 7), ["x2", "x3"])
    assert res["df2"] < 4


def test_small_without_complete_data_df_falls_back_to_large():
    pooled = dict(_pooled(3, 30))
    pooled["dfcom"] = np.inf
    res = sp.mi_test(pooled, ["x2", "x3"])
    ref = sp.mi_test(_pooled(3, 30), ["x2", "x3"], small=False)
    assert res["df_adjustment"] == "large"
    assert res["df2"] == ref["df2"]


def test_bad_terms_and_method_raise():
    pooled = _pooled(3)
    with pytest.raises(MethodIncompatibility, match="unknown"):
        sp.mi_test(pooled, ["nope"])
    with pytest.raises(MethodIncompatibility, match="method"):
        sp.mi_test(pooled, ["x2"], method="d3")
    with pytest.raises(MethodIncompatibility, match="ubar_matrix"):
        sp.mi_test({"params": [0.0]}, ["x2"])


def test_no_between_variance_reduces_to_complete_data_wald():
    fit = sp.regress("y ~ x1 + x2 + x3", data=FLONG[FLONG["_mi_m"] == 1])
    est = {
        "params": fit.params.to_numpy(),
        "var_cov": np.asarray(fit.data_info["var_cov"]),
    }
    pooled = _rubins_rules([est, dict(est)])
    pooled["var_names"] = list(fit.params.index)
    res = sp.mi_test(pooled, ["x2", "x3"], small=False)
    idx = [pooled["var_names"].index(t) for t in ("x2", "x3")]
    q = est["params"][idx]
    V = est["var_cov"][np.ix_(idx, idx)]
    assert res["df2"] == np.inf
    np.testing.assert_allclose(res["F"], q @ np.linalg.solve(V, q) / 2, rtol=1e-12)


def test_no_between_variance_small_sample_df():
    # B = 0: no information is missing, so nu_obs = nu_c (nu_c+1)/(nu_c+3).
    # Stata 18 on two identical completions (n = 400, nu_c = 396) reports
    # 394.01503759398503 for k = 1 and ufmitest, and 3/2 of that for the
    # equal-FMI test of two terms (t = 2 <= 4).
    fit = sp.regress("y ~ x1 + x2 + x3", data=FLONG[FLONG["_mi_m"] == 1])
    est = {
        "params": fit.params.to_numpy(),
        "var_cov": np.asarray(fit.data_info["var_cov"]),
        "df_resid": fit.data_info["df_resid"],
    }
    pooled = _rubins_rules([est, dict(est)])
    pooled["var_names"] = list(fit.params.index)
    star = 396 * 397 / 399
    np.testing.assert_allclose(sp.mi_test(pooled, "x2")["df2"], star, rtol=1e-12)
    np.testing.assert_allclose(
        sp.mi_test(pooled, ["x2", "x3"], method="unrestricted")["df2"], star, rtol=1e-12
    )
    np.testing.assert_allclose(
        sp.mi_test(pooled, ["x2", "x3"])["df2"], 1.5 * star, rtol=1e-12
    )
    np.testing.assert_allclose(star, 394.01503759398503, rtol=1e-15)


def test_end_to_end_after_mi_estimate():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
    df["y"] = 1 + df["x1"] + rng.normal(size=200)
    df.loc[rng.choice(200, 30, replace=False), "x2"] = np.nan
    pooled = sp.mi_estimate(sp.mice(df, m=5, seed=0), sp.regress, formula="y ~ x1 + x2")
    res = sp.mi_test(pooled, "x2")
    assert res["df1"] == 1 and 0.0 <= res["pvalue"] <= 1.0
