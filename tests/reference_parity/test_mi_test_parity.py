"""sp.mi_test against Stata 18 ``mi test, nosmall`` (T2).

Fixture: ``_fixtures/mi_test_flong.csv`` holds the original data and eight
Stata ``mi impute chained`` completions (flong, ``%21.0g``); the reference
numbers come from ``_fixtures/_generate_mi_test_stata.do`` run on that same
file. Pooling eight / three completions and testing two and three terms
covers both df branches of the equal-FMI test (``t = k(m-1) > 4`` and
``t <= 4``) and the unrestricted-FMI test (``ufmitest``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.imputation.mice import _rubins_rules

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "mi_test_stata.json").read_text(encoding="utf-8"))
FLONG = pd.read_csv(FIX / "mi_test_flong.csv")


def _pooled(m: int):
    ests, names = [], None
    for j in range(1, m + 1):
        fit = sp.regress("y ~ x1 + x2 + x3", data=FLONG[FLONG["_mi_m"] == j])
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
    res = sp.mi_test(_pooled(m), terms, method=method)
    assert res["df1"] == ref["df1"]
    np.testing.assert_allclose(res["F"], ref["F"], rtol=1e-9)
    np.testing.assert_allclose(res["df2"], ref["df2"], rtol=1e-9)
    np.testing.assert_allclose(res["pvalue"], ref["p"], rtol=1e-9)


def test_small_sample_df_is_refused_not_approximated():
    with pytest.raises(MethodIncompatibility, match="Reiter"):
        sp.mi_test(_pooled(3), ["x2"], small=True)


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
    res = sp.mi_test(pooled, ["x2", "x3"])
    idx = [pooled["var_names"].index(t) for t in ("x2", "x3")]
    q = est["params"][idx]
    V = est["var_cov"][np.ix_(idx, idx)]
    assert res["df2"] == np.inf
    np.testing.assert_allclose(res["F"], q @ np.linalg.solve(V, q) / 2, rtol=1e-12)


def test_end_to_end_after_mi_estimate():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
    df["y"] = 1 + df["x1"] + rng.normal(size=200)
    df.loc[rng.choice(200, 30, replace=False), "x2"] = np.nan
    pooled = sp.mi_estimate(sp.mice(df, m=5, seed=0), sp.regress, formula="y ~ x1 + x2")
    res = sp.mi_test(pooled, "x2")
    assert res["df1"] == 1 and 0.0 <= res["pvalue"] <= 1.0
