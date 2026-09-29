"""``sp.hdfe_ols`` against Stata ``reghdfe``: nested FEs and collinear regressors.

Reference: Stata 18, ``reghdfe`` 6.12.3 on ``_fixtures/hdfe_nested_collinear.csv``
(100 units in 10 regions, 10 years), from
``_fixtures/_generate_hdfe_nested_collinear_Stata.do``. Two silent errors from
the Web of Power replication (QJE 2023):

* **Nested FEs.** ``year`` is nested in ``region x year``. StatsPAI charged
  ``sum(G_k) - (K - 1)`` absorbed parameters, so the redundant ``year``
  levels inflated the SEs (0.5% here, 0.8% in the paper). It now follows
  ``reghdfe``'s default ``dofadjustments(pairwise clusters continuous)``:
  FEs nested in a cluster drop out, and every later FE of a pair has at
  least as many redundant levels as the pair has mobility groups.
* **Collinear regressors.** A regressor spanned by the FEs made the solve
  raise (exact) or, with float noise, returned a coefficient of order 1e5
  and moved the others. It is now omitted as ``reghdfe`` omits it
  (``collinear_tol = min(1e-6, tol/10)``), reported as NaN, with a warning.

Estimates and SEs are held to 1e-10 relative (observed ~1e-15; the Stata
side imports the CSV as double -- ``import delimited`` defaults to float,
which alone moves the numbers at 1e-7); the absorbed degrees of freedom
exactly.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

SPECS = {
    "a": ("x + x2", "year + unit + regionXyear", "unit"),
    "b": ("x + x2", "unit + regionXyear", "unit"),
    "c": ("x + x2", "year + unit + regionXyear", None),
    "d": ("x + x2", "unit + regionXyear", None),
    "e": ("x + x2 + v_exact", "unit + regionXyear", "unit"),
    "f": ("x + x2 + v_near", "unit + regionXyear", "unit"),
    "g": ("x + x2 + v_near", "year + unit + regionXyear", None),
}


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "hdfe_nested_collinear_Stata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_FIX / "hdfe_nested_collinear.csv")


@pytest.mark.parametrize("key", sorted(SPECS))
def test_matches_reghdfe(ref, panel, key):
    rhs, fe, cl = SPECS[key]
    R = ref[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.hdfe_ols(f"y ~ {rhs} | {fe}", data=panel, cluster=cl)
    assert r.n_obs == R["N"]
    if cl is None:
        # reghdfe's e(df_a) is the unclustered count
        assert r.dof_fe == R["df_a"]
    for v in ("x", "x2"):
        assert r.coef[v] == pytest.approx(R[f"b_{v}"], rel=1e-10)
        assert r.se[v] == pytest.approx(R[f"se_{v}"], rel=1e-10)
    for v in ("v_exact", "v_near"):
        if v in rhs:
            assert np.isnan(r.coef[v]) and np.isnan(r.se[v])


def test_redundant_fe_does_not_move_clustered_se(panel):
    a = sp.hdfe_ols(
        "y ~ x + x2 | year + unit + regionXyear", data=panel, cluster="unit"
    )
    b = sp.hdfe_ols("y ~ x + x2 | unit + regionXyear", data=panel, cluster="unit")
    assert a.se["x"] == pytest.approx(b.se["x"], rel=1e-10)


def test_collinear_regressor_warns(panel):
    with pytest.warns(UserWarning, match="v_near"):
        sp.hdfe_ols("y ~ x + v_near | unit + regionXyear", data=panel, cluster="unit")


def test_all_regressors_collinear_raises(panel):
    with pytest.raises(sp.MethodIncompatibility, match="collinear"):
        sp.hdfe_ols("y ~ v_exact | unit + regionXyear", data=panel)
