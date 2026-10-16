"""``sp.tobit`` and weighted ``sp.qreg`` cells the ldv_design fixture left empty.

Fixture: ``_fixtures/_generate_ldv_gaps_stata.do`` (Stata 18) on the same
``ldv_design_data.csv`` bytes as ``test_ldv_design_stata_parity.py``:

* tobit two-sided ``ll(0) ul(1.5)`` under ``vce(robust)``, ``vce(cluster g)``
  and ``[pw=w]``;
* tobit right-censored ``ul(1.5)`` under ``vce(oim)``, ``vce(robust)``,
  ``vce(cluster g)``;
* ``qreg yq x1 x2 [pw=w]`` at tau 0.25 / 0.5 / 0.75 (Stata's robust variance
  for sampling weights, which is ``sp.qreg``'s default with weights).
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
REF = json.loads((_FIX / "ldv_gaps_stata.json").read_text(encoding="utf-8"))
RTOL = 1e-6  # CLAUDE.md §5.1 default budget; observed <= 1.3e-8
NAMES = {"const": "_cons", "x1": "x1", "x2": "x2"}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "ldv_design_data.csv")


TOBIT = {
    "two_robust": dict(ll=0, ul=1.5, vce="robust"),
    "two_cluster": dict(ll=0, ul=1.5, cluster="g"),
    "two_pw": dict(ll=0, ul=1.5, weights="w"),
    "right_oim": dict(ll=None, ul=1.5),
    "right_robust": dict(ll=None, ul=1.5, vce="robust"),
    "right_cluster": dict(ll=None, ul=1.5, cluster="g"),
}


@pytest.mark.parametrize("case", sorted(TOBIT))
def test_tobit_matches_stata(data, case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.tobit(data, "y2", ["x1", "x2"], **TOBIT[case])
    ref = REF[case]
    for ours, theirs in NAMES.items():
        np.testing.assert_allclose(fit.params[ours], ref[f"b_{theirs}"], rtol=RTOL)
        np.testing.assert_allclose(fit.std_errors[ours], ref[f"se_{theirs}"], rtol=RTOL)


@pytest.mark.parametrize("q", [25, 50, 75])
def test_weighted_qreg_matches_stata_pweights(data, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.qreg(data, formula="yq ~ x1 + x2", quantile=q / 100, weights="w")
    assert fit.model_info["vce"] == "robust"
    ref = REF[f"qreg_pw_q{q}"]
    for ours, theirs in NAMES.items():
        np.testing.assert_allclose(fit.params[ours], ref[f"b_{theirs}"], rtol=1e-10)
        np.testing.assert_allclose(
            fit.std_errors[ours], ref[f"se_{theirs}"], rtol=1e-10
        )
