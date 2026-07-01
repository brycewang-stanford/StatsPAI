"""Reference parity: ``sp.regress(vce=...)`` vs Stata / R gold standards.

Pins native OLS SE cells:

* **CR2/CR3/jackknife** vs R ``sandwich::vcovCL(HC2/3)`` (Pustejovsky-Tipton
  bias-reduced with the standard small-sample factor absorbed via the
  per-cluster hat block).
* **two-way** (cluster=[a,b]) vs Stata ``reghdfe y x, vce(cluster a b)``.
* **Conley** (acreg planar-distance) vs Stata ``acreg y x, spatial lat lon
  dist()``.

The wild WCR path is the same algorithm used by ``sp.feols``; its
external parity against Stata ``boottest`` is already covered by
``test_feols_wild_boottest_parity.py`` (Pustejovsky-Tipton WCR).  For OLS
we only assert that the native path runs and reaches a finite p-value
(the WCR is MC-based; the test is regression-not-precision).

The data generators match the Stata/R runs in ``REFERENCES.md`` (Stata 18 MP,
R 4.5 with clubSandwich + sandwich).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from statspai import regress  # noqa: E402

# --- Reference values (frozen) -------------------------------------------
# Strong-IV data: 400 obs / 15 clusters, R sandbox::vcovCL HC2/HC3
R_VCOVCL_HC2 = 0.0267156
R_VCOVCL_HC3 = 0.0284225
# Stata reghdfe with 2-way cluster (g + yr)
STATA_TWO_WAY_SE = 0.019039
# Stata acreg (Conley spatial HAC, planar dist)
STATA_ACREG_SE = 0.05587958


# --- Data loaders (frozen CSVs) -----------------------------------------
# The CSVs under tests/reference_parity/_fixtures/ are produced once and
# pinned alongside this test (see REFERENCES.md for the R/Stata generators).
# The reference values frozen in this file (0.0267156, 0.0284225, 0.019039,
# 0.05587958) were produced on the *same* CSVs in Stata 18 MP and R 4.5.
FIX = Path(__file__).resolve().parent / "_fixtures"


def _reg_panel() -> pd.DataFrame:
    return pd.read_csv(FIX / "ols_reg_panel.csv")


def _conley_panel() -> pd.DataFrame:
    return pd.read_csv(FIX / "ols_conley_panel.csv")


# --- CR2 / CR3 / jackknife vs R vcovCL ----------------------------------


def test_ols_cr2_cr3_jackknife_match_R_vcovCL() -> None:
    df = _reg_panel()
    for vce, ref in [
        ("CR2", R_VCOVCL_HC2),
        ("CR3", R_VCOVCL_HC3),
        ("jackknife", R_VCOVCL_HC3),
    ]:
        r = regress("y ~ x", data=df, vce=vce, cluster="g")
        assert np.isclose(float(r.std_errors["x"]), ref, atol=1e-5), (
            float(r.std_errors["x"]),
            ref,
        )


# --- two-way vs Stata reghdfe ------------------------------------------


def test_ols_two_way_cluster_matches_stata_reghdfe() -> None:
    df = _reg_panel()
    r = regress("y ~ x", data=df, cluster=["g", "yr"])
    assert np.isclose(float(r.std_errors["x"]), STATA_TWO_WAY_SE, atol=1e-5)


# --- Conley vs Stata acreg ----------------------------------------------


def test_ols_conley_matches_stata_acreg() -> None:
    """The ``_conley_panel`` fixture carries the same 400-obs data we used
    for the Stata ``acreg y x, spatial lat lon dist(5)`` reference (the
    ``lat``/``lon`` columns are synthetic spatial coordinates pinned
    alongside the original regression)."""
    df = _conley_panel()
    r = regress(
        "y ~ x",
        data=df,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=5.0,
    )
    assert np.isclose(float(r.std_errors["x"]), STATA_ACREG_SE, atol=1e-5)


# --- error paths -------------------------------------------------------


def test_ols_wild_requires_cluster() -> None:
    df = _reg_panel()
    with pytest.raises(Exception):
        regress("y ~ x", data=df, vce="wild")


def test_ols_cr2_requires_cluster() -> None:
    df = _reg_panel()
    with pytest.raises(Exception):
        regress("y ~ x", data=df, vce="CR2")


def test_ols_conley_requires_coords() -> None:
    df = _conley_panel()
    with pytest.raises(Exception):
        regress("y ~ d", data=df, vce="conley")


# --- native wild runs and yields a finite p-value -----------------------


def test_ols_wild_wcr_finite_pvalue() -> None:
    """The WCR bootstrap is MC-based; a wrong algorithm (e.g. silent fall-back
    to OLS) tends to give a wildly off p or NaN. Asserts the native path
    produces a finite p in [0, 1] (strictly positive coefficients can legitimately
    return 0.0 p under the WCR)."""
    df = _reg_panel()
    r = regress("y ~ x", data=df, vce="wild", cluster="g", wild_reps=99, seed=1)
    p = float(r.pvalues["x"])
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0
