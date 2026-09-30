"""Covariate-tightened Lee bounds = Stata ``leebounds, tight()``.

The UCT replication (QJE 2016) needed Lee's tightened bounds, which
``sp.lee_bounds`` ignored with a warning. Reference: Stata ``leebounds``
(Tauchmann, SSC v1.5) on ``_fixtures/leebounds_tight.csv``
(``_generate_leebounds_tight_Stata.do``), with one and two covariates
(3 and 6 cells). With ``trimming='leebounds'`` the bounds agree to 1e-14 and
the analytic variances -- ``leebounds``' approximation, reproduced
including its between-cell divisor -- to 1e-8.
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
STATA = json.loads((_FIX / "leebounds_tight_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "leebounds_tight.csv")


@pytest.mark.parametrize("cov", ["x", "x z"])
def test_matches_leebounds_tight(data, cov):
    ref = STATA[cov]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.lee_bounds(
            data,
            y="y",
            treat="d",
            selection="s",
            covariates=cov.split(),
            trimming="leebounds",
            se_method="analytic",
        )
    mi = r.model_info
    assert mi["lower_bound"] == pytest.approx(ref["lower"], rel=1e-14, abs=1e-15)
    assert mi["upper_bound"] == pytest.approx(ref["upper"], rel=1e-14, abs=1e-15)
    assert mi["se_lower"] ** 2 == pytest.approx(ref["var_lower"], rel=1e-8)
    assert mi["se_upper"] ** 2 == pytest.approx(ref["var_upper"], rel=1e-8)
    assert mi["n_cells"] == ref["cells"]
    assert mi["cell_selection_direction"] == ref["cellsel"]
    cells = mi["cells"]
    assert cells["weight"].sum() == pytest.approx(1.0)


def test_tightening_narrows_and_bootstrap_runs(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.lee_bounds(
            data, y="y", treat="d", selection="s", n_bootstrap=50
        ).model_info
        tight = sp.lee_bounds(
            data, y="y", treat="d", selection="s", covariates=["x", "z"], n_bootstrap=50
        ).model_info
    assert plain["lower_bound"] <= tight["lower_bound"] < tight["upper_bound"]
    assert tight["upper_bound"] <= plain["upper_bound"]
    assert np.isfinite(tight["se_lower"]) and tight["se_lower"] > 0


def test_analytic_needs_leebounds_trimming_and_known_columns(data):
    with pytest.raises(sp.MethodIncompatibility, match="leebounds"):
        sp.lee_bounds(
            data,
            y="y",
            treat="d",
            selection="s",
            covariates=["x"],
            se_method="analytic",
        )
    with pytest.raises(ValueError, match="not found"):
        sp.lee_bounds(data, y="y", treat="d", selection="s", covariates=["nope"])
