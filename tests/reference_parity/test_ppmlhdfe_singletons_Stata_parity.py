"""``sp.ppmlhdfe`` Stata defaults: singletons, absorbed regressors, ``a#b``.

A replication of a PPML paper (five crossed fixed effects) found three
silent differences from Stata ``ppmlhdfe``: singletons were kept (``N``,
the cluster count and the pseudo R-squared differed); a regressor absorbed
by the fixed effects crashed the solver with ``LinAlgError`` where Stata
reports it as omitted; and interacted effects had to be built by hand.

Reference: Stata 18 ``ppmlhdfe`` on ``_fixtures/ppmlhdfe_singletons.csv``
(unbalanced panel with one-period units, all-zero units and five missing
regressor values), from ``_generate_ppmlhdfe_singletons_Stata.do``:

* A: ``ppmlhdfe y x1 d ever, absorb(id year) vce(cluster id)`` -- ``ever``
  is time-invariant and omitted.
* B: ``ppmlhdfe y x1 d, absorb(id ind#year) vce(cluster cy)`` with
  ``cy = group(city year)`` -- ``d`` is a function of ``ind#year`` and
  omitted.

``N``, the cluster count and the dropped count agree exactly.  Slopes, SEs
and the pseudo R-squared agree to ~2e-9, inside Stata's own IRLS
convergence tolerance (1e-8), hence ``rtol=1e-7``.
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
REF = json.loads((_FIX / "ppmlhdfe_singletons_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "ppmlhdfe_singletons.csv")


def _fit(data, spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec == "A":
            return sp.ppmlhdfe(
                "y ~ x1 + d + ever", data=data, absorb="id + year", cluster="id"
            )
        return sp.ppmlhdfe(
            "y ~ x1 + d", data=data, absorb="id + ind#year", cluster="city^year"
        )


@pytest.mark.parametrize("spec", ["A", "B"])
def test_sample_matches_stata(data, spec):
    r = _fit(data, spec)
    ref = REF[spec]
    assert r.data_info["nobs"] == ref["N"]
    assert r.data_info["n_clusters"] == ref["N_clust"]
    # Stata's e(num_singletons) counts singletons and FE-separated rows together.
    mi = r.model_info
    assert mi["n_singletons"] + mi["n_separated"] == ref["num_singletons"]


@pytest.mark.parametrize("spec", ["A", "B"])
def test_estimates_match_stata(data, spec):
    r = _fit(data, spec)
    ref = REF[spec]
    np.testing.assert_allclose(r.params["x1"], ref["b_x1"], rtol=1e-7)
    np.testing.assert_allclose(r.std_errors["x1"], ref["se_x1"], rtol=1e-7)
    np.testing.assert_allclose(r.model_info["pseudo_r2"], ref["r2_p"], rtol=1e-7)
    if spec == "A":
        np.testing.assert_allclose(r.params["d"], ref["b_d"], rtol=1e-7)
        np.testing.assert_allclose(r.std_errors["d"], ref["se_d"], rtol=1e-7)


def test_absorbed_regressors_are_omitted_with_a_warning(data):
    with pytest.warns(UserWarning, match="omitted regressor"):
        r = sp.ppmlhdfe(
            "y ~ x1 + d + ever", data=data, absorb="id + year", cluster="id"
        )
    assert r.model_info["omitted"] == ["ever"]
    assert list(r.params.index) == ["x1", "d"]
    assert _fit(data, "B").model_info["omitted"] == ["d"]


def test_interaction_terms_equal_hand_built_groups(data):
    d = data.copy()
    d["iy"] = d.groupby(["ind", "year"]).ngroup()
    d["cy"] = d.groupby(["city", "year"]).ngroup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hand = sp.ppmlhdfe("y ~ x1", data=d, absorb="id + iy", cluster="cy")
        caret = sp.ppmlhdfe("y ~ x1 | id + ind^year", data=d, cluster="city#year")
    np.testing.assert_allclose(caret.params, hand.params, rtol=1e-10)
    np.testing.assert_allclose(caret.std_errors, hand.std_errors, rtol=1e-10)


def test_drop_singletons_false_keeps_them_and_leaves_slopes(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = _fit(data, "A")
        off = sp.ppmlhdfe(
            "y ~ x1 + d + ever",
            data=data,
            absorb="id + year",
            cluster="id",
            drop_singletons=False,
        )
    assert off.model_info["n_singletons"] == 0
    # Zero-outcome singletons are all-zero groups, so separation still
    # removes those; only the positive-outcome singletons come back.
    assert on.data_info["nobs"] < off.data_info["nobs"]
    assert off.data_info["nobs"] <= on.data_info["nobs"] + on.model_info["n_singletons"]
    # A singleton is fitted exactly: slopes are unchanged.
    np.testing.assert_allclose(off.params, on.params, rtol=1e-7)
