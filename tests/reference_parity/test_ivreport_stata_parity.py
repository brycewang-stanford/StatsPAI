"""IV reporting tools vs Stata 18 (``regress, beta``, ``ivreg2`` / ``reghdfe``
on z-scored variables, ``acreg``, ``weakiv``).

Reference: ``_fixtures/ivreport_Stata.json``, written by
``_fixtures/_generate_ivreport_Stata.do`` (Stata 18 MP; ivreg2 4.1.12,
weakiv 2.4.07, acreg 1.1.0, reghdfe 6.13.1, ivreghdfe 1.1.4) on
``_fixtures/ivreport_fx.csv`` (``_generate_ivreport_data.py``: 600 grid
cells over 50-56 N, 6 W - 2 E, spatially correlated errors, an
endogenous regressor and two instruments; all values rounded to 6dp and
imported ``asdouble``, so both sides read the same doubles).

What is pinned (T2, same bytes):

* ``sp.standardize_coefs`` -- Stata's ``e(beta)`` (with and without
  ``[aw=]``), and every coefficient and SE of regress / ivreg2 / reghdfe /
  ivreghdfe run on pre-standardised variables, which is what AER papers
  report as "standardized coefficients" for IV and FE models. Observed
  1e-13 relative; asserted 1e-9 (SEs) and 1e-10 absolute (coefficients,
  all O(1), the centred intercept ~1e-17 on both sides).
* ``sp.conley_sensitivity`` / ``sp.conley`` (planar) -- acreg's SEs at
  25, 75, 150 and 300 km, OLS and 2SLS, uniform and Bartlett kernels;
  the uniform kernel's negative variances at 300 km are missing in acreg
  and NaN here. Observed <= 6e-13; asserted 1e-9.
* ``sp.hdfe_ols(conley_kernel='bartlett')`` with an absorbed effect
  (``acreg ..., pfe1() bartlett``), fixed in this change: it used to
  return the uniform-kernel SEs.
* The ``weakiv`` and ``acreg`` translations run through ``sp.stata``:
  AR / K / J / CLR statistics and p-values to 1e-9 relative, the CLR
  p-value of the one-regressor case to 1e-7 (weakiv integrates the
  conditional law with a 20-node Simpson rule, ~1e-7; we integrate exactly,
  see docs/dev/modules/iv.md). The confidence sets differ by design
  (weakiv interpolates on a grid; ours invert the test exactly) and are
  not compared here.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"
DATA = pd.read_csv(FX / "ivreport_fx.csv", float_precision="round_trip")
REF = json.loads((FX / "ivreport_Stata.json").read_text(encoding="utf-8"))
OLS = "y ~ x + w1 + w2 + d"
IV1 = "y ~ (x ~ z) + w1 + w2 + d"
IV2 = "y ~ (x ~ z + z2) + w1 + w2 + d"
NAMES = {"s_x": "x", "s_w1": "w1", "s_w2": "w2", "s_d": "d", "_cons": "Intercept"}


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _compare(key, params, ses, *, b_abs=1e-10, se_rel=1e-9):
    ref = REF[key]
    for i, name in enumerate(ref["names"]):
        ours = NAMES.get(name, name)
        if ours not in params.index:  # reghdfe's _cons with absorbed effects
            continue
        assert params[ours] == pytest.approx(ref["b"][i], abs=b_abs), (key, name)
        se_ref = ref["se"][i]
        if se_ref is None:  # acreg: negative variance shown as missing
            assert np.isnan(ses[ours]), (key, name)
        else:
            assert ses[ours] == pytest.approx(se_ref, rel=se_rel), (key, name)


# ---------------------------------------------------------------- beta


@pytest.mark.parametrize(
    "key, kw", [("regress_beta", {}), ("regress_aw_beta", {"weights": "wt"})]
)
def test_standardize_coefs_is_statas_e_beta(key, kw):
    fit = _quiet(sp.regress, OLS, data=DATA, **kw)
    beta = _quiet(sp.standardize_coefs, fit)
    ref = REF[key]
    for name, value in zip(ref["names"], ref["beta"]):
        assert beta.params[name] == pytest.approx(value, rel=1e-10)


@pytest.mark.parametrize(
    "key, fit",
    [
        ("std_regress_robust", lambda: sp.regress(OLS, data=DATA, vce="robust")),
        ("std_regress_cluster", lambda: sp.regress(OLS, data=DATA, cluster="region")),
        # ivreg2, robust (no small): HC0 and z
        (
            "std_ivreg2_robust",
            lambda: sp.iv(IV1, data=DATA, robust="hc0", small=False),
        ),
        ("std_ivreg2_cluster_small", lambda: sp.iv(IV2, data=DATA, cluster="region")),
    ],
)
def test_standardize_coefs_equals_fit_on_zscored_variables(key, fit):
    out = _quiet(sp.standardize_coefs, _quiet(fit))
    _compare(key, out.params, out.std_errors)


def test_standardize_coefs_after_reghdfe_uses_raw_standard_deviations():
    fit = _quiet(sp.hdfe_ols, OLS + " | fe", data=DATA, cluster="region")
    out = _quiet(sp.standardize_coefs, fit, data=DATA)
    _compare("std_reghdfe_cluster", out.params, out.std_errors)


def test_standardize_coefs_after_ivreghdfe():
    fit = _quiet(sp.iv, IV1, data=DATA, absorb="fe", robust="hc1")
    out = _quiet(sp.standardize_coefs, fit, data=DATA)
    _compare("std_ivreghdfe_robust_small", out.params, out.std_errors)


def test_stata_regress_beta_keeps_e_b_and_stores_e_beta():
    """As in Stata, the fit (e(b)) stays unstandardised; the Beta column is
    stored like e(beta), and later commands see the raw fit."""
    out = _quiet(sp.stata, "regress y x w1 w2 d, beta", data=DATA)
    raw = _quiet(sp.regress, OLS, data=DATA)
    assert out.params["x"] == raw.params["x"]
    ref = REF["regress_beta"]
    assert list(out.model_info["beta"]) == ref["names"]
    for name, value in zip(ref["names"], ref["beta"]):
        assert out.model_info["beta"][name] == pytest.approx(value, rel=1e-10)
    lc = _quiet(sp.stata, "regress y x w1 w2 d, beta\nlincom x", data=DATA)
    assert lc["estimate"] == pytest.approx(raw.params["x"], rel=1e-12)


# ---------------------------------------------------------------- acreg


@pytest.mark.parametrize("kernel, tag", [("uniform", "unif"), ("bartlett", "bart")])
@pytest.mark.parametrize("model", ["ols", "iv"])
def test_conley_sensitivity_matches_acreg_at_each_cutoff(kernel, tag, model):
    fit = (
        _quiet(sp.regress, OLS, data=DATA)
        if model == "ols"
        else _quiet(sp.iv, IV1, data=DATA)
    )
    cutoffs = [25, 75, 150, 300]
    tab = _quiet(
        sp.conley_sensitivity,
        fit,
        DATA,
        "lat",
        "lon",
        cutoffs,
        terms=list(fit.params.index),
        kernel=kernel,
    )
    for c in cutoffs:
        sub = tab.table[tab.table["cutoff_km"] == c].set_index("term")
        _compare(f"acreg_{model}_{tag}_{c}", sub["estimate"], sub["se"])


@pytest.mark.parametrize(
    "key, kernel",
    [("acreg_ols_pfe_unif_100", "uniform"), ("acreg_ols_pfe_bart_100", "bartlett")],
)
def test_hdfe_ols_conley_kernel_with_absorbed_effect(key, kernel):
    fit = _quiet(
        sp.hdfe_ols,
        OLS + " | fe",
        data=DATA,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=100,
        conley_kernel=kernel,
        drop_singletons=False,
    )
    _compare(key, fit.params, fit.std_errors)


def test_iv_with_absorbed_effect_recipe_in_the_acreg_refusal():
    """The call the acreg translation names for 2SLS + pfe1() is acreg's."""
    out = sp.from_stata(
        "acreg y w1 w2 d (x = z), spatial latitude(lat) longitude(lon) "
        "dist(100) pfe1(fe)"
    )
    assert not out["ok"]
    assert "sp.conley(sp.iv(" in out["error"] and "absorb=['fe']" in out["error"]
    fit = _quiet(sp.iv, IV1, data=DATA, absorb=["fe"])
    res = _quiet(
        sp.conley,
        fit,
        data=DATA,
        lat="lat",
        lon="lon",
        dist_cutoff=100.0,
        kernel="uniform",
        distance="planar",
    )
    _compare("acreg_iv_pfe_unif_100", res.params, res.std_errors)


@pytest.mark.parametrize(
    "line, key",
    [
        (
            "acreg y x w1 w2 d, spatial latitude(lat) longitude(lon) dist(75)",
            "acreg_ols_unif_75",
        ),
        (
            "acreg y w1 w2 d (x = z), spatial latitude(lat) longitude(lon) dist(150)",
            "acreg_iv_unif_150",
        ),
        (
            "acreg y x w1 w2 d, spatial latitude(lat) longitude(lon) dist(100) "
            "pfe1(fe) bartlett",
            "acreg_ols_pfe_bart_100",
        ),
    ],
)
def test_acreg_translation_runs_to_acregs_numbers(line, key):
    out = _quiet(sp.stata, line, data=DATA)
    _compare(key, out.params, out.std_errors)


# ---------------------------------------------------------------- weakiv

STATS = [
    ("ar_chi2", "ar_stat"),
    ("ar_p", "ar_pvalue"),
    ("k_chi2", "k_stat"),
    ("k_p", "k_pvalue"),
    ("j_chi2", "j_stat"),
    ("j_p", "j_pvalue"),
    ("clr_stat", "clr_stat"),
]


@pytest.mark.parametrize(
    "line, key",
    [
        ("weakiv ivreg2 y w1 w2 d (x = z z2), robust", "weakiv_robust"),
        (
            "weakiv ivreg2 y w1 w2 d (x = z z2), null(0.3) cluster(region)",
            "weakiv_cluster_null",
        ),
        ("weakiv ivreg2 y w1 w2 d (x = z z2), null(0.2)", "weakiv_iid_null"),
        (
            "ivreg2 y w1 w2 d (x = z z2), robust\nweakiv",
            "weakiv_after_ivreg2_robust",
        ),
        (
            "weakiv ivreg2 y w1 w2 d (x = z), robust level(90)",
            "weakiv_robust_justid_l90",
        ),
    ],
)
def test_weakiv_translation_reproduces_weakiv(line, key):
    out = _quiet(sp.stata, line, data=DATA)
    ref = REF[key]
    assert out["convention"] == "weakiv"
    for stata_key, ours in STATS:
        if stata_key in ref:
            assert out[ours] == pytest.approx(ref[stata_key], rel=1e-9), stata_key
    if "clr_p" in ref:  # just identified: weakiv reports no CLR (it is AR)
        # one regressor: weakiv's 20-node Simpson rule vs the exact integral
        assert out["clr_pvalue"] == pytest.approx(ref["clr_p"], rel=1e-7)
