"""Translations of ``regress, beta``, ``weakiv`` and ``acreg``.

What each line maps to, and what is refused or reported as not carried
over. The numbers these calls produce are checked against Stata 18 in
tests/reference_parity/test_ivreport_stata_parity.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(3)
    n = 200
    d = pd.DataFrame(
        {
            "z": rng.normal(size=n),
            "z2": rng.normal(size=n),
            "w": rng.normal(size=n),
            "g": rng.integers(0, 10, n),
            "lat": rng.uniform(50, 53, n),
            "lon": rng.uniform(-3, 1, n),
        }
    )
    d["x"] = d.z + 0.5 * d.z2 + rng.normal(size=n)
    d["y"] = 1 + 0.4 * d.x + 0.3 * d.w + rng.normal(size=n)
    return d


# ---------------------------------------------------------------- beta


def test_regress_beta_is_translated_not_display_only():
    out = sp.from_stata("regress y x w, vce(robust) beta")
    assert out["ok"] and out["complete"]
    assert out["ignored_display_options"] == []
    assert out["postprocess"]["tool"] == "standardize_coefs"
    assert out["postprocess"]["attach"] == "beta"
    assert out["report_code"].startswith("sp.standardize_coefs(sp.regress(")
    assert out["arguments"] == {"formula": "y ~ x + w", "robust": "hc1"}


def test_sp_stata_beta_attaches_standardize_coefs_slopes(df):
    got = _quiet(sp.stata, "reg y x w, r beta", data=df)
    raw = _quiet(sp.regress, "y ~ x + w", data=df, robust="hc1")
    want = _quiet(sp.standardize_coefs, raw)
    pd.testing.assert_series_equal(got.params, raw.params)
    assert got.model_info["beta"] == {
        "x": float(want.params["x"]),
        "w": float(want.params["w"]),
    }


# ---------------------------------------------------------------- weakiv


def test_weakiv_wrapper_maps_model_variance_and_options():
    out = sp.from_stata(
        "weakiv ivreg2 y w (x = z z2), cluster(g) null(0.5) level(90) gridpoints(50)"
    )
    assert out["ok"] and out["complete"], out
    assert out["tool"] == "weakrobust"
    assert out["arguments"] == {
        "y": "y",
        "endog": "x",
        "instruments": ["z", "z2"],
        "exog": ["w"],
        "cluster": "g",
        "h0": 0.5,
        "alpha": 0.1,
    }


def test_weakiv_iid_writes_test_vcov_out():
    """Without test_vcov sp.weakrobust keeps R ivmodel's F-form AR; the
    translation must ask for weakiv's LM form explicitly."""
    out = sp.from_stata("weakiv ivregress 2sls y w (x = z z2)")
    assert out["arguments"]["test_vcov"] == "classic"
    out = sp.from_stata("weakiv ivreg2 y w (x = z), robust")
    assert out["arguments"]["test_vcov"] == "robust" and out["complete"]


def test_weakiv_two_endogenous_null_is_a_list():
    out = sp.from_stata("weakiv ivreg2 y w (x w2 = z z2 z3), null(0.1 -0.2) subset(x)")
    assert out["arguments"]["endog"] == ["x", "w2"]
    assert out["arguments"]["h0"] == [0.1, -0.2]
    assert out["arguments"]["subset"] == "x"


@pytest.mark.parametrize(
    "line, lost",
    [
        ("weakiv ivreg2 y w (x = z z2), robust small", ["small"]),
        ("weakiv ivreg2 y w (x = z z2), md", ["md"]),
        ("weakiv ivreg2 y w (x = z z2), kwt(0.5)", ["kwt"]),
        ("weakiv ivreg2 y w (x = z z2), level(90 95)", ["level"]),
        ("weakiv, null(0.3)", ["model"]),
    ],
)
def test_weakiv_options_that_change_statistics_are_reported(line, lost):
    out = sp.from_stata(line)
    assert out["ok"]
    assert out["complete"] is False
    assert set(lost) <= set(out["dropped"])


@pytest.mark.parametrize(
    "line",
    [
        "weakiv ivregress liml y w (x = z z2)",
        "weakiv ivprobit y w (x = z z2)",
        "weakiv ivreg2 y w (x = z z2) [aw=g]",
        "weakiv ivreg2 y w (x = z z2), cluster(g lat)",
    ],
)
def test_weakiv_unsupported_forms_are_refused(line):
    assert not sp.from_stata(line)["ok"]


def test_bare_weakiv_runs_on_the_previous_iv_fit(df):
    chained = _quiet(
        sp.stata, "ivreg2 y w (x = z z2), robust\nweakiv, null(0.4)", data=df
    )
    direct = _quiet(
        sp.weakrobust,
        df,
        y="y",
        endog="x",
        instruments=["z", "z2"],
        exog=["w"],
        test_vcov="robust",
        h0=0.4,
    )
    for key in ("ar_stat", "k_stat", "j_stat", "clr_stat", "clr_pvalue"):
        assert chained[key] == direct[key]


def test_bare_weakiv_after_ols_is_refused(df):
    with pytest.raises(MethodIncompatibility, match="ivreg2"):
        _quiet(sp.stata, "regress y x w\nweakiv", data=df)


def test_bare_weakiv_after_small_ivreg2_is_refused(df):
    with pytest.raises(MethodIncompatibility, match="small"):
        _quiet(sp.stata, "ivreg2 y w (x = z z2), robust small\nweakiv", data=df)


# ---------------------------------------------------------------- acreg

_SP = "spatial latitude(lat) longitude(lon)"


def test_acreg_ols_and_iv_uniform_map_to_single_calls():
    out = sp.from_stata(f"acreg y x w, {_SP} dist(80)")
    assert out["tool"] == "regress" and out["complete"]
    assert out["arguments"] == {
        "formula": "y ~ x + w",
        "vce": "conley",
        "conley_lat": "lat",
        "conley_lon": "lon",
        "conley_cutoff": 80.0,
    }
    out = sp.from_stata(f"acreg y w (x = z), {_SP} distcutoff(80)")
    assert out["tool"] == "iv" and out["arguments"]["formula"] == "y ~ w + (x ~ z)"


def test_acreg_panel_with_absorbed_effects_maps_to_hdfe_ols():
    out = sp.from_stata(
        f"acreg y x w, {_SP} dist(80) id(g) time(t) lag(2) lagdist(1) hac "
        "pfe1(g) pfe2(t) bartlett dropsingletons"
    )
    assert out["ok"] and out["complete"], out
    assert out["tool"] == "hdfe_ols"
    a = out["arguments"]
    assert a["formula"] == "y ~ x + w | g + t"
    assert (a["conley_lag"], a["conley_lag_cross"]) == (2, 1)
    assert (a["conley_kernel"], a["conley_time_kernel"]) == ("bartlett", "bartlett")
    assert a["drop_singletons"] is True


@pytest.mark.parametrize(
    "line, match",
    [
        (f"acreg y w (x = z), {_SP} dist(80) bartlett", "sp.conley\\(sp.iv"),
        (f"acreg y x w, {_SP} dist(80) bartlett", "sp.conley\\(sp.regress"),
        (f"acreg y x w, {_SP} dist(80) id(g) time(t) lag(2)", "lag_cutoff=2"),
        ("acreg y x w, network links_mat(a b)", "network"),
        ("acreg y x w, cluster(g)", "cluster"),
        (f"acreg y x w, {_SP}", "dist"),
        (f"acreg y x w, {_SP} dist(80) lagdist(1)", "lagdist"),
    ],
)
def test_acreg_forms_without_a_single_call_are_refused_with_the_recipe(line, match):
    out = sp.from_stata(line)
    assert not out["ok"]
    assert pytest.importorskip("re").search(match, out["error"]), out["error"]


@pytest.mark.parametrize("opt", ["small", "ak0"])
def test_acreg_small_sample_options_are_reported(opt):
    out = sp.from_stata(f"acreg y x w, {_SP} dist(80) {opt}")
    assert out["ok"] and opt in out["dropped"]


def test_acreg_display_options_do_not_count_as_lost():
    out = sp.from_stata(f"acreg y x w, {_SP} dist(80) correctr2 storeweights")
    assert out["complete"]
    assert sorted(out["ignored_display_options"]) == ["correctr2", "storeweights"]


def test_acreg_translation_runs(df):
    got = _quiet(sp.stata, f"acreg y x w, {_SP} dist(80)", data=df)
    want = _quiet(
        sp.regress,
        "y ~ x + w",
        data=df,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=80.0,
    )
    pd.testing.assert_series_equal(got.std_errors, want.std_errors)
