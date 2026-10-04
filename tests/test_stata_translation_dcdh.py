"""Translation of the de Chaisemartin-D'Haultfoeuille Stata commands.

``twowayfeweights``, ``did_multiplegt_dyn``, ``did_had`` and
``did_multiplegt_old`` are the commands of the authors' textbook
applications; ``regress, vce(hc2 clustvar, dfadjust)`` is the variance the
textbook uses for its cross-section regressions. The numbers behind the
translated calls are pinned in
``tests/reference_parity/test_dcdh_textbook_stata_parity.py``; this file
checks the mapping.
"""

from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "reference_parity" / "_fixtures"


def _ok(line, **kwargs):
    out = sp.from_stata(line, **kwargs)
    assert out["ok"], out
    assert out["untranslated_options"] == [], out
    return out


def test_twowayfeweights_fetr():
    out = _ok(
        "twowayfeweights div_rate state year udl, type(feTR) "
        "test_random_weights(exposurelength) weight(stpop)"
    )
    assert out["tool"] == "twowayfeweights"
    assert out["arguments"] == {
        "y": "div_rate",
        "group": "state",
        "time": "year",
        "treat": "udl",
        "test_random_weights": ["exposurelength"],
        "weights": "stpop",
    }


def test_twowayfeweights_fdtr_takes_the_level_last():
    cols = ["dy", "g", "t", "dd", "d", "s1", "s2", "s3"]
    out = _ok("twowayfeweights dy g t dd d, type(fdTR) controls(s1-s3)", columns=cols)
    args = out["arguments"]
    assert args["type"] == "fdTR"
    assert args["treat_level"] == "d"
    assert args["covariates"] == ["s1", "s2", "s3"]


def test_twowayfeweights_other_treatments():
    out = _ok("twowayfeweights y g t d, other_treatments(lag_d) type(feTR)")
    assert out["arguments"]["other_treatments"] == ["lag_d"]


@pytest.mark.parametrize(
    "line",
    [
        "twowayfeweights y g t d, type(feS)",
        "twowayfeweights y g t d",
        "twowayfeweights dy g t dd, type(fdTR)",
        "twowayfeweights y g t d d0, type(feTR)",
        "twowayfeweights y g t, type(feTR)",
    ],
)
def test_twowayfeweights_refusals(line):
    assert not sp.from_stata(line)["ok"]


def test_did_multiplegt_dyn_numbers_effects_from_one():
    out = _ok(
        "did_multiplegt_dyn y g t d, effects(13) placebo(13) weight(pop) graph_off"
    )
    args = out["arguments"]
    assert args["dynamic"] == 12 and args["placebo"] == 13
    assert args["se_method"] == "analytic"
    assert args["aggregation"] == "switchers"
    assert args["weights"] == "pop"
    assert out["ignored_display_options"] == ["graph_off"]


def test_did_multiplegt_dyn_default_is_one_effect():
    args = _ok("did_multiplegt_dyn y g t d")["arguments"]
    assert args["dynamic"] == 0 and args["placebo"] == 0


def test_did_multiplegt_dyn_options():
    args = _ok(
        "did_multiplegt_dyn y g t d, effects(4) placebo(4) normalized "
        "normalized_weights effects_equal(all) same_switchers switchers(in) "
        "trends_nonparam(z) controls(x1 x2) cluster(c) only_never_switchers "
        "design(0.8, console) by_path(3) ci_level(90)"
    )["arguments"]
    assert args["normalized"] is True
    assert args["normalized_weights"] is True
    assert args["effects_equal"] is True
    assert args["same_switchers"] is True
    assert args["switchers"] == "in"
    assert args["trends_nonparam"] == ["z"]
    assert args["controls"] == ["x1", "x2"]
    assert args["cluster"] == "c"
    assert args["control"] == "never_treated"
    assert args["design"] == 0.8
    assert args["by_path"] == 3
    assert args["alpha"] == pytest.approx(0.1)


def test_did_multiplegt_dyn_effects_equal_bounds_are_shifted():
    args = _ok('did_multiplegt_dyn y g t d, effects(5) effects_equal("2, 4")')[
        "arguments"
    ]
    assert args["effects_equal"] == (1, 3)


@pytest.mark.parametrize(
    "option", ["trends_lin", "predict_het(x, all)", "dont_drop_larger_lower"]
)
def test_did_multiplegt_dyn_reports_what_it_does_not_translate(option):
    out = sp.from_stata(f"did_multiplegt_dyn y g t d, effects(2) {option}")
    assert out["untranslated_options"] == [option.split("(")[0]]


def test_did_had():
    args = _ok("did_had lemp indusid year ntrgap, effects(4) placebo(3) graph_off")[
        "arguments"
    ]
    assert args == {
        "y": "lemp",
        "group": "indusid",
        "time": "year",
        "treat": "ntrgap",
        "effects": 4,
        "placebo": 3,
    }


def test_did_multiplegt_old_is_the_pairwise_estimator():
    out = _ok("did_multiplegt_old y g t d, breps(0) placebo(2)")
    assert out["tool"] == "did_multiplegt"
    assert out["arguments"]["placebo_sign"] == "r"
    assert out["arguments"]["n_boot"] == 0
    refused = sp.from_stata(
        "did_multiplegt_old y g t d, robust_dynamic dynamic(2) breps(0)"
    )
    assert not refused["ok"]
    assert "did_multiplegt_dyn" in refused["error"]


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "reg y x, vce(hc2 g, dfadjust)",
            {"formula": "y ~ x", "vce": "cr2", "cluster": "g", "dfadjust": True},
        ),
        (
            "reg y x, vce(hc2, dfadjust)",
            {"formula": "y ~ x", "robust": "hc2", "dfadjust": True},
        ),
        ("reg y x, vce(hc2 g)", {"formula": "y ~ x", "vce": "cr2", "cluster": "g"}),
        ("reg y x, vce(hc2)", {"formula": "y ~ x", "robust": "hc2"}),
    ],
)
def test_regress_hc2_cluster_and_dfadjust(line, expected):
    """``vce(hc2 g, dfadjust)`` used to come back as plain ``robust='hc2'``."""
    assert _ok(line)["arguments"] == expected


def test_regress_hc3_with_a_cluster_is_refused():
    assert not sp.from_stata("reg y x, vce(hc3 g)")["ok"]


def test_translated_calls_run():
    df = pd.read_csv(FIX / "dcdh_textbook_data.csv")
    w = sp.stata(
        "twowayfeweights y g year d, type(feTR) controls(x) weight(wt)", data=df
    )
    direct = sp.twowayfeweights(
        df, "y", "g", "year", "d", covariates=["x"], weights="wt"
    )
    assert w.estimate == pytest.approx(direct.estimate, rel=1e-12)
    r = sp.stata(
        "did_multiplegt_dyn y g year d, effects(2) placebo(1) graph_off", data=df
    )
    assert list(r.model_info["event_study"]["relative_time"]) == [-1, 0, 1]
    assert r.model_info["se_method"] == "analytic"
    h = sp.stata("regress y x d, vce(hc2 state, dfadjust)", data=df)
    assert h.model_info["dfadjust"] is True
