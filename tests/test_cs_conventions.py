"""``callaway_santanna`` / ``aggte``: parity=, conventions record, pre-test
p-values, always-treated units, the ``n_boot`` alias.

The numbers against R ``did`` and Stata ``csdid`` are in
``tests/reference_parity/test_cs_parity_switch.py``; this file checks the
switch itself: what it sets, what it refuses, and identities between a
``parity=`` fit and the same conventions passed one by one.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import IdentificationFailure, MethodIncompatibility

_DIR = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"
MPDTA = pd.read_csv(_DIR / "cs_parity_mpdta.csv")
ALWAYS = pd.read_csv(_DIR / "cs_parity_always.csv")
KW = dict(y="lemp", g="first_treat", t="year", i="countyreal")


def _cs(data=MPDTA, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(data, **KW, **kw)


def _same_detail(a, b):
    pd.testing.assert_frame_equal(
        a.detail.reset_index(drop=True), b.detail.reset_index(drop=True)
    )


# ------------------------------------------------------------------ parity=


def test_parity_r_is_the_conventions_passed_one_by_one():
    by_hand = _cs(base_period="varying", balance="full")
    r = _cs(parity="r")
    _same_detail(r, by_hand)
    assert (r.estimate, r.se) == (by_hand.estimate, by_hand.se)
    pt = r.model_info["pretrend_test"]
    assert pt["pvalue"] == pt["pvalue_chi2"]
    assert by_hand.model_info["pretrend_test"]["pvalue"] == pt["pvalue_f"]


def test_parity_stata_is_the_conventions_passed_one_by_one():
    by_hand = _cs(
        base_period="varying", control_group="notyettreated", notyet_cutoff="cohort"
    )
    s = _cs(parity="stata", control_group="notyettreated")
    _same_detail(s, by_hand)
    g_hand = sp.aggte(by_hand, type="group", share_variance=False)
    g = sp.aggte(s, type="group")
    assert (g.estimate, g.se) == (g_hand.estimate, g_hand.se)
    d_hand = sp.aggte(by_hand, type="dynamic")
    d = sp.aggte(s, type="dynamic")
    np.testing.assert_array_equal(d.detail["se"], d_hand.detail["se"])


@pytest.mark.parametrize(
    "parity, kwargs",
    [
        ("stata", dict(base_period="universal")),
        ("r", dict(base_period="universal")),
        ("r", dict(notyet_cutoff="cohort", control_group="notyettreated")),
        ("stata", dict(notyet_cutoff="period", control_group="notyettreated")),
        ("r", dict(balance="pair")),
        ("r", dict(allow_unbalanced_panel=True)),
        ("stata", dict(balance="full")),
    ],
)
def test_parity_refuses_a_contradicting_convention(parity, kwargs):
    with pytest.raises(MethodIncompatibility, match="parity=") as exc:
        sp.callaway_santanna(MPDTA, **KW, parity=parity, **kwargs)
    assert "Drop" in exc.value.recovery_hint


def test_parity_accepts_the_same_value_and_rejects_unknown_programs():
    a = _cs(parity="stata", base_period="varying")
    b = _cs(parity="stata")
    _same_detail(a, b)
    assert a.model_info["conventions"]["base_period_explicit"] is True
    with pytest.raises(MethodIncompatibility, match="parity must be"):
        sp.callaway_santanna(MPDTA, **KW, parity="sas")


def test_parity_sets_bootstrap_defaults_but_leaves_them_options():
    s = _cs(parity="stata", bstrap=True, random_state=1)
    mi = s.model_info
    assert (mi["boot_weight_type"], mi["biters"], mi["cband"]) == ("mammen", 999, True)
    r = _cs(parity="r", bstrap=True, random_state=1)
    mi = r.model_info
    assert (mi["boot_weight_type"], mi["biters"], mi["cband"]) == (
        "rademacher",
        1000,
        True,
    )
    own = _cs(
        parity="stata",
        bstrap=True,
        biters=200,
        cband=False,
        boot_weight_type="rademacher",
        random_state=1,
    )
    mi = own.model_info
    assert (mi["boot_weight_type"], mi["biters"], mi["cband"]) == (
        "rademacher",
        200,
        False,
    )


def test_parity_stata_keeps_fit_aggregates_analytic_under_bootstrap():
    analytic = _cs(parity="stata")
    boot = _cs(parity="stata", bstrap=True, random_state=3)
    # csdid's estat: the simple ATT and event study stay analytic
    assert (boot.estimate, boot.se) == (analytic.estimate, analytic.se)
    pd.testing.assert_frame_equal(
        boot.model_info["event_study"], analytic.model_info["event_study"]
    )
    # while the ATT(g, t) SEs are the bootstrap ones
    assert not np.allclose(boot.detail["se"], analytic.detail["se"], rtol=1e-6)
    agg = sp.aggte(boot, type="calendar")
    ref = sp.aggte(analytic, type="calendar")
    assert (agg.estimate, agg.se) == (ref.estimate, ref.se)
    assert agg.model_info["bstrap"] is False


def test_parity_r_bootstrap_aggregates_follow_the_fit():
    boot = _cs(parity="r", bstrap=True, random_state=3)
    agg = sp.aggte(boot, type="group")
    assert agg.model_info["bstrap"] is True
    assert agg.model_info["conventions"]["se_method"] == "bootstrap"


def test_parity_r_balances_an_unbalanced_panel_like_r():
    holes = MPDTA.drop(MPDTA.sample(60, random_state=0).index)
    by_hand = _cs(holes, base_period="varying", balance="full")
    r = _cs(holes, parity="r")
    _same_detail(r, by_hand)
    assert r.model_info["balance"] == "full"
    assert r.model_info["balance_n_units_dropped"] > 0


def test_parity_stata_warns_on_csdid_ipw_name():
    with pytest.warns(UserWarning, match="ipw_abadie"):
        sp.callaway_santanna(MPDTA, **KW, parity="stata", estimator="ipw")


def test_parity_on_repeated_cross_sections():
    rcs = MPDTA.drop(columns="countyreal").assign(countyreal=np.arange(len(MPDTA)))
    by_hand = _cs(rcs, panel=False, base_period="varying")
    r = _cs(rcs, panel=False, parity="r")
    _same_detail(r, by_hand)
    conv = r.model_info["conventions"]
    assert conv["parity"] == "r" and conv["pretest_pvalue"] == "chi2"
    assert r.model_info["pretrend_test"]["pvalue_method"] == "chi2"


# ------------------------------------------------------------- aggte parity=


def test_aggte_refuses_what_csdid_estat_cannot_do():
    s = _cs(parity="stata")
    for kw in (
        dict(bstrap=True),
        dict(n_boot=100),
        dict(cband=True),
        dict(type="group", share_variance=True),
        dict(type="dynamic", share_variance=False),
    ):
        with pytest.raises(MethodIncompatibility, match="parity='stata'"):
            sp.aggte(s, **kw)
    # the value parity sets anyway is accepted
    a = sp.aggte(s, type="group", share_variance=False)
    b = sp.aggte(s, type="group")
    assert (a.estimate, a.se) == (b.estimate, b.se)


def test_aggte_parity_must_match_the_fit():
    with pytest.raises(MethodIncompatibility, match="Refit"):
        sp.aggte(_cs(), parity="r")
    with pytest.raises(MethodIncompatibility, match="parity='r' fixes"):
        sp.aggte(_cs(parity="r"), share_variance=False)


def test_aggte_records_conventions_without_parity():
    agg = sp.aggte(_cs(), type="group", share_variance=False)
    conv = agg.model_info["conventions"]
    assert conv == {
        "parity": None,
        "base_period": "universal",
        "base_period_explicit": False,
        "se_method": "analytic",
        "crit_dist": "normal",
        "crit_df": None,
        "share_variance": False,
        "pretest_pvalue": "hotelling_f",
        "agg_weights": "did",
    }


# ----------------------------------------------------------------- pre-test


def test_pretest_reports_both_p_values():
    fit = _cs(base_period="varying")
    pt = fit.model_info["pretrend_test"]
    W, r, G = pt["statistic"], pt["df"], fit.model_info["n_units"]
    # Stata csdid 1.81 estat pretrend on the same bytes: chi2(5) = 7.79...,
    # p = 0.1681224949238969 (tests/reference_parity/_fixtures/cs_parity_Stata.csv)
    assert W == pytest.approx(7.791236627200006, rel=1e-12)
    assert pt["pvalue_chi2"] == pytest.approx(0.1681224949238969, rel=1e-12)
    assert pt["pvalue_chi2"] == stats.chi2.sf(W, r)
    f_stat = W * (G - r) / (r * (G - 1))
    assert pt["pvalue_f"] == stats.f.sf(f_stat, r, G - r)
    assert pt["pvalue"] == pt["pvalue_f"] and pt["pvalue_method"] == "hotelling_f"
    assert pt["f_df"] == (r, G - r)
    assert fit.model_info["conventions"]["pretest_pvalue"] == "hotelling_f"
    text = fit.summary()
    assert "p (Hotelling F) = 0.1740; p (chi2, R did / csdid) = 0.1681" in text


def test_summary_notes_the_base_period_only_when_left_unset():
    note = "base_period='universal' (R did and Stata csdid default to 'varying'"
    assert note in _cs().summary()
    assert note not in _cs(base_period="universal").summary()
    assert note not in _cs(parity="r").summary()
    assert _cs().model_info["conventions"]["base_period_explicit"] is False
    assert (
        _cs(base_period="universal").model_info["conventions"]["base_period_explicit"]
        is True
    )


# --------------------------------------------------------- always treated


def test_always_treated_dropped_like_r_did():
    with pytest.warns(UserWarning, match=r"dropped 35 units .* g = 2001, 2003"):
        fit = sp.callaway_santanna(ALWAYS, **KW, base_period="varying")
    trimmed = _cs(ALWAYS[~ALWAYS.first_treat.isin([2001, 2003])], base_period="varying")
    mi = fit.model_info
    # R did: "Dropped 35 units that were already treated in the first
    # period", n = 465 (tests/reference_parity/_fixtures/cs_parity_R.json)
    assert mi["n_always_treated_dropped"] == 35
    assert mi["always_treated"]["cohorts"] == [2001.0, 2003.0]
    assert mi["always_treated"]["n_obs"] == 175
    assert mi["n_units"] == trimmed.model_info["n_units"] == 465
    assert fit.n_obs == trimmed.n_obs == 2325
    _same_detail(fit, trimmed)
    assert (fit.estimate, fit.se) == (trimmed.estimate, trimmed.se)
    assert mi["pretrend_test"] == trimmed.model_info["pretrend_test"]
    assert mi["n_treated_units"] == 191


def test_always_treated_with_anticipation_and_on_cross_sections():
    fit = _cs(ALWAYS, anticipation=1, base_period="varying")
    # R did: 55 units dropped "(accounting for anticipation = 1)", n = 445
    assert fit.model_info["n_always_treated_dropped"] == 55
    assert fit.model_info["n_units"] == 445
    rcs = _cs(ALWAYS, panel=False, base_period="varying")
    # R did, panel = FALSE: "Dropped 175 units", n = 2325
    assert rcs.model_info["n_always_treated_dropped"] == 175
    assert rcs.model_info["n_units"] == 2325


def test_always_treated_error_and_bad_value():
    with pytest.raises(IdentificationFailure, match="35 units") as exc:
        sp.callaway_santanna(ALWAYS, **KW, always_treated="error")
    assert "always_treated='drop'" in exc.value.recovery_hint
    with pytest.raises(MethodIncompatibility, match="always_treated"):
        sp.callaway_santanna(MPDTA, **KW, always_treated="keep")
    clean = _cs(always_treated="error")
    assert clean.model_info["n_always_treated_dropped"] == 0


# ------------------------------------------------------------- n_boot alias


def test_n_boot_is_an_alias_of_biters():
    a = _cs(bstrap=True, n_boot=50, random_state=11)
    b = _cs(bstrap=True, biters=50, random_state=11)
    _same_detail(a, b)
    assert a.model_info["biters"] == 50
    with pytest.raises(TypeError, match="n_boot"):
        sp.callaway_santanna(MPDTA, **KW, bstrap=True, n_boot=50, biters=50)
    # aggte already took both spellings
    g1 = sp.aggte(b, type="group", n_boot=40, random_state=2)
    g2 = sp.aggte(b, type="group", biters=40, random_state=2)
    assert g1.se == g2.se


def test_only_always_treated_cohorts_is_named_as_such():
    only = ALWAYS[ALWAYS.first_treat.isin([0, 2001, 2003])]
    with pytest.raises(sp.exceptions.DataInsufficient, match="No valid") as exc:
        sp.callaway_santanna(only, **KW)
    assert "g = 2001, 2003" in str(exc.value)
