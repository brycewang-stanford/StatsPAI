"""Configuration-level evidence for dynamic-panel GMM.

Pins what ``sp.validation_scope`` reports for fits of ``sp.xtabond`` and
``sp.xtdpdsys`` (``statspai/_validation_scope_dynpanel.py``): which
configurations a reference ran (Stata xtabond / xtabond2 / xtdpdsys / xtdpd,
R plm::pgmm / pdynmc), how the options the fit does not store in
``model_info`` are read from the recorded call, and that a configuration no
reference ran is not reported as covered.
"""

import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


@pytest.fixture(scope="module")
def ab():
    return pd.read_csv(_FIX / "dynpanel_abdata.csv")


@pytest.fixture(scope="module")
def gapped(ab):
    # the I* specs of the abdata parity file: 1980 dropped for every third firm
    return ab[~((ab["id"] % 3 == 0) & (ab["year"] == 1980))].reset_index(drop=True)


@pytest.fixture(scope="module")
def early(ab):
    # the singular two-step weight of the Hansen-methods file
    return ab[~((ab["year"] < 1978) & (ab["id"] > 6))].reset_index(drop=True)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _xtabond(d, **k):
    return _quiet(sp.xtabond, d, y="n", id="id", time="year", **k)


def _xtdpdsys(d, **k):
    return _quiet(sp.xtdpdsys, d, y="n", id="id", time="year", **k)


def _scope(result):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.validation_scope(result)


AB1991 = ["l(0/1).w", "l(0/2).k"]
WK = ["w", "k"]

CASES = [
    # --- difference GMM: Stata xtabond / xtabond2 on abdata
    ("default", lambda ab, g, e: _xtabond(ab, lags=1), "covered"),
    (
        "two_step_conventional",
        lambda ab, g, e: _xtabond(ab, lags=1, twostep=True, robust=False),
        "covered",
    ),
    ("ar2", lambda ab, g, e: _xtabond(ab, lags=2), "covered"),
    ("maxldep", lambda ab, g, e: _xtabond(ab, lags=1, gmm_lags=(2, 4)), "covered"),
    ("collapse", lambda ab, g, e: _xtabond(ab, lags=1, collapse=True), "covered"),
    (
        "ab1991_two_step",
        lambda ab, g, e: _xtabond(ab, x=AB1991, lags=2, twostep=True),
        "covered",
    ),
    (
        "predetermined_custom_window",
        lambda ab, g, e: _xtabond(
            ab,
            x=["l(0/2).k"],
            predetermined=["l(0/1).w"],
            predetermined_lags=(2, None),
            lags=2,
        ),
        "covered",
    ),
    (
        "fod_two_step",
        lambda ab, g, e: _xtabond(ab, x=WK, lags=1, orthogonal=True, twostep=True),
        "covered",
    ),
    (
        "cluster_industry",
        lambda ab, g, e: _xtabond(ab, x=WK, lags=1, cluster="ind"),
        "covered",
    ),
    (
        "anderson_hsiao_differences",
        lambda ab, g, e: _xtabond(
            ab, x=WK, lags=1, method="ah", ah_instrument="differences"
        ),
        "covered",
    ),
    ("gapped_two_step", lambda ab, g, e: _xtabond(g, lags=1, twostep=True), "covered"),
    (
        "gapped_collapsed_depth_4",
        lambda ab, g, e: _xtabond(g, lags=1, gmm_lags=(4, 4), collapse=True),
        "covered",
    ),
    # --- system GMM
    (
        "system_xtabond2",
        lambda ab, g, e: _xtabond(ab, x=WK, lags=1, method="system"),
        "covered",
    ),
    (
        "system_fod_two_step",
        lambda ab, g, e: _xtabond(
            ab, x=WK, lags=1, method="system", orthogonal=True, twostep=True
        ),
        "covered",
    ),
    ("xtdpdsys_ar1", lambda ab, g, e: _xtdpdsys(ab, lags=1), "covered"),
    (
        "xtdpdsys_classical",
        lambda ab, g, e: _xtdpdsys(ab, x=WK, lags=1, robust=False),
        "covered",
    ),
    (
        "xtdpd_period_dummies_robust",
        lambda ab, g, e: _xtdpdsys(
            ab, lags=2, time_dummies=True, h=2, iv_equation="both"
        ),
        "covered",
    ),
    (
        "xtdpd_singular_weight_windmeijer",
        lambda ab, g, e: _xtdpdsys(
            e, lags=2, h=2, iv_equation="both", twostep=True, robust=True
        ),
        "covered",
    ),
    # --- classical SEs where only robust ones were compared: the estimate
    # does not depend on robust= (the map's invariant), the SE has no row
    (
        "maxldep_classical",
        lambda ab, g, e: _xtabond(ab, lags=1, gmm_lags=(2, 4), robust=False),
        "estimate_only",
    ),
    (
        "system_xtabond2_classical",
        lambda ab, g, e: _xtabond(ab, x=WK, lags=1, method="system", robust=False),
        "estimate_only",
    ),
    (
        "gapped_classical",
        lambda ab, g, e: _xtabond(g, lags=1, robust=False),
        "estimate_only",
    ),
    # --- legal options no reference ran
    (
        "difference_with_period_dummies",
        lambda ab, g, e: _xtabond(ab, lags=2, time_dummies=True),
        "not_covered",
    ),
    (
        "iterated",
        lambda ab, g, e: _xtabond(ab, lags=1, steps="iterated"),
        "not_covered",
    ),
    ("cue", lambda ab, g, e: _xtabond(ab, lags=1, steps="cue"), "not_covered"),
    (
        "difference_x_with_period_dummies",
        lambda ab, g, e: _xtabond(ab, x=WK, lags=1, time_dummies=True),
        "not_covered",
    ),
    (
        "system_h3_x_in_differences_only",
        lambda ab, g, e: _xtabond(
            ab, x=WK, lags=1, method="system", iv_equation="diff"
        ),
        "not_covered",
    ),
    ("identity_weight", lambda ab, g, e: _xtabond(ab, x=WK, h=1), "not_covered"),
    ("three_lags", lambda ab, g, e: _xtabond(ab, lags=3), "not_covered"),
    (
        "xtdpd_period_dummies_one_lag",
        lambda ab, g, e: _xtdpdsys(
            ab, lags=1, time_dummies=True, h=2, iv_equation="both"
        ),
        "not_covered",
    ),
    (
        "system_without_constant",
        lambda ab, g, e: _xtabond(ab, lags=1, method="system", constant=False),
        "not_covered",
    ),
    (
        "predetermined_default_window",
        lambda ab, g, e: _xtabond(
            ab, x=["l(0/2).k"], predetermined=["l(0/1).w"], lags=2
        ),
        "not_covered",
    ),
    (
        "endogenous_default_window",
        lambda ab, g, e: _xtabond(ab, x=["l(0/2).k"], endogenous=["l(0/1).w"], lags=2),
        "not_covered",
    ),
    (
        "two_step_coarser_cluster",
        lambda ab, g, e: _xtabond(
            ab, lags=1, cluster="ind", twostep=True, collapse=True, gmm_lags=(2, 4)
        ),
        "not_covered",
    ),
    (
        "xtdpdsys_level_only_iv",
        lambda ab, g, e: _xtdpdsys(ab, x=WK, lags=1, iv_equation="level"),
        "not_covered",
    ),
]


@pytest.mark.parametrize("name,fit,expected", CASES, ids=[c[0] for c in CASES])
def test_status_of_representative_configurations(
    ab, gapped, early, name, fit, expected
):
    scope = _scope(fit(ab, gapped, early))
    assert scope["status"] == expected, (name, scope["configuration"])
    assert not scope["unchecked"], scope["unchecked"]
    if expected == "not_covered":
        for out in ("estimate", "se"):
            assert not scope["outputs"][out]["evidence"]
    if expected == "estimate_only":
        assert scope["configuration"]["robust"] == "false"
        assert scope["outputs"]["estimate"]["status"] == "reference"
        assert scope["outputs"]["se"]["status"] == "not_covered"


def test_the_options_model_info_does_not_store_are_read_from_the_call(ab):
    """Instrument classes and their windows, h, iv_equation."""
    custom = _scope(
        _xtabond(
            ab,
            x=["l(0/2).k"],
            endogenous=["l(0/1).w"],
            endogenous_lags=(3, None),
            lags=2,
        )
    )["configuration"]
    assert custom["regressors"] == "x_endo_window"
    assert custom["model"] == "difference"
    # the class default (2, None) spelled out is the default window
    spelled = _scope(
        _xtabond(
            ab,
            x=["l(0/2).k"],
            endogenous=["l(0/1).w"],
            endogenous_lags=(2, None),
            lags=2,
        )
    )
    assert spelled["configuration"]["regressors"] == "other"
    assert spelled["status"] == "not_covered"


def test_each_name_reads_its_own_defaults(ab):
    """sp.xtdpdsys defaults to h=2 and x in the differenced equation only;
    sp.xtabond(method='system') to xtabond2's h=3 and iv() in both."""
    by_xtabond = _scope(_xtabond(ab, x=WK, lags=1, method="system"))
    by_xtdpdsys = _scope(_xtdpdsys(ab, x=WK, lags=1))
    assert by_xtabond["configuration"]["model"] == "system_h3_iv_both"
    assert by_xtdpdsys["configuration"]["model"] == "system_h2_iv_diff"
    # without x or period dummies iv_equation places nothing
    assert _scope(_xtdpdsys(ab, lags=1))["configuration"]["model"] == (
        "system_h2_no_iv"
    )


def test_xtdpdsys_with_xtabond2_defaults_borrows_the_system_rows(ab):
    """sp.xtdpdsys(iv_equation='both', h=3) is sp.xtabond(method='system')."""
    scope = _scope(_xtdpdsys(ab, x=WK, lags=1, iv_equation="both", h=3))
    assert scope["function"] == "xtdpdsys"
    assert scope["status"] == "covered"
    entries = {e["entry_point"] for e in scope["outputs"]["se"]["evidence"]}
    assert entries == {"sp.xtabond(method='system')"}
    kinds = {e["kind"] for e in scope["outputs"]["estimate"]["evidence"]}
    assert {"T1", "T2"} <= kinds


def test_equivalent_spellings_read_as_the_configuration_they_run(ab):
    # steps=2 is twostep=True
    a = _scope(_xtabond(ab, lags=1, steps=2))["configuration"]
    b = _scope(_xtabond(ab, lags=1, twostep=True))["configuration"]
    assert a == b and a["steps"] == "twostep"
    # cluster on the panel id is the default unit grouping
    assert (
        _scope(_xtabond(ab, x=WK, lags=1, cluster="id"))["configuration"]["cluster"]
        == "unit"
    )
    # Anderson-Hsiao uses the minimum lag only
    ah = _scope(_xtabond(ab, lags=1, method="ah", gmm_lags=(2, 4)))
    assert ah["configuration"]["gmm_window"] == "default"
    assert ah["status"] == "covered"
    # h(2) and h(3) are one matrix without a level equation
    assert _scope(_xtabond(ab, lags=1, h=2))["configuration"]["model"] == ("difference")


def test_three_lags_of_y_have_no_reference(ab):
    """The references ran one and two lags of y; three is its own cell."""
    one = _scope(_xtabond(ab, lags=1))
    two = _scope(_xtabond(ab, lags=2))
    three = _scope(_xtabond(ab, lags=3))
    assert (one["configuration"]["lags"], two["configuration"]["lags"]) == ("1", "2")
    assert three["configuration"]["lags"] == "3plus"
    assert one["status"] == two["status"] == "covered"
    assert three["status"] == "not_covered"
    # everything but the lags matches: the default-call rows are near misses
    assert {m["differs_in"] for m in three["near_misses"]} == {"lags"}


def test_rows_are_specific_to_the_lags_they_ran(ab):
    """Arellano-Bond (1991) table 4 ran two lags of y only; Track A module 50
    (one lag) is the reference for one lag with x."""
    two = _scope(_xtabond(ab, x=["l(0/1).w", "l(0/2).k"], lags=2))
    one = _scope(_xtabond(ab, x=["l(0/1).w", "l(0/2).k"], lags=1))
    two_rows = {e["compares"] for e in two["outputs"]["se"]["evidence"]}
    one_rows = {e["compares"] for e in one["outputs"]["se"]["evidence"]}
    assert any("table 4" in c for c in two_rows)
    assert not any("table 4" in c for c in one_rows)
    assert any("Track A" in c for c in one_rows)
    assert not any("Track A" in c for c in two_rows)


def test_robust_se_row_does_not_vouch_for_the_classical_one(ab):
    """xtdpd with period dummies: the robust and the classical fits match
    Stata through different rows, and the estimate does not depend on robust=."""
    rob = _scope(_xtdpdsys(ab, lags=2, time_dummies=True, h=2, iv_equation="both"))
    se_rows = [e["compares"] for e in rob["outputs"]["se"]["evidence"]]
    assert se_rows and all("vce(robust)" in c for c in se_rows)
    assert "robust" in rob["invariant"]
    assert rob["outputs"]["diagnostic"]["status"] == "not_covered"


def test_sargan_scale_is_a_disclosure_next_to_the_reference_diagnostics(ab):
    scope = _scope(_xtabond(ab, x=WK, lags=1))
    kinds = {e["kind"] for e in scope["outputs"]["diagnostic"]["evidence"]}
    assert kinds == {"T2", "T4"}
    assert scope["outputs"]["diagnostic"]["status"] == "reference"


def test_xtdpd_translated_by_sp_stata_is_covered(ab):
    fit = _quiet(
        sp.stata,
        """
        xtset id year
        xi: xtdpd L(0/2).n L(0/1).w k i.year, iv(k i.year) ///
            dgmmiv(n w, lag(2 4)) lgmmiv(n w) twostep vce(robust)
        """,
        data=ab,
    )
    scope = _scope(fit)
    assert scope["function"] == "xtdpdsys"
    assert scope["configuration"]["regressors"] == "x_endo_window_dummies"
    assert scope["configuration"]["gmm_window"] == "max_capped"
    assert scope["status"] == "covered"
    entries = {e["entry_point"] for e in scope["outputs"]["se"]["evidence"]}
    assert entries == {"sp.stata(...)"}


def test_a_fit_without_a_call_record_leaves_those_dimensions_unchecked(ab):
    """Nothing is guessed: the options read from the call stay unchecked."""
    r = _xtabond(ab, lags=1)
    object.__delattr__(r, "_statspai_call")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scope = sp.validation_scope(r, function="xtabond")
    assert scope["configuration"]["transform"] == "fd"
    assert set(scope["unchecked"]) == {"model", "regressors", "lags"}
    assert scope["status"] == "not_covered"


def test_xtdpdsys_map_is_system_only():
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(function="xtdpdsys", model="difference")
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(function="xtabond", steps="iterated")


def test_steps_two_is_labelled_two_step():
    # steps=2 is twostep=True bit for bit; model_info and the method label
    # used to say one-step because they read the twostep flag.
    from pathlib import Path

    import pandas as pd

    ab = pd.read_csv(
        Path(__file__).parent / "reference_parity/_fixtures/dynpanel_abdata.csv"
    )
    a = sp.xtabond(ab, y="n", id="id", time="year", steps=2)
    b = sp.xtabond(ab, y="n", id="id", time="year", twostep=True)
    assert a.method == b.method == "Arellano-Bond (Two-step GMM)"
    assert a.model_info["twostep"] and a.model_info["windmeijer"]
    assert a.se == b.se
