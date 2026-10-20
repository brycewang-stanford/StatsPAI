"""``sp.from_stata('wyoung ...')``: what is translated and what is refused.

The executed translation is checked against Stata ``wyoung`` in
``tests/reference_parity/test_westfall_young_wyoung_parity.py``.
"""

from __future__ import annotations

import pytest

import statspai as sp


def test_cluster_form_maps_every_argument():
    out = sp.from_stata(
        "wyoung y1 y2, cmd(reg OUTCOMEVAR d x, vce(cluster g)) familyp(d) "
        "permute(d) cluster(g) reps(5000) seed(7)"
    )
    assert out["ok"]
    assert out["arguments"] == {
        "y": ["y1", "y2"],
        "treat": "d",
        "covariates": ["x"],
        "cluster": "g",
        "n_perms": 5000,
        "seed": 7,
    }


@pytest.mark.parametrize(
    "line, reason",
    [
        ("wyoung y1 y2, cmd(regress OUTCOMEVAR d, r) familyp(d)", "bootstraps"),
        (
            "wyoung y1 y2, cmd(regress OUTCOMEVAR d, vce(hc3)) familyp(d) permute(d)",
            "classical, HC1",
        ),
        (
            "wyoung y1 y2, cmd(regress OUTCOMEVAR d) familyp(d) permute(d) "
            "cluster(g)",
            "cluster",
        ),
        (
            "wyoung y1 y2, cmd(regress OUTCOMEVAR d, r) familyp(d) permute(d) "
            "strata(s)",
            "stratum effects",
        ),
        (
            "wyoung y1 y2, cmd(regress OUTCOMEVAR d i.g, r) familyp(d) permute(d)",
            "stratum effects",
        ),
        ("wyoung y1 y2, cmd(logit OUTCOMEVAR d) familyp(d) permute(d)", "regress"),
        (
            "wyoung y1 y2, cmd(regress OUTCOMEVAR d, cluster(g)) familyp(d) permute(d)",
            "cluster",
        ),
        (
            "wyoung y1 y2, cmd(regress OUTCOMEVAR d, r) familyp(d) permute(d) "
            "subgroup(s)",
            "subgroup",
        ),
        (
            'wyoung, cmd("regress y1 d" "regress y2 d") familyp(d) permute(d)',
            "syntax 1",
        ),
    ],
)
def test_refusals_name_the_reason_and_the_function(line, reason):
    out = sp.from_stata(line)
    assert out["ok"] is False
    assert reason in out["error"]
    assert "westfall_young" in out["statspai_functions"]


@pytest.mark.parametrize("opts", ["", ", vce(ols)"])
def test_regress_without_se_option_is_the_classical_t(opts):
    out = sp.from_stata(
        f"wyoung y1 y2, cmd(regress OUTCOMEVAR d x{opts}) familyp(d) "
        "permute(d) reps(2000)"
    )
    assert out["ok"] and out["complete"]
    assert out["arguments"]["robust"] is False
