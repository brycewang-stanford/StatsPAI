"""``RecommendationResult.identification`` (review §6.2).

A recommendation picks a tool; it does not identify an effect. Each result
now states the assumptions no data can verify, the checks that bear on them,
the questions to ask, and whether the design was declared or only inferred
from the data's shape.
"""

import warnings

import pytest

import statspai as sp


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_observational_recommendation_names_unconfoundedness():
    df = sp.cps_wage()
    rec = sp.recommend(
        df, y="log_wage", treatment="union", covariates=["education", "experience"]
    )
    ident = rec.identification
    assert ident["design_source"] == "detected_from_data_shape"
    assert ident["claim"] == "causal only under selection on observables"
    assert any("unmeasured confounding" in a for a in ident["untestable_assumptions"])
    assert {"sp.sensemakr", "sp.overlap_plot"} <= {
        c["function"] for c in ident["checks"]
    }
    assert "inferred from the data's shape" in ident["questions"][0]
    assert "identification" in rec.to_dict()
    assert "IDENTIFICATION" in rec.summary()


def test_declared_did_does_not_ask_about_the_design():
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=1)
    rec = sp.recommend(
        df, y="y", treatment="treated", id="unit", time="time", design="did"
    )
    ident = rec.identification
    assert ident["design_source"] == "declared"
    assert "inferred" not in " ".join(ident["questions"])
    assert any("Parallel trends" in a for a in ident["untestable_assumptions"])
    pre = [c for c in ident["checks"] if c["function"] == "sp.pretrends_test"]
    assert pre and "not evidence" in pre[0]["check"]


def test_no_treatment_is_descriptive_only():
    rec = sp.recommend(sp.cps_wage(), y="log_wage")
    assert rec.identification["claim"] == "descriptive_only"


def test_every_named_check_exists():
    from statspai.smart._identification import _BRIEFS

    for brief in _BRIEFS.values():
        for _, fn in brief["checks"]:
            assert hasattr(sp, fn), fn
