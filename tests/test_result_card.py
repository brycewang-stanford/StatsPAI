"""``sp.result_card``: one auditable summary per fitted result (review §6.1).

The card consolidates what already existed in scattered places -- estimand,
provenance call arguments and data fingerprint, covariance convention,
validation_scope evidence, registry assumptions / limitations -- and is the
same object in Python and in MCP tool responses.
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


SECTIONS = {
    "function",
    "estimand",
    "sample",
    "specification",
    "inference",
    "provenance",
    "evidence",
    "assumptions",
    "limitations",
}


def test_cs_card_has_configuration_evidence_and_declared_assumptions():
    df = sp.datasets.mpdta()
    fit = sp.callaway_santanna(df, y="lemp", g="first_treat", t="year", i="countyreal")
    card = sp.result_card(fit)
    assert set(card) == SECTIONS
    assert card["function"] == "callaway_santanna"
    assert card["estimand"]["label"] == "ATT"
    assert card["estimand"]["control_group"] == "nevertreated"
    assert card["sample"]["n_used"] == len(df)
    assert card["evidence"]["level"] == "configuration"
    assert card["evidence"]["status"] == sp.validation_scope(fit)["status"]
    assert "not established by this fit" in card["assumptions"]["status"]
    assert "pretrend_test" in card["assumptions"]["diagnostics_run"]
    json.dumps(card)  # JSON-safe
    assert card == fit.result_card()


def test_iv_card_reports_se_gap_not_just_function_tier():
    card_df = sp.datasets.card_1995()
    fit = sp.iv(
        "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)",
        data=card_df,
        robust="hc3",
    )
    ev = sp.result_card(fit)["evidence"]
    assert ev["status"] == "estimate_only"
    assert ev["outputs"]["se"] == "not_covered"


def test_unmapped_function_says_so():
    d = pd.DataFrame(
        {"y": np.r_[np.zeros(30), np.ones(30)], "x": np.linspace(-1, 1, 60)}
    )
    d["y"] = (d["x"] + np.sin(np.arange(60)) > 0).astype(int)
    card = sp.result_card(sp.logit("y ~ x", d))
    assert card["function"] == "logit"
    assert card["evidence"]["level"] == "function"
    assert card["evidence"]["status"] == "no_configuration_map"
    assert "Pr(y=1)" in card["estimand"]["scale"]
    assert card["inference"]["reference_distribution"] == "normal"
    assert card["specification"]["formula"] == "y ~ x"


def test_sample_section_reports_rows_marked_out():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=200), "g": rng.integers(0, 20, 200)})
    d["y"] = d["x"] + rng.normal(size=200)
    d.loc[:4, "g"] = np.nan
    card = sp.result_card(sp.regress("y ~ x", d, vce="cluster g"))
    assert card["sample"]["n_input_rows"] == 200
    assert card["sample"]["n_used"] == 195
    assert card["sample"]["n_not_used"] == 5
    assert card["inference"]["reference_distribution"] == "t(19)"
    assert card["inference"]["full_covariance"] is True


def test_provenance_now_recorded_for_glm_family_and_feols():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"x": rng.normal(size=300), "g": rng.integers(0, 5, 300)})
    d["c"] = rng.poisson(np.exp(0.2 + 0.3 * d["x"]))
    d["b"] = (d["x"] + rng.normal(size=300) > 0).astype(int)
    for fit, fn in [
        (sp.logit("b ~ x", d), "sp.logit"),
        (sp.probit("b ~ x", d), "sp.probit"),
        (sp.poisson("c ~ x", d), "sp.poisson"),
        (sp.nbreg("c ~ x", d), "sp.nbreg"),
        (sp.glm("c ~ x", d, family="poisson"), "sp.glm"),
    ]:
        assert fit._provenance.function == fn
        assert fit._provenance.data_shape == [300, 4]
    pytest.importorskip("pyfixest")
    assert sp.feols("c ~ x | g", d)._provenance.function == "sp.feols"


def test_binary_results_pickle():
    """Regression: a closure attribute made every logit/probit result
    unpicklable (MCP result cache, replication packs)."""
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"x": rng.normal(size=200)})
    d["b"] = (d["x"] + rng.normal(size=200) > 0).astype(int)
    fit = sp.logit("b ~ x", d)
    back = pickle.loads(pickle.dumps(fit))
    np.testing.assert_allclose(back.predict(), fit.predict())
    assert back.classification_table()["pcp"] == fit.classification_table()["pcp"]


def test_markdown_render_names_sections():
    fit = sp.regress("lwage ~ educ", sp.datasets.card_1995())
    md = sp.result_card(fit).to_markdown()
    for sec in ("estimand", "sample", "inference", "evidence", "provenance"):
        assert f"**{sec}**" in md


def test_mcp_tool_response_carries_the_same_card(tmp_path):
    from statspai.agent.tools._dispatch import execute_tool

    df = sp.datasets.card_1995()
    out = execute_tool("regress", {"formula": "lwage ~ educ + exper"}, data=df)
    assert out["result_card"]["function"] == "regress"
    ref = sp.result_card(sp.regress("lwage ~ educ + exper", df))
    for sec in ("estimand", "inference", "evidence"):
        assert out["result_card"][sec] == ref[sec]
    minimal = execute_tool(
        "regress", {"formula": "lwage ~ educ"}, data=df, detail="minimal"
    )
    assert "result_card" not in minimal
