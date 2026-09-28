"""Natural-language discovery: ``sp.search_functions`` on research questions.

The 2026-09-28 audit found domain-noun queries returning ``[]`` ("minimum
wage employment difference in differences"), estimand vocabulary missing
its estimator ("local average treatment effect" never reached ``iv``) and
family queries ranking a variant above the dispatcher ("synthetic control
single treated state" put ``synth_survival`` first). Each row below is a
query an applied researcher or agent actually types, with the function that
must appear in the top three.
"""

from __future__ import annotations

import pytest

import statspai as sp

QUERIES = [
    ("minimum wage employment difference in differences", "did"),
    ("how do I estimate a local average treatment effect", "iv"),
    ("effect on compliers with an instrument", "iv"),
    ("synthetic control single treated state", "synth"),
    ("regression kink design", "rkd"),
    ("bunching at a tax kink", "bunching"),
    ("heterogeneous treatment effects", "metalearner"),
    ("conditional average treatment effect with a causal forest", "causal_forest"),
    ("propensity score matching", "match"),
    ("doubly robust estimate of the average treatment effect", "aipw"),
    ("two stage least squares", "ivreg"),
    ("event study with leads and lags", "event_study"),
    ("causal mediation analysis", "mediate"),
    ("targeted maximum likelihood", "tmle"),
    ("inverse probability weighting", "ipw"),
    ("staggered adoption across states", "callaway_santanna"),
    ("regression discontinuity design", "rdrobust"),
    ("two-way fixed effects bias with staggered timing", "bacon_decomposition"),
    ("synthetic difference in differences", "sdid"),
    ("triple differences", "ddd"),
    ("double machine learning partially linear model", "dml"),
    ("mendelian randomization with genetic instruments", "mr"),
    ("oaxaca blinder wage gap decomposition", "decompose"),
    ("sensitivity to unobserved confounding", "sensemakr"),
    ("power analysis for a cluster randomized trial", "power"),
    ("marginal effects after logit", "margins"),
    ("quantile treatment effect", "qte"),
    ("multiple hypothesis testing correction", "adjust_pvalues"),
]


@pytest.mark.parametrize("query, expected", QUERIES)
def test_research_question_finds_the_estimator(query, expected):
    names = [h["name"] for h in sp.search_functions(query)[:3]]
    assert expected in names, (query, names)


@pytest.mark.parametrize(
    "query, dispatcher",
    [
        ("synthetic control", "synth"),
        ("synthetic control single treated state", "synth"),
        ("difference in differences", "did"),
        ("regression discontinuity", "rdrobust"),
        ("instrumental variables", "iv"),
    ],
)
def test_family_query_ranks_the_dispatcher_first(query, dispatcher):
    assert sp.search_functions(query)[0]["name"] == dispatcher


def test_domain_nouns_never_empty_a_design_query():
    hits = sp.search_functions("minimum wage employment difference in differences")
    assert hits and hits[0]["match"] == "full"


def test_partial_fallback_is_flagged_and_bounded():
    # Three unrelated domain nouns plus one real word: nothing clears the
    # half-of-the-words bar, but the query is not empty-handed.
    hits = sp.search_functions("zebra giraffe okapi bootstrap")
    assert hits, "a query with a real term must not return []"
    assert all(h["match"] == "partial" for h in hits)
    assert len(hits) <= 25


def test_nonsense_query_returns_empty():
    assert sp.search_functions("qqqqzzzz") == []


def test_every_hit_carries_match_flag():
    for h in sp.search_functions("staggered adoption event study"):
        assert h["match"] in {"full", "partial"}


def test_alias_ranks_below_canonical_and_is_labelled():
    names = [h["name"] for h in sp.search_functions("front door adjustment")]
    assert names.index("front_door") < names.index("frontdoor")
    hit = next(h for h in sp.search_functions("front door") if h["name"] == "frontdoor")
    assert hit["alias_of"] == "front_door"


def test_card_text_breaks_ties_without_creating_matches():
    from statspai import _discovery_search as ds
    from statspai import registry

    registry._ensure_full_registry()
    # Card text alone never makes a function match.
    spec = registry._REGISTRY["did"]
    assert "parallel" in ds._card_text(spec)
    hits = ds.rank({"did": spec}, "qqqqzzzz parallel")
    assert hits == [] or hits[0]["match"] == "partial"


def test_concept_targets_are_registered():
    from statspai import _discovery_search as ds

    registered = set(sp.list_functions())
    missing = {
        concept: [t for t in targets if t not in registered]
        for concept, targets in ds.CONCEPT_TARGETS.items()
    }
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, missing


def test_phrase_tokens_have_synonyms_or_targets():
    from statspai import _discovery_search as ds

    for _phrase, token in ds.PHRASES:
        assert token in ds.SYNONYMS or token in ds.CONCEPT_TARGETS, token
