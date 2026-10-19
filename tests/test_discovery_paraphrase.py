"""Paraphrased research questions find the design's entry point (top 3).

The 2026-10-19 audit: queries that describe the design without naming it
("treatment adopted at different times", "only one treated unit ...
donors", "first stage F statistic is small") put none of the right
functions in the top three. This is the eval set for
``_discovery_search.PATTERNS``; every row is a sentence a user types.
"""

from __future__ import annotations

import pytest

import statspai as sp

STAGGERED = {"callaway_santanna", "sun_abraham", "did_imputation", "bjs", "etwfe"}
SYNTH = {"synth", "sdid", "augsynth"}
WEAK_IV = {"iv_diag", "effective_f_test", "weakrobust", "anderson_rubin_ci"}
RD = {"rdrobust", "rd"}

EVAL = [
    ("treatment adopted at different times", STAGGERED),
    ("states adopted the policy in different years", STAGGERED),
    ("units are treated at different times", STAGGERED),
    ("policy rolled out at different times across counties", STAGGERED),
    ("only one treated unit build comparison from donors", SYNTH),
    ("single treated unit with a donor pool", SYNTH),
    ("one treated state and many control states", SYNTH),
    ("first stage F statistic is small", WEAK_IV),
    ("weak first stage", WEAK_IV),
    ("my instrument is weak", WEAK_IV),
    ("eligibility determined by a score cutoff", RD),
    ("treatment assigned above a threshold of a running variable", RD),
]


@pytest.mark.parametrize("query, acceptable", EVAL)
def test_paraphrase_top3(query, acceptable):
    top = [h["name"] for h in sp.search_functions(query, limit=3)]
    assert set(top) & acceptable, (query, top)


def test_eval_set_hit_rate_is_complete():
    hits = sum(
        bool({h["name"] for h in sp.search_functions(q, limit=3)} & ok)
        for q, ok in EVAL
    )
    assert hits == len(EVAL)


def test_limit_and_category():
    assert len(sp.search_functions("regression")) == 10
    everything = sp.search_functions("regression", limit=None)
    assert len(everything) > 10
    assert sp.search_functions("regression", limit=3) == everything[:3]
    cat = sp.search_functions("regression", category="regression", limit=None)
    assert cat and all(h["category"] == "regression" for h in cat)
    with pytest.raises(ValueError):
        sp.search_functions("regression", limit=0)
    with pytest.raises(TypeError):
        sp.search_functions("regression", limit="5")
