"""Stata / R command names reach the StatsPAI function they translate to.

The 2026-10-19 audit: ``sp.search_functions('csdid')`` ranked
``cs_jackknife`` first, ``ivreg2`` / ``bacondecomp`` / ``boottest`` /
``matchit`` returned nothing, and ``sp.csdid`` suggested ``sp.sdid``. The
index is read from the ``sp.from_stata`` / ``sp.from_r`` translator tables,
so the expected answer for each name below is the translator's own target.
"""

from __future__ import annotations

import pytest

import statspai as sp
from statspai import _discovery_commands as dc
from statspai.agent._translation import R_FUNCTION_MAP, STATA_COMMAND_MAP
from statspai.registry import _REGISTRY, _ensure_full_registry


def _translated_tool(line: str, lang: str = "stata") -> str:
    out = sp.from_stata(line) if lang == "stata" else sp.from_r(line)
    assert out.get("tool"), out
    return out["tool"]


# (query, a line the translator accepts, language)
TRANSLATED = [
    ("csdid", "csdid y, ivar(id) time(t) gvar(g)", "stata"),
    ("ivreghdfe", "ivreghdfe y (d = z), absorb(id)", "stata"),
    ("ivreg2", "ivreg2 y (d = z)", "stata"),
    ("ivregress", "ivregress 2sls y (d = z)", "stata"),
    ("boottest", "boottest x", "stata"),
    ("bacondecomp", "bacondecomp y d, ddetail", "stata"),
    (
        "did2s",
        "did2s y, first_stage(i.id i.t) second_stage(i.d) treatment(d) cluster(id)",
        "stata",
    ),
    ("reghdfe", "reghdfe y x, absorb(id)", "stata"),
    ("xtreg", "xtreg y x", "stata"),
    ("att_gt", 'att_gt(yname="y", tname="t", idname="id", gname="g", data=df)', "r"),
    ("matchit", "matchit(d ~ x1 + x2, data = df)", "r"),
]


@pytest.mark.parametrize("query, line, lang", TRANSLATED)
def test_command_name_ranks_translator_target_first(query, line, lang):
    expected = _translated_tool(line, lang)
    hits = sp.search_functions(query)
    assert hits[0]["name"] == expected, [h["name"] for h in hits[:3]]
    assert hits[0]["matched_command"].endswith(query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("eventstudyinteract", "sun_abraham"),
        ("sunab", "sun_abraham"),
        ("fixest::sunab", "sun_abraham"),
        ("xtabond2", "xtabond"),
        ("csdid2", "callaway_santanna"),
        ("honestdid", "honest_did"),
    ],
)
def test_names_outside_the_translators_resolve(query, expected):
    assert sp.search_functions(query)[0]["name"] == expected


def test_registered_name_is_not_overridden_by_a_command():
    # R's did::att_gt package is called `did`; sp.did is a function.
    assert sp.search_functions("did")[0]["name"] == "did"
    assert sp.search_functions("synth")[0]["name"] == "synth"


def test_command_word_inside_a_longer_query():
    names = [h["name"] for h in sp.search_functions("csdid with covariates")[:3]]
    assert names[0] == "callaway_santanna"


def test_supplement_holds_only_names_the_translators_lack():
    # When a translator learns one of these, the entry belongs there.
    from statspai.agent._translation import _stata

    known = {k.lower() for k in STATA_COMMAND_MAP} | {k.lower() for k in R_FUNCTION_MAP}
    known |= {k.lower() for k in getattr(_stata, "_UNTRANSLATED_GUIDANCE", {})}
    assert not (set(dc.SUPPLEMENT) & known), sorted(set(dc.SUPPLEMENT) & known)
    _ensure_full_registry()
    for name, (_lang, targets) in dc.SUPPLEMENT.items():
        for t in targets:
            assert t in _REGISTRY, (name, t)


def test_every_translator_command_is_indexed_to_registered_functions():
    _ensure_full_registry()
    index = dc.command_index()
    for command in STATA_COMMAND_MAP:
        for _lang, targets in index.get(command.lower(), []):
            assert all(t in _REGISTRY for t in targets), (command, targets)
    # Every handler with a parseable target is reachable.
    assert "csdid" in index and "att_gt" in index


@pytest.mark.parametrize(
    "name, target",
    [("csdid", "sp.callaway_santanna"), ("att_gt", "sp.callaway_santanna")],
)
def test_module_getattr_points_at_the_translation(name, target):
    with pytest.raises(AttributeError) as exc:
        getattr(sp, name)
    msg = str(exc.value)
    assert target in msg
    assert "sp.sdid" not in msg


def test_hasattr_stays_false_for_command_names():
    assert not hasattr(sp, "csdid")
