"""``sp.dag`` queries and the do-calculus rule checks against R ``dagitty``.

Reference: dagitty 0.3.4 (GPL, run as a black box) on 120 random acyclic
graphs with 4 to 9 nodes, forty of them with declared latent nodes and
forty with bidirected edges (``_fixtures/_generate_dag_dagitty_cases.py``
draws the graphs and the queries, ``_generate_dag_dagitty_R.R`` answers
them; the answers and the graphs are in ``_fixtures/dag_dagitty_R.json``).

Everything compared is a set or a truth value, so the comparison is exact:

* minimal adjustment sets for the total effect
  (``adjustmentSets(type = "minimal")``);
* implied conditional independencies
  (``impliedConditionalIndependencies(type = "missing.edge")``);
* 1,440 d-separation statements (``dseparated``);
* the variables that are instruments (``instrumentalVariables``). The
  conditioning set reported with an instrument is the smallest one here
  and the nearest separator in dagitty; both satisfy the definition, so
  only the instruments are compared;
* every path between exposure and outcome and whether it is open
  (``paths``), on the graphs without bidirected edges, where a path is the
  same node sequence on both sides;
* the three rules of do-calculus on 720 queries ``(Y, X, Z, W)``. dagitty
  has no function for the rules. The reference builds each mutilated
  graph in R from Pearl's definitions (edges into ``X`` removed; edges
  out of ``Z``; edges into the members of ``Z`` that are not ancestors of
  ``W`` once the edges into ``X`` are gone) and asks ``dseparated`` on it.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_R = json.loads((_FIX / "dag_dagitty_R.json").read_text(encoding="utf-8"))
CASES = _R["cases"]
IDS = sorted(CASES)


def _graph(case):
    spec = "; ".join(
        [f"{a} -> {b}" for a, b in case["directed"]]
        + [f"{a} <-> {b}" for a, b in case["bidirected"]]
        + list(case["nodes"])
    )
    return sp.dag(spec, latent=case["latent"] or None)


def test_fixture_is_dagitty_output_on_all_three_kinds_of_graph():
    assert _R["versions"]["dagitty"] == "0.3.4"
    assert len(CASES) == 120
    assert sum(bool(c["case"]["latent"]) for c in CASES.values()) >= 30
    assert sum(bool(c["case"]["bidirected"]) for c in CASES.values()) == 40
    # the queries are not all trivially true or all false
    answers = [a for c in CASES.values() for a in c["dsep"]]
    assert 300 < sum(answers) < len(answers) - 300
    assert sum(bool(c["adjustment"]) for c in CASES.values()) > 60
    assert sum(not c["adjustment"] for c in CASES.values()) > 3
    assert sum(bool(c["instruments"]) for c in CASES.values()) > 30


@pytest.mark.parametrize("cid", IDS)
def test_minimal_adjustment_sets_are_dagittys(cid):
    ref = CASES[cid]
    case = ref["case"]
    ours = _graph(case).adjustment_sets(case["exposure"], case["outcome"])
    assert sorted(sorted(s) for s in ours) == sorted(
        sorted(s) for s in ref["adjustment"]
    )


@pytest.mark.parametrize("cid", IDS)
def test_implied_independencies_are_dagittys(cid):
    ref = CASES[cid]
    ours = sorted(
        (min(a, b), max(a, b), tuple(sorted(z)))
        for a, b, z in _graph(ref["case"]).implied_independencies()
    )
    want = sorted(
        (min(i["x"], i["y"]), max(i["x"], i["y"]), tuple(i["z"]))
        for i in ref["implied"]
    )
    assert ours == want


@pytest.mark.parametrize("cid", IDS)
def test_d_separation_statements_are_dagittys(cid):
    ref = CASES[cid]
    g = _graph(ref["case"])
    ours = [g.d_separated(q["x"], q["y"], set(q["z"])) for q in ref["case"]["dsep"]]
    assert ours == ref["dsep"]


@pytest.mark.parametrize("cid", IDS)
def test_instruments_are_dagittys(cid):
    ref = CASES[cid]
    case = ref["case"]
    found = _graph(case).instrumental_variables(case["exposure"], case["outcome"])
    assert sorted({i["instrument"] for i in found}) == sorted(
        {i["instrument"] for i in ref["instruments"]}
    )


@pytest.mark.parametrize("cid", [c for c in IDS if not CASES[c]["case"]["bidirected"]])
def test_paths_and_their_status_are_dagittys(cid):
    ref = CASES[cid]
    case = ref["case"]
    ours = sorted(
        (tuple(p["path"]), p["open"])
        for p in _graph(case).path_status(case["exposure"], case["outcome"])
    )
    want = sorted(
        (tuple(re.split(r"\s*(?:<->|->|<-)\s*", s)), is_open)
        for s, is_open in zip(ref["paths"]["paths"], ref["paths"]["open"])
    )
    assert ours == want


def test_rule_queries_cover_both_answers_for_every_rule():
    answers = [r for c in CASES.values() for r in c["rules"]]
    assert len(answers) == 720
    for rule in ("rule1", "rule2", "rule3"):
        n_true = sum(a[rule] for a in answers)
        assert 100 < n_true < 620, rule
    # rule 3 can hold where rule 2 fails and the reverse
    assert any(a["rule3"] and not a["rule2"] for a in answers)
    assert any(a["rule2"] and not a["rule3"] for a in answers)


@pytest.mark.parametrize("cid", IDS)
def test_do_calculus_rules_match_d_separation_in_dagitty(cid):
    ref = CASES[cid]
    g = _graph(ref["case"])
    for q, want in zip(ref["case"]["rules"], ref["rules"]):
        kw = dict(Y=q["Y"], X=q["X"], Z=q["Z"], W=q["W"] or None)
        assert sp.do_rule1(g, **kw).applicable is want["rule1"], q
        assert sp.do_rule2(g, **kw).applicable is want["rule2"], q
        assert sp.do_rule3(g, **kw).applicable is want["rule3"], q
        both = sp.do_calculus_apply(g, **kw)
        assert [c.rule for c in both] == [1, 2, 3]
        assert [c.applicable for c in both] == [
            want["rule1"],
            want["rule2"],
            want["rule3"],
        ], q
