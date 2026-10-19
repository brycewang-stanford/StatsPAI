"""``DAG.instrumental_variables`` against R ``dagitty::instrumentalVariables``.

Reference: dagitty 0.3.4 (GPL-2, used as a black box) on 200 random DAGs
with latent confounders (``_fixtures/dag_instruments_graphs.json``, from
``_generate_dag_instruments_graphs.py`` and
``_generate_dag_instruments_dagitty.R``). The instrument sets are equal in
all 200 graphs. dagitty reports its nearest separator as the conditioning
set and we report the smallest one, so the sets are not compared directly:
each of them is checked against the definition of an ancestral instrument
(``statspai.dag.instruments``) by d-separation.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


def _dag(gr):
    d = sp.DAG()
    for n in gr["nodes"]:
        d.add_node(n)
    for a, b in gr["edges"]:
        d.add_edge(a, b)
    for a, b in gr["bi"]:
        d.add_bidirected(a, b)
    return d


def _is_ancestral_instrument(d, z, w, x="x", y="y"):
    """The ancestral-instrument conditions, checked directly."""
    on_path = d.descendants(x) & (d.ancestors(y) | {y})
    gc = sp.DAG()
    for n in d.nodes:
        gc.add_node(n)
    for p, c in d.edges:
        if not (p == x and c in on_path):
            gc.add_edge(p, c)
    forbidden = {x, y} | d.descendants(y)
    for v in on_path:
        forbidden |= d.descendants(v) | {v}
    anc = d.ancestors(y) | d.ancestors(z)
    w = set(w)
    return (
        not (w & forbidden)
        and w <= anc
        and w <= d.observed_nodes
        and gc.d_separated(z, y, w)
        and not d.d_separated(z, x, w)
    )


@pytest.fixture(scope="module")
def cases():
    graphs = json.loads(
        (_FIX / "dag_instruments_graphs.json").read_text(encoding="utf-8")
    )
    ref = json.loads(
        (_FIX / "dag_instruments_dagitty.json").read_text(encoding="utf-8")
    )
    return list(zip(graphs, ref))


def test_instrument_sets_match_dagitty(cases):
    n_with = 0
    for i, (gr, r) in enumerate(cases):
        d = _dag(gr)
        ours = {
            e["instrument"]: e["conditioning"]
            for e in d.instrumental_variables("x", "y")
        }
        theirs = {e["I"]: e["Z"] for e in (r.values() if isinstance(r, dict) else r)}
        assert set(ours) == set(theirs), i
        n_with += bool(ours)
        for z, w in ours.items():
            assert _is_ancestral_instrument(d, z, w), (i, z, w)
        for z, w in theirs.items():
            assert _is_ancestral_instrument(d, z, w), (i, z, w)
            # ours is the smallest valid set
            assert len(ours[z]) <= len(w), (i, z)
    assert n_with > 50  # the fixture exercises many graphs with instruments


def test_textbook_cases():
    classic = sp.dag("i -> x; x -> y; x <-> y")
    assert classic.instrumental_variables("x", "y") == [
        {"instrument": "i", "conditioning": []}
    ]
    conditional = sp.dag("i -> x; x -> y; x <-> y; z -> y; z -> i")
    assert conditional.instrumental_variables("x", "y") == [
        {"instrument": "i", "conditioning": ["z"]}
    ]
    # dagitty lists c and u here; z needs the collider c in W, so it is a
    # conditional instrument but not an ancestral one
    collider = sp.dag("z -> c; u -> c; u -> x; x -> y; x <-> y")
    assert collider.instrumental_variables("x", "y") == [
        {"instrument": "c", "conditioning": []},
        {"instrument": "u", "conditioning": []},
    ]
    loose = collider.instrumental_variables("x", "y", ancestral=False)
    assert {"instrument": "z", "conditioning": ["c"]} in loose


def test_unknown_node_is_refused():
    with pytest.raises(sp.MethodIncompatibility, match="not a node"):
        sp.dag("i -> x; x -> y").instrumental_variables("x", "w")
