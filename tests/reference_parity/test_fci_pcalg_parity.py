"""``sp.fci`` against ``pcalg::fci``, mark for mark.

FCI with the Fisher-z test is deterministic, so the PAG can be compared with
the reference exactly: the skeleton after the Possible-D-SEP pass and every
edge mark after Zhang's ten rules.

* 24 data sets without latent variables (``pc_pcalg_data.csv``), small
  enough that the tests err;
* 12 data sets with two or three hidden common causes
  (``fci_latent_data.csv``), where the reference PAGs contain bidirected
  edges and the Possible-D-SEP pass removes edges the PC skeleton keeps.

Reference: ``_fixtures/fci_pcalg_R.json`` from ``_generate_fci_pcalg.R``
(pcalg 2.7-12, R 4.5.2). pcalg is a black box on the output side; the
implementation follows Zhang (2008) and Colombo et al. (2012).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.causal_discovery import _fci_core as core

FIX = Path(__file__).parent / "_fixtures"
SETS = {"no_latent": "pc_pcalg_data.csv", "latent": "fci_latent_data.csv"}


@pytest.fixture(scope="module")
def cases():
    ref = json.loads((FIX / "fci_pcalg_R.json").read_text(encoding="utf-8"))
    out = {}
    for key, file in SETS.items():
        data = pd.read_csv(FIX / file)
        rows = []
        for case in ref[key]:
            d = data[data["case"] == case["case"]].drop(columns="case")
            d = d.dropna(axis=1, how="all").reset_index(drop=True)
            assert d.shape == (case["n"], case["p"])
            rows.append((case, d))
        out[key] = rows
    assert len(out["no_latent"]) == 24 and len(out["latent"]) == 12
    return out


@pytest.mark.parametrize("key", sorted(SETS))
def test_pag_equals_pcalg_fci(cases, key):
    for case, d in cases[key]:
        out = sp.fci(d, alpha=case["alpha"])
        expected = np.array(case["fci"]).reshape(case["p"], case["p"])
        np.testing.assert_array_equal(
            out.pag_right.values, expected, err_msg=f"{key} case {case['case']}"
        )
        # the two mark matrices are each other's transpose
        np.testing.assert_array_equal(out.pag_left.values, expected.T)
        np.testing.assert_array_equal(out.skeleton.values, (expected != 0).astype(int))


@pytest.mark.parametrize("key", sorted(SETS))
def test_without_possible_dsep_the_skeleton_is_rfcis(cases, key):
    for case, d in cases[key]:
        out = sp.fci(d, alpha=case["alpha"], possible_dsep=False)
        expected = np.array(case["rfci_skeleton"]).reshape(case["p"], case["p"])
        np.testing.assert_array_equal(out.skeleton.values, expected)
        assert out.n_removed_by_possible_dsep == 0


def test_the_fixture_exercises_what_it_claims(cases):
    """Bidirected edges in the latent references, and edges that only the
    Possible-D-SEP pass removes in both sets."""
    for key, min_removed in (("no_latent", 10), ("latent", 5)):
        removed = bidirected = 0
        for case, d in cases[key]:
            removed += sp.fci(d, alpha=case["alpha"]).n_removed_by_possible_dsep
            g = np.array(case["fci"]).reshape(case["p"], case["p"])
            bidirected += int(((g == 2) & (g.T == 2)).sum() // 2)
        assert removed >= min_removed
        assert bidirected >= 10


def test_a_latent_common_cause_is_drawn_bidirected():
    """A -> X <- L -> Y <- B with L hidden: X <-> Y, and neither is drawn
    as a cause of the other."""
    rng = np.random.default_rng(0)
    n = 4000
    a, b, hidden = rng.standard_normal((3, n))
    x = a + hidden + 0.5 * rng.standard_normal(n)
    y = b + hidden + 0.5 * rng.standard_normal(n)
    out = sp.fci(pd.DataFrame({"A": a, "X": x, "Y": y, "B": b}), alpha=0.01)
    marks = {(i, j): m for i, m, j in out.edges}
    assert marks[("X", "Y")] == "<->"
    assert marks[("A", "X")] == "o->" and marks[("Y", "B")] == "<-o"


def test_possible_dsep_reaches_through_colliders_and_triangles():
    # 0 *-> 1 <-* 2 - 3, and 4 hanging off 2 through a non-collider
    P = np.zeros((5, 5), dtype=int)

    def edge(i, j, at_i, at_j):
        P[j, i], P[i, j] = at_i, at_j

    edge(0, 1, core.CIRCLE, core.ARROW)
    edge(2, 1, core.CIRCLE, core.ARROW)
    edge(2, 3, core.CIRCLE, core.CIRCLE)
    assert core.possible_dsep(P, 0) == [1, 2]  # through the collider at 1
    assert core.possible_dsep(P, 3) == [2]  # 2 - 1 is not reached: no collider
    edge(3, 1, core.CIRCLE, core.CIRCLE)  # now 1, 2, 3 form a triangle
    assert core.possible_dsep(P, 3) == [0, 1, 2]


def test_rule_one_and_rule_four_on_hand_built_graphs():
    # R1: a *-> b o-o c with a, c non-adjacent  =>  b -> c
    P = np.zeros((3, 3), dtype=int)
    P[0, 1], P[1, 0] = core.ARROW, core.CIRCLE
    P[1, 2], P[2, 1] = core.CIRCLE, core.CIRCLE
    core.apply_rules(P, {})
    assert P[1, 2] == core.ARROW and P[2, 1] == core.TAIL
    # R4: discriminating path d *-> a <-> b o-* c with a -> c and d, c
    # non-adjacent; b in sepset(d, c)  =>  b -> c
    d, a, b, c = 0, 1, 2, 3
    P = np.zeros((4, 4), dtype=int)
    P[d, a], P[a, d] = core.ARROW, core.CIRCLE
    P[a, b], P[b, a] = core.ARROW, core.ARROW
    P[a, c], P[c, a] = core.ARROW, core.TAIL
    P[b, c], P[c, b] = core.CIRCLE, core.CIRCLE
    Q = P.copy()
    core.apply_rules(P, {(d, c): {b}, (c, d): {b}})
    assert P[b, c] == core.ARROW and P[c, b] == core.TAIL
    core.apply_rules(Q, {(d, c): set(), (c, d): set()})
    assert Q[b, c] == core.ARROW and Q[c, b] == core.ARROW  # b <-> c
