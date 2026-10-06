"""``sp.pc_algorithm`` against ``pcalg::pc`` where the sample contradicts itself.

The PC algorithm with the Fisher-z test is deterministic, so its graph can be
compared with the reference exactly. The 24 data sets here are small random
linear-Gaussian SEMs (5 to 11 variables, 120 or 400 rows): small enough that
the tests err and, in 14 of them, two colliders claim one edge in opposite
directions. That is the one place where implementations of PC-stable can
differ, because the algorithm does not say which collider wins.

* The skeleton equals pcalg's in all 24, whatever the rule.
* ``collider_conflict='last'`` (a later collider overwrites an earlier one)
  reproduces pcalg's CPDAG in all 24.
* The default ``'first'`` reproduces it wherever no clash is reported, and
  differs only where one is.

Fixtures: ``_fixtures/pc_pcalg_data.csv`` (``_generate_pc_pcalg_data.py``) and
``_fixtures/pc_pcalg_R.json`` (``_generate_pc_pcalg.R``, pcalg 2.7-12, R
4.5.2). pcalg is used as a black box on the output side only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.causal_discovery.pc import PCAlgorithm
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def cases():
    ref = json.loads((FIX / "pc_pcalg_R.json").read_text(encoding="utf-8"))
    data = pd.read_csv(FIX / "pc_pcalg_data.csv")
    out = []
    for case in ref["cases"]:
        d = data[data["case"] == case["case"]].drop(columns="case")
        d = d.dropna(axis=1, how="all").reset_index(drop=True)
        assert d.shape == (case["n"], case["p"])
        expected = np.array(case["stable"]).reshape(case["p"], case["p"])
        out.append((case, d, expected))
    assert len(out) == 24
    return out


def _skeleton(g: np.ndarray) -> np.ndarray:
    return ((g + g.T) > 0).astype(int)


def test_overwriting_colliders_reproduce_pcalg_on_every_case(cases):
    for case, d, expected in cases:
        out = sp.pc_algorithm(d, alpha=case["alpha"], collider_conflict="last")
        np.testing.assert_array_equal(
            out["cpdag"].values, expected, err_msg=f"case {case['case']}"
        )


def test_default_rule_differs_from_pcalg_only_where_it_reports_a_clash(cases):
    n_clash = n_differ = 0
    for case, d, expected in cases:
        out = sp.pc_algorithm(d, alpha=case["alpha"])
        np.testing.assert_array_equal(out["skeleton"].values, _skeleton(expected))
        same = np.array_equal(out["cpdag"].values, expected)
        if out["orientation_conflicts"]:
            n_clash += 1
            n_differ += not same
        else:
            assert same, f"case {case['case']} has no clash and differs"
    # the fixture is only evidence if it contains the hard cases
    assert n_clash == 14
    assert n_differ >= 10


def test_both_rules_report_the_same_cases_and_keep_every_edge(cases):
    for case, d, _ in cases:
        first = sp.pc_algorithm(d, alpha=case["alpha"])
        last = sp.pc_algorithm(d, alpha=case["alpha"], collider_conflict="last")
        assert bool(first["orientation_conflicts"]) == bool(
            last["orientation_conflicts"]
        )
        for out in (first, last):
            np.testing.assert_array_equal(
                _skeleton(out["cpdag"].values), out["skeleton"].values
            )
        if not first["orientation_conflicts"]:
            pd.testing.assert_frame_equal(first["cpdag"], last["cpdag"])


def test_meek_rule_three_under_the_overwrite_rule():
    """a - b, a - c1 -> b <- c2 - a with c1, c2 non-adjacent  =>  a -> b."""
    est = PCAlgorithm(pd.DataFrame({"a": [0.0], "b": [0.0]}), collider_conflict="last")
    a, b, c1, c2 = 0, 1, 2, 3
    adj = np.zeros((4, 4), dtype=int)
    for i, j in [(a, b), (a, c1), (a, c2), (c1, b), (c2, b)]:
        adj[i, j] = adj[j, i] = 1
    g = est._orient_edges(adj, {(c1, c2): {a}, (c2, c1): {a}}, 4)
    assert g[c1, b] == 1 and g[b, c1] == 0
    assert g[c2, b] == 1 and g[b, c2] == 0
    assert g[a, b] == 1 and g[b, a] == 0
    assert g[a, c1] == 1 and g[c1, a] == 1


def test_a_required_edge_survives_an_overwriting_collider():
    """Background knowledge outranks either rule."""
    est = PCAlgorithm(
        pd.DataFrame({"x": [0.0], "y": [0.0]}),
        required=[("m", "x")],
        collider_conflict="last",
    )
    x, m, z = 0, 1, 2
    est._required_idx_directed = [(m, x)]
    adj = np.zeros((3, 3), dtype=int)
    for i, j in [(x, m), (m, z)]:
        adj[i, j] = adj[j, i] = 1
    # the tests say x -> m <- z; the user says m -> x
    g = est._orient_edges(adj, {(x, z): set(), (z, x): set()}, 3)
    assert g[m, x] == 1 and g[x, m] == 0
    assert g[z, m] == 1 and g[m, z] == 0


def test_collider_conflict_is_validated():
    df = pd.DataFrame(np.zeros((5, 2)), columns=["a", "b"])
    with pytest.raises(MethodIncompatibility, match="collider_conflict"):
        sp.pc_algorithm(df, collider_conflict="majority")
