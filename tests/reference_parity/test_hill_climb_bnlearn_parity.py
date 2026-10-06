"""``sp.hill_climb`` against ``bnlearn``: the score, and where the search ends.

The BIC of a graph is a closed form of the data, so it is compared exactly:
four fixed graphs on categorical data (``score(type = "bic")``) and four on
continuous data (``"bic-g"``). The search itself is greedy and ends at a
local optimum that depends on how ties are broken, so what is asserted about
it is what must hold of any correct implementation: the score it reaches is
bnlearn's on these two data sets, no single arc change improves it, and the
skeleton is the one bnlearn finds.

Fixtures: ``_fixtures/hill_climb_discrete.csv`` (1,500 rows, 5 factors),
``hill_climb_gaussian.csv`` (800 rows, 6 variables) and
``hill_climb_bnlearn_R.json`` from ``_generate_hill_climb_bnlearn.R``
(bnlearn 5.2.1, R 4.5.2).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.causal_discovery.hill_climb import (
    _creates_cycle,
    _dag_to_cpdag,
    _DiscreteScore,
    _GaussianScore,
)
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
SETS = {
    "discrete": ("hill_climb_discrete.csv", _DiscreteScore),
    "gaussian": ("hill_climb_gaussian.csv", _GaussianScore),
}


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "hill_climb_bnlearn_R.json").read_text(encoding="utf-8"))


def _parents(model: str, cols):
    out = {}
    for node, pa in re.findall(r"\[([^\]|]+)(?:\|([^\]]+))?\]", model):
        out[cols.index(node)] = frozenset(cols.index(p) for p in pa.split(":") if p)
    return out


@pytest.mark.parametrize("key", sorted(SETS))
def test_bic_of_fixed_graphs_equals_bnlearn(ref, key):
    file, scorer = SETS[key]
    df = pd.read_csv(FIX / file)
    score = scorer(df)
    cols = list(df.columns)
    assert len(ref[key]["fixed"]) == 4
    for item in ref[key]["fixed"]:
        total = sum(
            score(node, pa) for node, pa in _parents(item["graph"], cols).items()
        )
        assert total == pytest.approx(item["score"], rel=1e-12), item["graph"]


@pytest.mark.parametrize("key", sorted(SETS))
def test_search_reaches_bnlearns_score_and_skeleton(ref, key):
    file, _ = SETS[key]
    df = pd.read_csv(FIX / file)
    out = sp.hill_climb(df, data_type=key)
    assert out["score"] == pytest.approx(ref[key]["hc_score"], rel=1e-10)
    mine = {frozenset(e) for e in out["edges"]}
    theirs = {frozenset(e) for e in ref[key]["hc_arcs"]}
    assert mine == theirs
    assert out["score_type"] == ("bic" if key == "discrete" else "bic-g")


@pytest.mark.parametrize("key", sorted(SETS))
def test_result_is_a_local_optimum_and_acyclic(key):
    file, scorer = SETS[key]
    df = pd.read_csv(FIX / file)
    out = sp.hill_climb(df, data_type=key)
    dag = out["dag"].to_numpy()
    d = dag.shape[0]
    score = scorer(df)
    parents = [set(np.flatnonzero(dag[:, b]).tolist()) for b in range(d)]
    base = [score(b, frozenset(parents[b])) for b in range(d)]
    assert sum(base) == pytest.approx(out["score"], rel=1e-12)
    for a in range(d):
        for b in range(d):
            if a == b:
                continue
            if a in parents[b]:
                assert score(b, frozenset(parents[b] - {a})) <= base[b] + 1e-8
            elif b not in parents[a] and not _creates_cycle(parents, a, b):
                assert score(b, frozenset(parents[b] | {a})) <= base[b] + 1e-8
    # acyclic: a topological order exists
    remaining = set(range(d))
    while remaining:
        roots = [v for v in remaining if not (parents[v] & remaining)]
        assert roots
        remaining -= set(roots)


def test_collider_is_oriented_and_a_chain_is_not():
    rng = np.random.default_rng(0)
    n = 3000
    a, b = rng.normal(size=(2, n))
    c = a + b + 0.5 * rng.normal(size=n)
    out = sp.hill_climb(pd.DataFrame({"a": a, "b": b, "c": c}))
    assert sorted(out["edges"]) == [("a", "c"), ("b", "c")]
    assert out["cpdag"].loc["c", "a"] == 0  # the collider fixes the direction
    x = rng.normal(size=n)
    m = x + rng.normal(size=n)
    y = m + rng.normal(size=n)
    chain = sp.hill_climb(pd.DataFrame({"x": x, "m": m, "y": y}))
    g = chain["cpdag"]
    assert g.loc["x", "m"] == 1 and g.loc["m", "x"] == 1  # direction not identified
    assert g.loc["x", "y"] == 0 and g.loc["y", "x"] == 0


def test_cpdag_of_a_dag():
    # a -> c <- b, c -> d : everything is oriented (R1 after the collider)
    dag = np.zeros((4, 4), dtype=int)
    dag[0, 2] = dag[1, 2] = dag[2, 3] = 1
    np.testing.assert_array_equal(_dag_to_cpdag(dag), dag)
    # a -> b -> c : nothing is
    chain = np.zeros((3, 3), dtype=int)
    chain[0, 1] = chain[1, 2] = 1
    np.testing.assert_array_equal(_dag_to_cpdag(chain), chain + chain.T)


def test_background_knowledge_and_parent_cap():
    df = pd.read_csv(FIX / "hill_climb_gaussian.csv")
    free = sp.hill_climb(df)
    assert ("x3", "x4") in free["edges"] or ("x4", "x3") in free["edges"]
    out = sp.hill_climb(df, forbidden=[("x3", "x4"), ("x4", "x3")])
    assert ("x3", "x4") not in out["edges"] and ("x4", "x3") not in out["edges"]
    assert out["score"] < free["score"]
    forced = sp.hill_climb(df, required=[("x6", "x1")])
    assert ("x6", "x1") in forced["edges"]
    capped = sp.hill_climb(df, max_parents=1)
    assert capped["dag"].sum(axis=0).max() <= 1
    again = sp.hill_climb(df, restarts=5, seed=3)
    assert again["score"] >= free["score"] - 1e-9


def test_data_type_is_not_guessed_for_mixed_columns():
    df = pd.read_csv(FIX / "hill_climb_discrete.csv")
    assert sp.hill_climb(df)["data_type"] == "discrete"
    mixed = df.assign(z=np.arange(len(df), dtype=float))
    with pytest.raises(MethodIncompatibility, match="mix categorical and numeric"):
        sp.hill_climb(mixed)
    with pytest.raises(MethodIncompatibility, match="needs numeric"):
        sp.hill_climb(df, data_type="gaussian")
    with pytest.raises(MethodIncompatibility, match="both required and forbidden"):
        sp.hill_climb(df, required=[("a", "c")], forbidden=[("a", "c")])


def test_bootstrap_edges_separates_real_edges_from_noise():
    rng = np.random.default_rng(1)
    n = 500
    a, b, noise = rng.normal(size=(3, n))
    c = a + b + 0.5 * rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "noise": noise})
    out = sp.bootstrap_edges(df, "pc", n_boot=60, seed=2)
    assert out.attrs["n_boot"] == 60 and out.attrs["method"] == "pc"
    strength = {(r["from"], r["to"]): r for _, r in out.iterrows()}
    assert strength[("a", "c")]["strength"] > 0.95
    assert strength[("a", "c")]["direction"] > 0.9  # the collider orients it
    assert strength[("c", "a")]["direction"] < 0.1
    spurious = out[(out["from"] == "noise") | (out["to"] == "noise")]
    assert (spurious["strength"] < 0.3).all()
    assert not spurious["stable"].any()
    # direction shares of a pair add to one
    assert strength[("a", "c")]["direction"] + strength[("c", "a")]["direction"] == (
        pytest.approx(1.0)
    )


def test_bootstrap_edges_with_other_algorithms_and_a_callable():
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    m = x + rng.normal(size=n)
    y = m + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "m": m, "y": y})
    hc = sp.bootstrap_edges(df, "hill_climb", n_boot=30, seed=1)
    row = hc[(hc["from"] == "x") & (hc["to"] == "m")].iloc[0]
    # a chain's direction is not identified: about half each way
    assert row["strength"] > 0.95 and row["direction"] == pytest.approx(0.5)
    pag = sp.bootstrap_edges(df, "fci", n_boot=20, seed=1)
    assert set(pag.columns) == {"from", "to", "strength", "direction", "stable"}
    same = sp.bootstrap_edges(df, sp.pc_algorithm, n_boot=20, seed=5, alpha=0.01)
    named = sp.bootstrap_edges(df, "pc", n_boot=20, seed=5, alpha=0.01)
    pd.testing.assert_frame_equal(same, named)
    with pytest.raises(MethodIncompatibility, match="unknown method"):
        sp.bootstrap_edges(df, "tabu")
