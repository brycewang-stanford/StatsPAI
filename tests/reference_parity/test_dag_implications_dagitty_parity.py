"""``sp.dag``: testable implications and named latents, against dagitty 0.3.4.

Three things the DAG chapters of *Applied Causal Inference Powered by ML
and AI* do with ``dagitty`` / ``pgmpy`` that ``sp.dag`` could not:

* list the conditional independencies a graph implies
  (``impliedConditionalIndependencies``);
* test them on data (``localTests(type = "cis")``);
* declare a *named* node unobserved, so that it is not offered as an
  adjustment variable and identification treats it as unmeasured.

The set comparisons are exact. The partial correlations and Fisher-z
p-values are compared at 1e-10 relative: both sides compute them from the
same 2,000 simulated rows.

Fixture: ``_generate_dag_implications.R``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def reference() -> dict:
    path = FIX / "dag_implications_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graph(reference):
    return sp.dag(reference["spec"])


def _as_sets(sets) -> set:
    """dagitty's set lists arrive from jsonlite as a dict or as a list."""
    values = sets.values() if isinstance(sets, dict) else sets
    return {frozenset(v) for v in values}


def test_adjustment_sets_match_dagitty(graph, reference):
    minimal = {frozenset(s) for s in graph.adjustment_sets("D", "Y")}
    assert minimal == _as_sets(reference["minimal_adjustment"])
    every = {frozenset(s) for s in graph.adjustment_sets("D", "Y", minimal=False)}
    assert every == _as_sets(reference["all_adjustment"])


def test_implied_independencies_match_dagitty(graph, reference):
    ours = {(a, b, frozenset(z)) for a, b, z in graph.implied_independencies()}
    theirs = {(t["x"], t["y"], frozenset(t["given"])) for t in reference["tests"]}
    assert len(ours) == len(theirs) == 43
    assert ours == theirs


def test_local_tests_match_dagitty(graph, reference):
    data = pd.read_csv(FIX / "dag_implications.csv")
    out = graph.test_implications(data)
    ours = {(r.x, r.y, r.given): (r.partial_corr, r.p_value) for r in out.itertuples()}
    for t in reference["tests"]:
        est, p = ours[(t["x"], t["y"], ", ".join(t["given"]))]
        assert est == pytest.approx(t["estimate"], rel=1e-10, abs=1e-13)
        assert p == pytest.approx(t["p_value"], rel=1e-10)
    assert (out["p_holm"] >= out["p_value"]).all()


def test_a_wrong_graph_is_rejected(graph):
    """Data from a graph with an extra Y -> D edge fails the implications."""
    data = pd.read_csv(FIX / "dag_implications.csv")
    assert (graph.test_implications(data)["p_holm"] < 0.05).sum() == 0
    wrong = data.assign(D=(data["D"] + data["Y"]) / 2)
    assert (graph.test_implications(wrong)["p_holm"] < 0.05).sum() >= 10


def test_named_latents_match_dagitty(reference):
    for case in reference["latent_cases"]:
        g = sp.dag(case["spec"], latent=case["latent"])
        ours = {frozenset(s) for s in g.adjustment_sets("D", "Y")}
        assert ours == _as_sets(case["minimal"]), case["spec"]
        assert len(g.implied_independencies()) == case["n_implied"]
        res = sp.identify(g, "D", "Y")
        # In these graphs the effect is identified exactly when a
        # back-door set of observed variables exists.
        assert bool(res.identifiable) == bool(ours)


def test_latent_declaration_spellings_agree():
    spec = "D -> Y; F -> D; F -> Y"
    a = sp.dag(spec, latent=["F"])
    b = sp.dag(spec + "; F [latent]")
    c = sp.dag(spec).set_latent("F")
    for g in (a, b, c):
        assert g.latent_nodes == {"F"}
        assert g.adjustment_sets("D", "Y") == []
        assert not sp.identify(g, "D", "Y").identifiable
    assert sp.dag(spec).adjustment_sets("D", "Y") == [{"F"}]
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not in the graph"):
        sp.dag(spec, latent=["Q"])
    data = pd.DataFrame({"D": [0.0, 1.0, 2.0, 3.0], "F": [1.0, 0.0, 1.0, 2.0]})
    with pytest.raises(sp.exceptions.ColumnNotFound, match="Y"):
        sp.dag("D -> M -> Y; F -> D").test_implications(data)


def test_front_door_through_an_observed_mediator():
    """Only (Y, D, M) observed: the front-door formula, as dosearch finds."""
    spec = (
        "z1 -> x1; z1 -> x2; z2 -> x2; z2 -> x3; x2 -> d; x2 -> y; "
        "x3 -> y; x1 -> d; d -> m; m -> y"
    )
    g = sp.dag(spec, latent=["z1", "z2", "x1", "x2", "x3"])
    res = sp.identify(g, "d", "y")
    assert res.identifiable
    assert "P(m | d)" in res.estimand and "P(y | d', m)" in res.estimand
    assert g.frontdoor_sets("d", "y") == [{"m"}]


def test_latent_projection_keeps_directed_paths_through_latents():
    g = sp.dag("X -> F; F -> D; F -> M; D -> Y", latent=["F"])
    p = g.latent_projection()
    directed = {e for e in p.edges if not e[0].startswith("_L_")}
    assert directed == {("X", "D"), ("X", "M"), ("D", "Y")}
    assert p.latent_nodes == {"_L_D_M"}
    assert p.observed_nodes == {"X", "D", "M", "Y"}


def test_identification_formula_is_numerically_right_with_a_named_latent():
    """Back-door through X with latent F -> X, F -> D: check the number.

    Linear SEM, so E[Y | do(D = d)] has slope equal to the structural
    coefficient; adjusting for X recovers it and omitting X does not.
    """
    rng = np.random.default_rng(3)
    n = 40_000
    F = rng.normal(size=n)
    X = F + rng.normal(size=n)
    D = X + F + rng.normal(size=n)
    Y = 2.0 * D + 1.5 * X + rng.normal(size=n)
    g = sp.dag("D -> Y; X -> D; X -> Y; F -> X; F -> D", latent=["F"])
    (adj,) = g.adjustment_sets("D", "Y")
    assert adj == {"X"}
    Z = np.column_stack([np.ones(n), D, X])
    slope = np.linalg.lstsq(Z, Y, rcond=None)[0][1]
    assert slope == pytest.approx(2.0, abs=0.02)
    naive = np.linalg.lstsq(Z[:, :2], Y, rcond=None)[0][1]
    assert abs(naive - 2.0) > 0.3
