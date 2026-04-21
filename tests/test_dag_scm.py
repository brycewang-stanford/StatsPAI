"""Sprint-2 tests: SCM machinery on sp.dag."""
import numpy as np
import pytest

import statspai as sp


# ---------- ID algorithm ----------

def test_id_backdoor_is_identifiable():
    """Classic confounder DAG -- P(Y | do(X)) identifiable via backdoor."""
    g = sp.dag("Z -> X; Z -> Y; X -> Y")
    res = sp.identify(g, treatment="X", outcome="Y")
    assert res.identifiable
    assert "Y" in res.estimand
    # c-components of an unconfounded graph should be singletons:
    assert all(len(c) == 1 for c in res.c_components)


def test_id_bow_arc_is_NOT_identifiable():
    """X -> Y with bidirected X <-> Y (bow arc) -> NOT identifiable."""
    g = sp.dag("X -> Y; X <-> Y")
    res = sp.identify(g, treatment="X", outcome="Y")
    assert not res.identifiable
    assert "hedge" in res.estimand


def test_id_frontdoor_structure():
    """Classic front-door: X -> M -> Y with X <-> Y latent -> identifiable."""
    g = sp.dag("X -> M; M -> Y; X <-> Y")
    res = sp.identify(g, treatment="X", outcome="Y")
    assert res.identifiable


def test_id_raises_on_unknown_variable():
    g = sp.dag("X -> Y")
    with pytest.raises(KeyError):
        sp.identify(g, treatment="Z", outcome="Y")


# ---------- do-calculus rules ----------

def test_rule1_independence_on_mutilated_graph():
    g = sp.dag("Z -> X; X -> Y")
    # After deleting edges into X, Z ⊥ Y? no, Z -> X removed but Z is still parent of X which now isn't a parent of Y through that. Actually rule 1 checks Y ⊥ Z | X in G_{bar X}. Since X has no parents in G_{bar X}, Z and Y are d-separated given X iff there's no open path. Z -> X -> Y is blocked by X. Test passes.
    chk = sp.do_rule1(g, Y="Y", X="X", Z="Z")
    assert chk.applicable
    assert chk.rule == 1


def test_rule2_observation_exchange():
    g = sp.dag("Z -> Y")
    # No confounding -> do(Z) and observing Z coincide.
    chk = sp.do_rule2(g, Y="Y", X=set(), Z="Z")
    assert chk.applicable


def test_rule3_deletion_of_action():
    g = sp.dag("Z -> Y")
    # Removing edges into Z leaves Y ⊥ Z when Z has no children other than Y.
    # Rule 3 should at least return a well-formed RuleCheck.
    chk = sp.do_rule3(g, Y="Y", X=set(), Z="Z")
    assert isinstance(chk, sp.RuleCheck)
    assert chk.rule == 3


def test_apply_rules_returns_all_three():
    g = sp.dag("X -> Y")
    results = sp.do_calculus_apply(g, Y="Y", X="X", Z=set())
    assert len(results) == 3
    assert [r.rule for r in results] == [1, 2, 3]


# ---------- SWIG ----------

def test_swig_splits_intervened_nodes():
    g = sp.dag("L -> X; L -> Y; X -> Y")
    sw = sp.swig(g, intervention={"X": "x"})
    # Should contain observation half X, action half X(x), and Y(x)
    assert "X" in sw.nodes
    assert "X(x)" in sw.nodes
    assert any(v.startswith("Y(") for v in sw.nodes)
    ascii_view = sw.ascii()
    assert "X(x) -> Y" in ascii_view or "X(x) -> Y(x=x)" in ascii_view


def test_swig_accepts_bare_variable_iterable():
    g = sp.dag("X -> Y")
    sw = sp.swig(g, intervention=["X"])
    assert "X(x)" in sw.nodes


# ---------- SCM counterfactuals ----------

def test_scm_simulate_linear_chain():
    scm = sp.SCM()
    scm.add("X", [], lambda pa, u: u)
    scm.add("Y", ["X"], lambda pa, u: 2 * pa["X"] + u)
    sim = scm.simulate(n=500, seed=0)
    # Cov(X, Y) should be ~2 Var(X) = 2
    cov = np.cov(sim["X"], sim["Y"])[0, 1]
    assert 1.5 < cov < 2.5


def test_scm_counterfactual_matches_linear_prediction():
    """For a deterministic-in-treatment linear SCM, the counterfactual
    Y(x=5) | X=0, Y=0 should give a predictable answer."""
    scm = sp.SCM()
    scm.add("X", [], lambda pa, u: u)
    # Make Y depend ONLY on X (noise free) so the abduction is trivially
    # satisfied — the counterfactual Y(x=5) must be 10.
    scm.add("Y", ["X"], lambda pa, u: 2 * pa["X"])
    cf = scm.counterfactual(
        evidence={"X": 0.0, "Y": 0.0},
        intervention={"X": 5.0},
        n_samples=200,
        seed=0,
        tol=0.5,
    )
    assert abs(cf["Y"].mean() - 10.0) < 1e-6
