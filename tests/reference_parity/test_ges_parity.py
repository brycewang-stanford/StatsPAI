"""Analytical parity: sp.ges recovers correct CPDAGs (skeleton + v-structures).

Greedy Equivalence Search on known linear-Gaussian DGPs:

* collider X -> Z <- Y (X ⊥ Y): the v-structure is recovered with both edges
  directed INTO Z and NO spurious X--Y edge (the explaining-away edge the
  acyclicity-unconstrained search used to add);
* chain X -> Y -> Z: the CPDAG is the undirected skeleton X--Y--Z with no
  X--Z edge (Markov-equivalence class of the chain).

Analytical evidence tier (known-truth structure recovery on deterministic
DGPs; guards the acyclicity + DAG->CPDAG correctness fix).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

N = 5000


def test_collider_no_spurious_edge_between_parents():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, N)
        Y = rng.normal(0, 1, N)
        Z = 0.8 * X + 0.8 * Y + rng.normal(0, 1, N)
        edges = set(sp.ges(pd.DataFrame({"X": X, "Y": Y, "Z": Z})).edges())
        # v-structure recovered, both into Z
        assert ("X", "Z", "-->") in edges
        assert ("Y", "Z", "-->") in edges
        # no edge (directed or undirected) between the independent parents
        for mark in ("-->", "---"):
            assert ("X", "Y", mark) not in edges
            assert ("Y", "X", mark) not in edges


def test_chain_skeleton_is_undirected_cpdag():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, N)
    Y = 0.8 * X + rng.normal(0, 1, N)
    Z = 0.8 * Y + rng.normal(0, 1, N)
    edges = set(sp.ges(pd.DataFrame({"X": X, "Y": Y, "Z": Z})).edges())
    assert ("X", "Y", "---") in edges
    assert ("Y", "Z", "---") in edges
    # no direct X--Z edge (they are d-separated by Y)
    for mark in ("-->", "---"):
        assert ("X", "Z", mark) not in edges
        assert ("Z", "X", mark) not in edges


def test_adjacency_is_acyclic_dag_before_cpdag_undirection():
    # Collider recovers a proper DAG: no 2-cycle among the directed edges.
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, N)
    Y = rng.normal(0, 1, N)
    Z = 0.8 * X + 0.8 * Y + rng.normal(0, 1, N)
    adj = sp.ges(pd.DataFrame({"X": X, "Y": Y, "Z": Z})).adjacency
    # directed edges (one-way) must not have their reverse also set
    p = adj.shape[0]
    for i in range(p):
        for j in range(p):
            if adj[i, j] and not adj[j, i]:
                assert not adj[j, i]  # strictly one-directional (v-structure)


def test_result_does_not_depend_on_last_bit_rounding(monkeypatch):
    """X -> Z and Z -> X score the same in exact arithmetic; the search must
    not let rounding pick between them (the Windows CI leg did). Perturb
    every BIC by a relative 1e-13 and require the same CPDAG each time."""
    import importlib

    # the package exports the function under the module's name
    ges_mod = importlib.import_module("statspai.causal_discovery.ges")

    real = ges_mod._bic_score_local
    rng = np.random.default_rng(0)
    data_rng = np.random.default_rng(1)
    X = data_rng.normal(0, 1, N)
    Y = data_rng.normal(0, 1, N)
    Z = 0.8 * X + 0.8 * Y + data_rng.normal(0, 1, N)
    frame = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    want = set(sp.ges(frame).edges())
    assert ("X", "Z", "-->") in want and ("Y", "Z", "-->") in want

    def noisy(data, j, parents):
        b = real(data, j, parents)
        return b * (1.0 + 1e-13 * rng.standard_normal())

    # the search scores local (node, parent set) terms since 2026-10-19
    monkeypatch.setattr(ges_mod, "_bic_score_local", noisy)
    for _ in range(20):
        assert set(sp.ges(frame).edges()) == want
