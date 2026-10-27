"""Option scan, round 6: ``sp.network`` on directed and weighted graphs.

Reference values are R ``igraph`` 2.x (and, where noted, ``networkx``) on
the graphs built by ``_graphs()``; the R calls are given next to each
number. ERGM references are R ``ergm`` with ``estimate = "MPLE"``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _graphs():
    rng = np.random.default_rng(7)
    n = 12
    A = (rng.uniform(size=(n, n)) < 0.25).astype(float)
    np.fill_diagonal(A, 0)
    A[10, :] = 0
    A[:, 10] = 0  # an isolate
    A[11, :] = 0
    A[:, 11] = 0
    A[11, 0] = 1  # one out-tie only
    W = A * rng.integers(1, 5, size=(n, n))
    return A, W


MEMBERSHIP = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2]


def test_eigenvector_directed_in_mode_matches_igraph():
    # eigen_centrality(g, directed = TRUE), rescaled to unit L2 norm;
    # networkx.eigenvector_centrality gives the same vector.
    A, _ = _graphs()
    ref = [
        2.9820061667104825e-01,
        1.4988960110969438e-01,
        3.3560271925506774e-01,
        3.5954676536713998e-01,
        1.1644966146118041e-01,
        1.8455580216101325e-01,
        4.2532078443406968e-01,
        4.4270210345124755e-01,
        3.7970418903469910e-01,
        2.7933291433489532e-01,
        0.0,
        0.0,
    ]
    ev = sp.eigenvector_centrality(sp.network_graph(A, directed=True), mode="in")
    assert ev.to_numpy() == pytest.approx(ref, abs=1e-12)
    # mode='out' (the default) is sna::evcent's orientation: the in-mode
    # vector of the transposed graph
    out = sp.eigenvector_centrality(sp.network_graph(A, directed=True))
    flipped = sp.eigenvector_centrality(sp.network_graph(A.T, directed=True), mode="in")
    assert out.to_numpy() == pytest.approx(flipped.to_numpy(), abs=1e-12)
    with pytest.raises(ValueError):
        sp.eigenvector_centrality(sp.network_graph(A, directed=True), mode="all")


def test_transitivity_directed_collapses_to_undirected():
    # transitivity(g, type = "global") and type = "local", isolates = "zero"
    A, _ = _graphs()
    g = sp.network_graph(A, directed=True)
    assert sp.transitivity(g) == pytest.approx(0.34782608695652173, abs=1e-14)
    local = [0.4, 1 / 3, 0.5, 1 / 3, 0, 0, 0.5, 1 / 6, 0.5, 0.1, 0, 0]
    assert sp.clustering(g).to_numpy() == pytest.approx(local, abs=1e-14)


def test_assortativity_directed_is_out_in():
    # assortativity_degree(g, directed = TRUE); networkx
    # degree_assortativity_coefficient gives 0.28764698611864725
    A, _ = _graphs()
    val = sp.assortativity(sp.network_graph(A, directed=True))
    assert val == pytest.approx(0.28764698611864808, abs=1e-12)


def test_hits_uses_tie_weights():
    # hits_scores(gw) rescaled to sum one; networkx.hits agrees
    _, W = _graphs()
    hub = [
        7.5509972081912916e-02,
        7.8362018364596150e-02,
        1.1464018128578891e-01,
        0.0,
        1.1262873599691463e-01,
        2.4275660416020201e-01,
        2.5214392750194936e-01,
        5.5714765709392634e-03,
        9.4497961700850910e-02,
        4.0928162364940925e-03,
        0.0,
        1.9796306100351695e-02,
    ]
    auth = [
        0.0539849175148830115,
        0.0,
        0.1332025054566504108,
        0.1509011641392654091,
        0.0017299261709852478,
        0.0015699447640573120,
        0.0141468967472338503,
        0.3071407863556399054,
        0.1854436318563427166,
        0.1518802269949421913,
        0.0,
        0.0,
    ]
    out = sp.hits(sp.network_graph(W, directed=True))
    assert out["hub"].to_numpy() == pytest.approx(hub, abs=1e-10)
    assert out["authority"].to_numpy() == pytest.approx(auth, abs=1e-10)


def test_hits_unweighted_option_matches_binary_graph():
    A, W = _graphs()
    a = sp.hits(sp.network_graph(W, directed=True), weighted=False)
    b = sp.hits(sp.network_graph(A, directed=True))
    pd.testing.assert_frame_equal(a, b)


def test_modularity_directed_is_leicht_newman():
    # modularity(g, m, directed = TRUE)
    A, W = _graphs()
    q = sp.network_modularity(sp.network_graph(A, directed=True), MEMBERSHIP)
    assert q == pytest.approx(-0.064236111111111077, abs=1e-14)
    qw = sp.network_modularity(sp.network_graph(W, directed=True), MEMBERSHIP)
    assert qw == pytest.approx(-0.044444444444444453, abs=1e-14)


def test_modularity_undirected_unchanged():
    # undirected Q of the karate-club factional split, Newman (2006)
    g = sp.karate_club()
    faction = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    faction += [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert round(sp.network_modularity(g, faction), 4) == 0.3715


# ------------------------------------------------------------------ netlm
def test_netlm_dsp_intercept_p_value_is_not_the_floor():
    # Under double-semi-partialling the intercept was never permuted, so its
    # p-value was 1 / (nperm + 1) whatever the data. With a true intercept
    # of zero it must not be significant.
    rng = np.random.default_rng(11)
    n = 20
    X1 = (rng.uniform(size=(n, n)) < 0.3).astype(float)
    X2 = rng.normal(size=(n, n))
    Y = 0.4 * X2 + rng.normal(size=(n, n))
    for M in (X1, X2, Y):
        np.fill_diagonal(M, 0)
    res = sp.netlm(Y, [X1, X2], nperm=500, seed=3)
    assert res.p_qap["(Intercept)"] > 0.05
    assert res.p_qap["x2"] == pytest.approx(1 / 501)


# ------------------------------------------------------------------- ergm
def _ergm_data():
    rng = np.random.default_rng(5)
    n = 30
    grp = rng.integers(0, 3, size=n)
    x = rng.normal(size=n)
    P = (
        0.08
        + 0.15 * (grp[:, None] == grp[None, :])
        + 0.05 * (grp[:, None] + grp[None, :])
    )
    A = (rng.uniform(size=(n, n)) < P).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    return A, pd.DataFrame({"grp": grp, "x": x})


def test_ergm_nodefactor_has_one_term_per_level():
    # ergm(nw ~ edges + nodefactor("grp"), estimate = "MPLE")
    A, att = _ergm_data()
    res = sp.ergm(
        sp.network_graph(A), terms=["edges", "nodefactor:grp"], node_attrs=att
    )
    coefs = res.coefficients.set_index("term")["estimate"]
    assert list(coefs.index) == ["edges", "nodefactor.grp.1", "nodefactor.grp.2"]
    assert coefs.to_numpy() == pytest.approx(
        [-1.6364107, 0.2262668, 0.3032145], abs=5e-7
    )
    se = res.coefficients["se"].to_numpy()
    assert se == pytest.approx([0.2303100, 0.2078807, 0.2057325], abs=5e-7)


def test_ergm_attribute_frame_with_default_index_and_string_labels():
    # ergm(nw ~ edges + nodematch("grp") + nodecov("x"), estimate = "MPLE")
    A, att = _ergm_data()
    labels = [f"n{i}" for i in range(A.shape[0])]
    g = sp.network_graph(A, node_labels=labels)
    res = sp.ergm(g, terms=["edges", "nodematch:grp", "nodecov:x"], node_attrs=att)
    assert res.coefficients["estimate"].to_numpy() == pytest.approx(
        [-1.63733713, 0.70064253, -0.02681961], abs=5e-8
    )


def test_ergm_attribute_frame_indexed_by_label_is_aligned():
    A, att = _ergm_data()
    labels = [f"n{i}" for i in range(A.shape[0])]
    g = sp.network_graph(A, node_labels=labels)
    shuffled = att.set_axis(labels).iloc[::-1]
    a = sp.ergm(g, terms=["edges", "nodematch:grp"], node_attrs=shuffled)
    b = sp.ergm(sp.network_graph(A), terms=["edges", "nodematch:grp"], node_attrs=att)
    assert a.coefficients["estimate"].to_numpy() == pytest.approx(
        b.coefficients["estimate"].to_numpy(), abs=1e-12
    )


def test_ergm_attribute_frame_with_unknown_labels_raises():
    A, att = _ergm_data()
    g = sp.network_graph(A, node_labels=[f"n{i}" for i in range(A.shape[0])])
    bad = att.set_axis([f"m{i}" for i in range(A.shape[0])])
    with pytest.raises(ValueError):
        sp.ergm(g, terms=["edges", "nodematch:grp"], node_attrs=bad)
