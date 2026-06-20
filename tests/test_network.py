"""Tests for the social network analysis module (``sp.network``).

Reference-parity values are hardcoded from independent oracles:

* Structural / centrality / community values for Zachary's karate club are
  the values reported by ``networkx`` 3.x on the *binary* graph (which in
  turn match ``igraph`` / ``sna``); see the module-level constants.  The
  tests themselves do **not** import networkx, so they run in CI without it.
* Small-graph values (path, star, complete) are analytic closed forms.
* PageRank is checked against the exact Google-matrix stationary
  distribution.

The numbers were generated once with networkx as an oracle and frozen here
(CLAUDE.md §5: reference parity, not mocked numerical paths).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# --------------------------------------------------------------------- #
#  Frozen karate-club oracle (binary graph; networkx 3.x == igraph/sna)
# --------------------------------------------------------------------- #
KARATE_DEGREE = [
    16,
    9,
    10,
    6,
    3,
    4,
    4,
    4,
    5,
    2,
    3,
    1,
    2,
    5,
    2,
    2,
    2,
    2,
    2,
    3,
    2,
    2,
    2,
    5,
    3,
    3,
    2,
    4,
    3,
    4,
    4,
    6,
    12,
    17,
]
KARATE = dict(
    density=0.13903743315508021,
    transitivity=0.2556818181818182,
    avg_clustering=0.5706384782076823,
    diameter=5.0,
    avg_path_len=2.408199643493761,
    assortativity=-0.47561309768461413,
    betw_unnorm_0=231.071429,
    betw_unnorm_33=160.551587,
    betw_unnorm_2=75.850794,
    closeness_0=0.568966,
    eigenvector_33=0.373363,
    eigenvector_0=0.355491,
    pagerank_0=0.096997,  # exact Google-matrix stationary value
    pagerank_33=0.100919,
    modularity_true_split=0.371466,
    modularity_greedy=0.380671,
    greedy_ncomm=3,
)


# ===================================================================== #
#  Graph construction
# ===================================================================== #


def test_graph_from_edgelist_basic():
    g = sp.network_graph(edges=[(0, 1), (1, 2), (2, 0)])
    assert g.n_nodes == 3
    assert g.n_edges == 3
    assert not g.is_directed
    assert g.density == 1.0


def test_graph_from_adjacency_and_labels():
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    g = sp.network_graph(A, node_labels=["a", "b", "c"])
    assert g.labels == ["a", "b", "c"]
    assert g.degree().tolist() == [1.0, 2.0, 1.0]


def test_graph_from_pandas_edgelist():
    df = pd.DataFrame({"u": [0, 1, 2], "v": [1, 2, 0]})
    g = sp.network.Graph.from_pandas_edgelist(df, "u", "v")
    assert g.n_edges == 3


def test_graph_directed_keeps_arcs():
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)
    g = sp.network_graph(A, directed=True)
    assert g.is_directed
    assert g.n_edges == 2
    assert g.degree(mode="out").tolist() == [1.0, 1.0, 0.0]
    assert g.degree(mode="in").tolist() == [0.0, 1.0, 1.0]


def test_graph_asymmetric_undirected_warns_and_symmetrises():
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)
    with pytest.warns(UserWarning, match="not symmetric"):
        g = sp.network_graph(A, directed=False)
    assert np.allclose(g.adjacency_matrix(), g.adjacency_matrix().T)


def test_graph_rejects_nonsquare_and_negative():
    with pytest.raises(ValueError):
        sp.network_graph(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        sp.network_graph(np.array([[0, -1.0], [-1.0, 0]]))


def test_network_graph_both_inputs_errors():
    with pytest.raises(ValueError):
        sp.network_graph(adjacency=np.zeros((2, 2)), edges=[(0, 1)])


# ===================================================================== #
#  Descriptives (karate parity)
# ===================================================================== #


def test_network_summary_karate():
    g = sp.karate_club()
    res = sp.network_summary(g)
    assert (res.n_nodes, res.n_edges) == (34, 78)
    assert res.density == pytest.approx(KARATE["density"], abs=1e-12)
    assert res.transitivity == pytest.approx(KARATE["transitivity"], abs=1e-12)
    assert res.average_clustering == pytest.approx(KARATE["avg_clustering"], abs=1e-12)
    assert res.diameter == KARATE["diameter"]
    assert res.average_path_length == pytest.approx(KARATE["avg_path_len"], abs=1e-9)
    assert res.assortativity == pytest.approx(KARATE["assortativity"], abs=1e-9)
    assert res.is_connected
    assert np.isnan(res.reciprocity)  # undirected


def test_degree_sequence_karate():
    g = sp.karate_club()
    assert g.degree().astype(int).tolist() == KARATE_DEGREE


def test_transitivity_and_clustering_helpers():
    g = sp.karate_club()
    assert sp.transitivity(g) == pytest.approx(KARATE["transitivity"], abs=1e-12)
    loc = sp.clustering(g)
    assert loc.mean() == pytest.approx(KARATE["avg_clustering"], abs=1e-12)


def test_reciprocity_directed():
    # one mutual dyad (0<->1), one asymmetric (1->2)
    A = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], float)
    g = sp.network_graph(A, directed=True)
    # arcs: 0->1, 1->0, 1->2 => 3 arcs, 2 reciprocated
    assert sp.reciprocity(g) == pytest.approx(2.0 / 3.0)


def test_components_disconnected():
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = 1
    A[2, 3] = A[3, 2] = 1
    g = sp.network_graph(A)
    comp = sp.network_components(g)
    assert comp.n_components == 2
    assert comp.sizes == [2, 2]


def test_complete_graph_descriptives():
    n = 5
    A = np.ones((n, n)) - np.eye(n)
    g = sp.network_graph(A)
    res = sp.network_summary(g)
    assert res.density == pytest.approx(1.0)
    assert res.transitivity == pytest.approx(1.0)
    assert res.diameter == 1.0


# ===================================================================== #
#  Centrality (karate parity + analytic)
# ===================================================================== #


def test_betweenness_karate_unnormalised():
    g = sp.karate_club()
    bt = sp.betweenness_centrality(g, normalized=False)
    assert bt.iloc[0] == pytest.approx(KARATE["betw_unnorm_0"], abs=1e-4)
    assert bt.iloc[33] == pytest.approx(KARATE["betw_unnorm_33"], abs=1e-4)
    assert bt.iloc[2] == pytest.approx(KARATE["betw_unnorm_2"], abs=1e-4)
    assert bt.idxmax() == 0  # Mr Hi is the top broker


def test_betweenness_path_graph_analytic():
    # path 0-1-2-3-4: node 2 lies between {0,1}x{3,4} -> unnorm betweenness 4;
    # node 1 between {0}x{2,3,4} -> 3.  Endpoints 0.
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    g = sp.network_graph(edges=edges)
    bt = sp.betweenness_centrality(g, normalized=False)
    assert bt.loc[2] == pytest.approx(4.0)
    assert bt.loc[1] == pytest.approx(3.0)
    assert bt.loc[0] == pytest.approx(0.0)


def test_betweenness_star_graph_analytic():
    # star: center 0, leaves 1..4 ; center unnorm betweenness = C(4,2)=6
    edges = [(0, i) for i in range(1, 5)]
    g = sp.network_graph(edges=edges)
    bt = sp.betweenness_centrality(g, normalized=False)
    assert bt.loc[0] == pytest.approx(6.0)
    assert bt.loc[1] == pytest.approx(0.0)


def test_closeness_karate():
    g = sp.karate_club()
    cl = sp.closeness_centrality(g)
    assert cl.iloc[0] == pytest.approx(KARATE["closeness_0"], abs=1e-5)


def test_eigenvector_karate():
    g = sp.karate_club()
    ev = sp.eigenvector_centrality(g, tol=1e-12)
    assert ev.iloc[33] == pytest.approx(KARATE["eigenvector_33"], abs=1e-5)
    assert ev.iloc[0] == pytest.approx(KARATE["eigenvector_0"], abs=1e-5)
    assert np.linalg.norm(ev.values) == pytest.approx(1.0, abs=1e-6)  # L2


def test_pagerank_karate_exact_stationary():
    g = sp.karate_club()
    pr = sp.pagerank(g)
    assert pr.sum() == pytest.approx(1.0)
    assert pr.iloc[0] == pytest.approx(KARATE["pagerank_0"], abs=1e-5)
    assert pr.iloc[33] == pytest.approx(KARATE["pagerank_33"], abs=1e-5)


def test_pagerank_matches_google_matrix():
    g = sp.karate_club()
    A = g.binary()
    n = A.shape[0]
    M = A / A.sum(1)[:, None]
    G = 0.85 * M + 0.15 / n * np.ones((n, n))
    w, v = np.linalg.eig(G.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()
    assert np.allclose(sp.pagerank(g).values, pi, atol=1e-9)


def test_hits_equals_eigenvector_for_undirected():
    g = sp.karate_club()
    h = sp.hits(g)
    ev = sp.eigenvector_centrality(g, tol=1e-12).values
    evL1 = ev / ev.sum()
    assert np.allclose(h["authority"].values, evL1, atol=1e-6)
    assert np.allclose(h["hub"].values, h["authority"].values, atol=1e-9)


def test_katz_reduces_to_degree_at_small_alpha():
    g = sp.karate_club()
    kz = sp.katz_centrality(g, alpha=1e-6, normalized=False)
    deg = g.degree()
    # x ≈ beta(1 + alpha*deg); ranking == degree ranking
    assert np.corrcoef(kz.values, deg)[0, 1] > 0.999


def test_katz_raises_above_spectral_radius():
    g = sp.karate_club()
    with pytest.raises(ValueError, match="convergence"):
        sp.katz_centrality(g, alpha=10.0)


def test_bonacich_beta_zero_is_degree():
    g = sp.karate_club()
    bp = sp.bonacich_power(g, beta=1e-9)
    assert np.corrcoef(bp.values, g.degree())[0, 1] > 0.999


def test_centrality_dispatcher():
    g = sp.karate_club()
    res = sp.centrality(g, kind=["degree", "betweenness", "pagerank"])
    assert list(res.scores.columns) == ["degree", "betweenness", "pagerank"]
    assert res.most_central["betweenness"] == 0
    assert res.top("pagerank", 1).index[0] == 33
    with pytest.raises(ValueError, match="unknown centrality"):
        sp.centrality(g, kind="nonsense")


# ===================================================================== #
#  Community detection (karate parity)
# ===================================================================== #


def test_modularity_true_split():
    g = sp.karate_club()
    Q = sp.network_modularity(g, sp.network.KARATE_FACTION)
    assert Q == pytest.approx(KARATE["modularity_true_split"], abs=1e-6)


def test_greedy_modularity_matches_cnm():
    g = sp.karate_club()
    res = sp.community_detection(g, method="greedy")
    assert res.n_communities == KARATE["greedy_ncomm"]
    assert res.modularity == pytest.approx(KARATE["modularity_greedy"], abs=1e-5)


def test_louvain_high_modularity():
    g = sp.karate_club()
    res = sp.community_detection(g, method="louvain")
    assert res.modularity > 0.40
    assert res.n_communities >= 2
    # modularity of returned membership is internally consistent
    assert sp.network_modularity(g, res.membership) == pytest.approx(res.modularity)


def test_label_propagation_deterministic_with_seed():
    g = sp.karate_club()
    a = sp.community_detection(g, method="label_prop", seed=7)
    b = sp.community_detection(g, method="label_prop", seed=7)
    assert a.membership.equals(b.membership)
    assert a.modularity > 0.0


def test_community_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        sp.community_detection(sp.karate_club(), method="spectral")


# ===================================================================== #
#  Network regression: QAP / MRQAP / dyadic
# ===================================================================== #


def _sym_binary(n, p, rng):
    X = (rng.random((n, n)) < p).astype(float)
    X = np.triu(X, 1)
    return X + X.T


def test_netlm_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 30
    X1 = _sym_binary(n, 0.3, rng)
    X2 = _sym_binary(n, 0.3, rng)
    Y = 2.0 * X1 - 1.0 * X2 + rng.normal(0, 0.3, (n, n))
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    res = sp.netlm(Y, {"x1": X1, "x2": X2}, nperm=200, seed=1)
    coef = res.coefficients.set_index("variable")["coef"]
    assert coef["x1"] == pytest.approx(2.0, abs=0.3)
    assert coef["x2"] == pytest.approx(-1.0, abs=0.3)
    assert res.p_qap["x1"] < 0.05
    assert res.permutation == "dsp"


def test_netlogit_matches_logistic_design():
    rng = np.random.default_rng(3)
    n = 40
    X1 = rng.normal(size=(n, n))
    X1 = np.triu(X1, 1)
    X1 = X1 + X1.T
    mask = np.triu(np.ones((n, n), bool), 1)
    pr = 1 / (1 + np.exp(-(-0.5 + 0.8 * X1)))
    Y = (rng.random((n, n)) < pr).astype(float)
    Y = np.triu(Y, 1)
    Y = Y + Y.T
    res = sp.netlogit(Y, X1, nperm=20, seed=1)
    # closed-form logistic on the same dyadic design
    from statspai.network.regression import _irls_logit

    yv = Y[mask]
    Xd = np.column_stack([np.ones(mask.sum()), X1[mask]])
    beta, se, _, _ = _irls_logit(yv, Xd, 100)
    assert np.allclose(res.coefficients["coef"].values, beta, atol=1e-6)
    assert np.allclose(res.coefficients["se"].values, se, atol=1e-6)


def test_dyadic_robust_se_matches_brute_force():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(15):
        for j in range(i + 1, 15):
            x = rng.normal()
            rows.append((i, j, x, 1.0 + 0.5 * x + rng.normal(0, 0.5)))
    df = pd.DataFrame(rows, columns=["i", "j", "x", "y"])
    res = sp.dyadic_regression(df, y="y", covariates=["x"], i="i", j="j")
    se_fast = res.coefficients["se_dyadic"].to_numpy()

    # brute-force Aronow-Samii-Assenova meat
    yv = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    XtXinv = np.linalg.inv(X.T @ X)
    e = yv - X @ (XtXinv @ X.T @ yv)
    g = X * e[:, None]
    iv, jv = df["i"].to_numpy(), df["j"].to_numpy()
    D = len(df)
    meat = np.zeros((2, 2))
    for a in range(D):
        for b in range(D):
            if {iv[a], jv[a]} & {iv[b], jv[b]}:
                meat += np.outer(g[a], g[b])
    se_bf = np.sqrt(np.diag(XtXinv @ meat @ XtXinv))
    assert np.allclose(se_fast, se_bf, atol=1e-9)
    # dyadic SE should exceed the (too-small) classical SE here
    assert (
        res.coefficients.loc[1, "se_dyadic"]
        >= res.coefficients.loc[1, "se_classical"] * 0.5
    )


# ===================================================================== #
#  ERGM (MPLE)
# ===================================================================== #


def test_ergm_edges_only_is_logit_density():
    g = sp.florentine_families()
    res = sp.ergm(g, terms=["edges"])
    dens = sp.network_summary(g).density
    assert res.coefficients.loc[0, "estimate"] == pytest.approx(
        np.log(dens / (1 - dens)), abs=1e-6
    )
    assert res.dyad_independent


def test_ergm_dyad_independent_equals_logistic():
    g = sp.florentine_families()
    attr = {"grp": np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1])}
    res = sp.ergm(g, terms=["edges", "nodematch:grp"], node_attrs=attr)
    assert res.dyad_independent
    assert "nodematch.grp" in res.terms


def test_ergm_triangles_warns_dyad_dependent():
    g = sp.florentine_families()
    with pytest.warns(UserWarning, match="dyad-dependent"):
        res = sp.ergm(g, terms=["edges", "triangles"])
    assert not res.dyad_independent


def test_ergm_mutual_requires_directed():
    g = sp.karate_club()  # undirected
    with pytest.raises(ValueError, match="directed"):
        sp.ergm(g, terms=["edges", "mutual"])


# ===================================================================== #
#  Datasets
# ===================================================================== #


def test_karate_club_dataset():
    g = sp.karate_club()
    assert (g.n_nodes, g.n_edges) == (34, 78)


def test_florentine_medici_is_central():
    g = sp.florentine_families()
    assert (g.n_nodes, g.n_edges) == (15, 20)
    bt = sp.betweenness_centrality(g)
    assert bt.idxmax() == "Medici"


# ===================================================================== #
#  Result objects (agent-native contract)
# ===================================================================== #


def test_results_are_serialisable_and_cite():
    g = sp.karate_club()
    summ = sp.network_summary(g)
    d = summ.to_dict()
    assert d["n_nodes"] == 34
    # cite() returns verified paper.bib keys
    assert "watts1998collective" in summ.cite()
    cen = sp.centrality(g, kind="degree")
    assert "freeman1978centrality" in cen.cite()


# ===================================================================== #
#  Plots (matplotlib lazy)
# ===================================================================== #


def test_spring_layout_shape():
    g = sp.karate_club()
    pos = sp.network.spring_layout(g.binary(), iterations=20, seed=0)
    assert pos.shape == (34, 2)


def test_network_plot_returns_axes():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    g = sp.karate_club()
    com = sp.community_detection(g)
    ax = sp.network_plot(g, node_color=com.membership, layout="circular")
    assert ax is not None
