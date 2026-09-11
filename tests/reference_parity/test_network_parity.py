"""Cross-language parity: StatsPAI's network module vs R igraph / sna /
ergm / dyadRobust.

Regenerate with::

    python tests/reference_parity/_fixtures/_generate_network_data.py
    Rscript tests/reference_parity/_fixtures/_generate_network_R.R

Three evidence grades live in this file and are kept apart:

* **Exact** (rtol <= 1e-9): centrality scores, PageRank, transitivity,
  assortativity, reciprocity, modularity, components, Bonacich power, Katz,
  Wasserman-Faust closeness, QAP regression coefficients, ERGM MPLE
  coefficients, dyadic-robust standard errors, and the bundled datasets.
* **Documented convention**: eigenvector centrality and HITS. igraph
  max-scales both; StatsPAI follows networkx (L2 for eigenvector, L1 for
  HITS). The test asserts the ratio is constant across nodes, which a
  genuine difference in the scores could not satisfy.
* **T3**: Louvain. Every implementation randomises the node order, so a
  single run proves nothing either way; 200 seeded runs are compared as
  distributions. (A single-seed comparison against igraph looked like a
  1% shortfall and was not one.)

The sweep that produced this file found one defect and one reference bug:

* ``sp.dyadic_regression`` weighted each pair of dyads by the NUMBER of
  members they share. The Aronow-Samii-Assenova estimator weights by
  whether they share one. The two agree when every unordered pair appears
  once, and differ on directed data, where (i, j) and (j, i) share both
  members: 1.8% on the standard errors here.
* ``dyadRobust`` (the R implementation used as the reference) recodes node
  ids inside a single ``dplyr::mutate()``, whose sequential evaluation
  builds the alter codes from the already-recoded ego column. The fixture
  passes ids for which that recode is the identity and asserts the
  precondition; with other ids its output does not match its own formula.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def R():
    path = _FIX / "network_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_network_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


def _mat(name, delimiter=",", header=False):
    path = _FIX / name
    if header:
        return pd.read_csv(path).to_numpy(float)
    return np.loadtxt(path, delimiter=delimiter)


@pytest.fixture(scope="module")
def karate():
    return sp.karate_club()


@pytest.fixture(scope="module")
def directed():
    return sp.network_graph(_mat("network_directed.csv"), directed=True)


def _adj(g):
    A = g.adjacency_matrix
    A = A() if callable(A) else A
    return np.asarray(A.toarray() if hasattr(A, "toarray") else A, float)


def _close(got, ref, rtol=1e-10):
    got = np.asarray(got, float).ravel()
    ref = np.asarray(ref, float).ravel()
    assert got.shape == ref.shape
    assert np.allclose(got, ref, rtol=rtol, atol=0.0), np.max(
        np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
    )


# ── bundled datasets ──────────────────────────────────────────────────────


def test_karate_club_is_igraphs_zachary_graph(karate):
    ref = _mat("network_karate_igraph.csv", header=True)
    assert np.array_equal(_adj(karate), ref)


def test_florentine_is_flomarriage_without_the_pucci_isolate():
    ref = pd.read_csv(_FIX / "network_flomarriage.csv", index_col=0)
    g = sp.florentine_families()
    labels = list(g.labels)
    assert sorted(set(ref.index) - set(labels)) == ["Pucci"]
    assert int(ref.loc["Pucci"].sum()) == 0
    assert np.array_equal(_adj(g), ref.loc[labels, labels].to_numpy(float))


# ── centrality ────────────────────────────────────────────────────────────


def test_centrality_scores_match_igraph(R, karate):
    s = sp.centrality(karate, kind="all", normalized=True).scores
    _close(s["degree"], np.asarray(R["k_degree"]) / 33.0)
    _close(s["betweenness"], R["k_betweenness"])
    _close(s["closeness"], R["k_closeness"])
    _close(s["pagerank"], R["k_pagerank"])


def test_raw_betweenness_matches_igraph(R, karate, directed):
    raw = sp.centrality(karate, kind="betweenness", normalized=False).scores
    _close(raw["betweenness"], R["k_betweenness_raw"])
    d = sp.centrality(directed, kind="betweenness", normalized=False).scores
    _close(d["betweenness"], R["d_betweenness_raw"])


def test_eigenvector_is_igraph_up_to_normalisation(R, karate):
    """L2 here (networkx), max = 1 in igraph: same vector, one scalar apart."""
    e = sp.centrality(karate, kind="eigenvector").scores["eigenvector"].to_numpy()
    ref = np.asarray(R["k_eigen_maxscaled"])
    ratio = e / ref
    assert (ratio.max() - ratio.min()) / ratio.mean() < 1e-10
    assert np.linalg.norm(e) == pytest.approx(1.0, rel=1e-12)


def test_hits_is_igraph_up_to_normalisation(R, directed):
    """L1 here (documented in sp.hits), max = 1 in igraph."""
    h = sp.hits(directed)
    for col, key in (
        ("hub", "d_hub_maxscaled"),
        ("authority", "d_authority_maxscaled"),
    ):
        ours = h[col].to_numpy()
        ratio = ours / np.asarray(R[key])
        assert (ratio.max() - ratio.min()) / ratio.mean() < 1e-9
        assert ours.sum() == pytest.approx(1.0, rel=1e-12)


def test_pagerank_directed_matches_igraph(R, directed):
    _close(sp.pagerank(directed), R["d_pagerank"])


def test_bonacich_power_matches_igraph_and_sna(R, karate):
    got = sp.bonacich_power(karate, beta=0.1)
    _close(got, R["k_bonpow_igraph"])
    _close(got, R["k_bonpow_sna"])


def test_katz_matches_alpha_centrality(R, karate):
    _close(sp.katz_centrality(karate, alpha=0.1, normalized=False), R["k_alpha_01"])


def test_closeness_uses_the_wasserman_faust_correction(R):
    """On a disconnected graph -- the case the correction exists for."""
    g = sp.network_graph(_mat("network_disconnected.csv"), directed=False)
    got = np.asarray(sp.closeness_centrality(g), float)
    ref = np.asarray(R["disc_wf_closeness"], float)
    assert np.allclose(got, ref, rtol=1e-12, atol=0.0)
    assert np.all(got[ref == 0] == 0)  # isolates


# ── graph-level statistics ────────────────────────────────────────────────


def test_graph_statistics_match_igraph(R, karate, directed):
    assert sp.transitivity(karate) == pytest.approx(R["k_transitivity"], rel=1e-12)
    assert sp.assortativity(karate) == pytest.approx(R["k_assortativity"], rel=1e-12)
    assert sp.reciprocity(directed) == pytest.approx(R["d_reciprocity"], rel=1e-12)


def test_network_summary_matches_igraph(R, karate):
    s = sp.network_summary(karate)
    assert s.density == pytest.approx(R["k_density"], rel=1e-12)
    assert s.diameter == R["k_diameter"]
    assert s.average_path_length == pytest.approx(R["k_avg_path"], rel=1e-12)
    # isolates count as 0, igraph's isolates = "zero"
    assert s.average_clustering == pytest.approx(R["k_avg_clustering"], rel=1e-12)


def test_modularity_matches_igraph(R, karate):
    half = np.where(np.arange(34) < 17, 1, 2)
    assert sp.network_modularity(karate, half) == pytest.approx(
        R["k_modularity_half"], rel=1e-12
    )
    fg = np.asarray(R["k_fastgreedy_membership"])
    assert sp.network_modularity(karate, fg) == pytest.approx(
        R["k_fastgreedy_Q"], rel=1e-12
    )


def test_components_match_igraph(R, directed):
    g = sp.network_graph(_mat("network_disconnected.csv"), directed=False)
    c = sp.network_components(g)
    assert c.n_components == R["disc_ncomp"]
    assert sorted(c.sizes) == R["disc_comp_sizes"]
    assert (
        sp.network_components(directed, connection="weak").n_components
        == R["d_ncomp_weak"]
    )
    assert (
        sp.network_components(directed, connection="strong").n_components
        == R["d_ncomp_strong"]
    )


def test_louvain_distribution_matches_igraph(R, karate):
    """T3: compare 200 seeded runs as distributions, never run for run."""
    ours = np.array(
        [
            sp.community_detection(karate, method="louvain", seed=s).modularity
            for s in range(200)
        ]
    )
    ref = np.asarray(R["k_louvain_Q"], float)
    se = np.sqrt(ours.var(ddof=1) / ours.size + ref.var(ddof=1) / ref.size)
    assert abs(ours.mean() - ref.mean()) < 4 * se
    # both reach the same optimum, which is the known maximum for this graph
    assert ours.max() == pytest.approx(ref.max(), rel=1e-12)


# ── QAP network regression ────────────────────────────────────────────────


def _qap(name):
    return _mat(f"network_qap_{name}.csv")


def test_netlm_coefficients_match_sna(R):
    Y, X1, X2 = _qap("Y"), _qap("X1"), _qap("X2")
    r = sp.netlm(Y, [X1, X2], directed=True, nperm=10, seed=0)
    _close(r.coefficients["coef"], R["netlm_coef"], rtol=1e-12)
    ru = sp.netlm(Y + Y.T, [X1 + X1.T, X2 + X2.T], directed=False, nperm=10, seed=0)
    _close(ru.coefficients["coef"], R["netlm_graph_coef"], rtol=1e-12)


def test_netlogit_coefficients_match_sna(R):
    r = sp.netlogit(
        _qap("Yb"), [_qap("X1"), _qap("X2")], directed=True, nperm=10, seed=0
    )
    _close(r.coefficients["coef"], R["netlogit_coef"], rtol=1e-8)


# ── ERGM ──────────────────────────────────────────────────────────────────


def test_ergm_mple_matches_ergm_undirected(R, karate):
    at = pd.read_csv(_FIX / "network_karate_attrs.csv")
    res = sp.ergm(
        karate,
        terms=["edges", "triangles", "nodematch:grp", "nodecov:x", "absdiff:x"],
        node_attrs={"grp": at["grp"].to_numpy(), "x": at["x"].to_numpy()},
    )
    _close(res.coefficients["estimate"], R["ergm_undirected"]["coef"], rtol=1e-10)
    _close(res.coefficients["se"], R["ergm_undirected"]["se"], rtol=1e-6)


def test_ergm_mple_matches_ergm_directed(R, directed):
    res = sp.ergm(directed, terms=["edges", "mutual"])
    _close(res.coefficients["estimate"], R["ergm_directed"]["coef"], rtol=1e-12)
    _close(res.coefficients["se"], R["ergm_directed"]["se"], rtol=1e-6)


# ── dyadic-robust OLS ─────────────────────────────────────────────────────


@pytest.mark.parametrize("tag", ["und", "dir"])
def test_dyadic_regression_matches_dyadRobust(R, tag):
    df = pd.read_csv(_FIX / f"network_dyad_{tag}.csv")
    res = sp.dyadic_regression(df, y="y", covariates=["x"], i="i", j="j")
    _close(res.coefficients["coef"], R[f"dyad_{tag}"]["coef"], rtol=1e-12)
    _close(res.coefficients["se_dyadic"], R[f"dyad_{tag}"]["se"], rtol=1e-10)


def test_dyadic_se_is_the_asa_indicator_on_directed_data():
    """Regression guard for the shared-node COUNT defect, independent of R.

    Brute-force the definition: every pair of dyads sharing at least one
    member contributes once. Before the fix, (i, j) and (j, i) contributed
    twice and this assertion failed by 1.8%.
    """
    df = pd.read_csv(_FIX / "network_dyad_dir.csv")
    res = sp.dyadic_regression(df, y="y", covariates=["x"], i="i", j="j")
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    y = df["y"].to_numpy()
    XtXi = np.linalg.inv(X.T @ X)
    g = X * (y - X @ (XtXi @ X.T @ y))[:, None]
    I, J = df["i"].to_numpy(), df["j"].to_numpy()
    share = (
        (I[:, None] == I[None, :])
        | (I[:, None] == J[None, :])
        | (J[:, None] == I[None, :])
        | (J[:, None] == J[None, :])
    ).astype(float)
    se = np.sqrt(np.diag(XtXi @ (g.T @ share @ g) @ XtXi))
    _close(res.coefficients["se_dyadic"], se, rtol=1e-12)


def test_dyadic_regression_refuses_self_dyads():
    df = pd.DataFrame(
        {
            "i": [0, 1, 2, 2],
            "j": [1, 2, 0, 2],
            "x": [0.1, 0.4, 0.2, 0.9],
            "y": [1.0, 2.0, 1.4, 2.2],
        }
    )
    with pytest.raises(ValueError, match="not dyads"):
        sp.dyadic_regression(df, y="y", covariates=["x"], i="i", j="j")


# ── the individual centrality functions, called directly ──────────────────
# These four were graded cross-language before 1.27.0 on a test of closed
# forms (star, triangle, path) that never consulted R. The dispatcher tests
# above exercise them indirectly; these call each one by name so the index
# entry for each function points at a comparison that actually ran.


def test_degree_centrality_matches_igraph(R, karate, directed):
    _close(sp.degree_centrality(karate), np.asarray(R["k_degree"]) / 33.0)
    for mode, key in (
        ("in", "d_degree_in"),
        ("out", "d_degree_out"),
        ("all", "d_degree_all"),
    ):
        _close(sp.degree_centrality(directed, mode=mode, normalized=False), R[key])


def test_betweenness_centrality_matches_igraph(R, karate, directed):
    _close(sp.betweenness_centrality(karate), R["k_betweenness"])
    _close(sp.betweenness_centrality(karate, normalized=False), R["k_betweenness_raw"])
    _close(
        sp.betweenness_centrality(directed, normalized=False), R["d_betweenness_raw"]
    )


def test_eigenvector_centrality_matches_sna_evcent(R, karate, directed):
    """sna::evcent is unit-L2 like StatsPAI; igraph max-scales (see above).

    Both sides use power iteration, so the budget is the iteration
    tolerance (observed 5e-11 undirected, 8e-11 directed), not 1e-15.
    """
    _close(sp.eigenvector_centrality(karate), R["k_evcent_sna"], rtol=1e-9)
    _close(sp.eigenvector_centrality(directed), R["d_evcent_sna"], rtol=1e-9)


def test_local_clustering_matches_igraph(R, karate):
    _close(sp.clustering(karate), R["k_local_clustering"], rtol=1e-12)
    g = sp.network_graph(_mat("network_disconnected.csv"), directed=False)
    got = np.asarray(sp.clustering(g), float)
    ref = np.asarray(R["disc_local_clustering"], float)
    assert np.allclose(got, ref, rtol=1e-12, atol=0.0)  # degree < 2 -> 0 on both
