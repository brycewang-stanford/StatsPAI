# Social network analysis (`sp.network`)

StatsPAI ships a numpy/scipy-native social-network-analysis toolkit that
mirrors the workflow of R's `igraph` / `sna` / `statnet` and Stata's
`nwcommands` — without pulling in `networkx` as a dependency. One `Graph`
object flows through every layer: descriptives → centrality → community
detection → network regression → ERGM.

```python
import statspai as sp

g = sp.karate_club()                 # Zachary (1977), 34 nodes / 78 edges
sp.network_summary(g)                # structural summary
sp.centrality(g, kind="all")         # per-node centrality table
sp.community_detection(g)            # Louvain partition
```

## Building a graph

`sp.network_graph` is the single entry point. Pass **either** an adjacency
matrix **or** an edge list:

```python
import numpy as np

# from an edge list
g = sp.network_graph(edges=[(0, 1), (1, 2), (2, 0)])

# from a (possibly sparse) adjacency, with labels
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
g = sp.network_graph(A, node_labels=["a", "b", "c"])

# directed, weighted, or from a tidy DataFrame
g = sp.network.Graph.from_pandas_edgelist(df, "source", "target",
                                          weight="w", directed=True)
```

Undirected graphs are stored symmetrically; passing an asymmetric matrix with
`directed=False` symmetrises it **with a warning** (never silently). Self-loops
are dropped by default (the SNA convention).

## Descriptive statistics

```python
res = sp.network_summary(g)
res.density, res.transitivity, res.average_clustering
res.diameter, res.average_path_length, res.assortativity
res.n_components, res.is_connected
```

| Function | What it returns |
| --- | --- |
| `sp.network_summary(g)` | omnibus structural summary |
| `sp.transitivity(g)` | global clustering coefficient |
| `sp.clustering(g)` | per-node local clustering (Series) |
| `sp.reciprocity(g)` | share of reciprocated arcs (directed) |
| `sp.assortativity(g)` | Newman degree assortativity |
| `sp.network_components(g)` | connected-component decomposition |

These match `igraph`/`networkx` to machine precision (see
`tests/test_network.py`).

## Centrality

The `sp.centrality` dispatcher computes one or more measures and returns a
per-node table:

```python
cen = sp.centrality(g, kind="all")          # degree, closeness, betweenness,
                                            # eigenvector, pagerank
cen.scores                                  # DataFrame indexed by node
cen.top("betweenness", 5)                   # 5 highest brokers
cen.most_central                            # {measure: node}
```

Named functions are also exposed: `sp.degree_centrality`,
`sp.closeness_centrality`, `sp.betweenness_centrality` (Brandes 2001),
`sp.eigenvector_centrality`, `sp.katz_centrality`, `sp.pagerank` (Brin-Page),
`sp.bonacich_power`, and `sp.hits`.

PageRank returns the exact stationary distribution of the Google matrix (the
`igraph`-equivalent value); on a binary undirected graph it equals the
degree-proportional walk shrunk toward uniform by the damping factor.

## Community detection

```python
com = sp.community_detection(g, method="louvain")   # or "greedy", "label_prop"
com.membership          # community id per node
com.n_communities
com.modularity          # Newman Q
sp.network_modularity(g, com.membership)            # Q of any partition
```

- **Louvain** (Blondel et al. 2008) — multi-level modularity optimisation;
  reaches Q ≈ 0.42 on the karate club.
- **Greedy / CNM** (Clauset-Newman-Moore 2004) — agglomerative; reproduces
  `networkx`'s partition exactly (Q = 0.3807, 3 communities).
- **Label propagation** (Raghavan et al. 2007) — near-linear-time; pass `seed`
  for reproducibility.

## Network regression (QAP / MRQAP / dyadic)

Regress one relational matrix on others with permutation inference robust to
network autocorrelation — the `sna::netlm` / `netlogit` workflow:

```python
res = sp.netlm(Y, {"distance": D, "same_dept": S}, nperm=1000)
res.coefficients          # coef, se, z, QAP p-value
res.p_qap                 # permutation p-values (the headline inference)
```

`method="dsp"` (default) is Dekker-Krackhardt-Snijders double-semi-partialling,
robust to collinearity among predictors. `sp.netlogit` is the logistic version
for a binary dependent network.

For dyad-level data with covariates, `sp.dyadic_regression` reports
**dyadic-cluster-robust** standard errors (Aronow-Samii-Assenova 2015;
Fafchamps-Gubert 2007) that allow arbitrary correlation between any two dyads
sharing a node — the dependence that invalidates classical SEs in network data:

```python
res = sp.dyadic_regression(df, y="trade", covariates=["log_dist", "shared_lang"],
                           i="country_i", j="country_j")
res.coefficients[["variable", "coef", "se_dyadic", "se_classical"]]
```

## Network formation (ERGM)

`sp.ergm` fits exponential random graph models by maximum pseudo-likelihood
(MPLE):

```python
res = sp.ergm(g, terms=["edges", "nodematch:dept", "absdiff:age"],
              node_attrs=attrs)
res.coefficients          # term, estimate, se, z, p
```

Supported terms: `edges`, `mutual` (directed reciprocity), `triangles`,
`nodematch:<attr>` (homophily), `nodecov:<attr>`, `absdiff:<attr>`.

!!! note "MPLE scope"
    For **dyad-independent** terms (everything except `triangles`) MPLE
    coincides with the exact MLE and matches `statnet`'s MPLE option. For
    **dyad-dependent** terms (`triangles`) MPLE is consistent but approximate
    and its standard errors understate uncertainty — `sp.ergm` warns loudly.
    Full **MCMC-MLE** and **SAOM / RSiena** network-dynamics models are on the
    roadmap.

## Plotting

```python
com = sp.community_detection(g)
sp.network_plot(g, node_color=com.membership, layout="spring")
```

`matplotlib` is imported lazily (install the `plotting` extra); the
Fruchterman-Reingold and circular layouts are computed in numpy.

## Reference networks

- `sp.karate_club()` — Zachary's (1977) karate club (the community-detection
  benchmark); the observed factional split is `sp.network.KARATE_FACTION`.
- `sp.florentine_families()` — Padgett & Ansell's (1993) Renaissance marriage
  network, where the Medici sit at the structural centre (highest betweenness).

## How this relates to the interference module

`sp.network` is about network **structure** (who is central, what the
communities are, what predicts ties). For **causal inference under
interference / spillovers on a network** — direct + spillover effects, peer
effects, exposure mapping — see `sp.interference` (`sp.spillover`,
`sp.network_exposure`, `sp.peer_effects`, `sp.network_hte`).
