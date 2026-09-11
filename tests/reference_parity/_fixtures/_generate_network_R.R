#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_network_parity.py
#
# Requires: R 4.5 + igraph + sna + network + ergm + dyadRobust (GitHub:
# jbisbee1/dyadRobust). Run _generate_network_data.py first, then this file
# from any directory.
#
# Conventions this fixture pins, each of which decides a number below
# -------------------------------------------------------------------
# * igraph and sna both export degree / betweenness / closeness; everything
#   is namespace-qualified so the result cannot depend on package load order.
# * igraph max-scales eigenvector and HITS scores; StatsPAI L2- / L1-
#   normalises (networkx's conventions). Raw vectors are emitted and the
#   test asserts the RATIO is constant rather than rescaling here.
# * transitivity(type = "average") is emitted with isolates = "zero", the
#   convention StatsPAI's network_summary uses; igraph's default excludes
#   degree < 2 vertices instead.
# * Louvain is randomised in every implementation, so 200 seeded runs are
#   emitted and compared as a distribution, not run for run.
# * dyadRobust recodes ego / alter inside one dplyr::mutate() call, which
#   evaluates sequentially: the alter recode builds its levels from the
#   ALREADY-recoded ego column, so a node gets different codes as ego and
#   as alter unless ids are already 1..N in first-appearance order (then
#   the recode is the identity). Ids are passed that way here, and the
#   stopifnot() below checks the precondition rather than assuming it.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(jsonlite); library(network); library(ergm)
  library(dyadRobust); library(dplyr)
})
ig <- asNamespace("igraph")
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
rd <- function(f) { m <- as.matrix(read.csv(file.path(OUT, f), header = FALSE)); dimnames(m) <- NULL; m }

out <- list()

# ---- bundled datasets, from independent copies ----------------------------
gk <- ig$make_graph("Zachary")
K <- as.matrix(ig$as_adjacency_matrix(gk, sparse = FALSE)); dimnames(K) <- NULL
write.csv(K, file.path(OUT, "network_karate_igraph.csv"), row.names = FALSE)
data(florentine)
FM <- as.matrix(flomarriage)
write.csv(FM, file.path(OUT, "network_flomarriage.csv"))

# ---- undirected: karate ---------------------------------------------------
out$k_degree          <- as.numeric(ig$degree(gk))
out$k_betweenness     <- as.numeric(ig$betweenness(gk, normalized = TRUE))
out$k_betweenness_raw <- as.numeric(ig$betweenness(gk, normalized = FALSE))
out$k_closeness       <- as.numeric(ig$closeness(gk, normalized = TRUE))
out$k_eigen_maxscaled <- as.numeric(ig$eigen_centrality(gk)$vector)
# sna::evcent returns the leading eigenvector unrescaled, i.e. unit L2 norm --
# StatsPAI's convention -- where igraph (2.x ignores scale = FALSE) always
# max-scales. abs() fixes eigen()'s arbitrary sign; the Perron vector of a
# connected graph is positive.
out$k_evcent_sna <- abs(as.numeric(sna::evcent(K, gmode = "graph")))
out$k_local_clustering <- as.numeric(ig$transitivity(gk, type = "local", isolates = "zero"))
out$k_pagerank        <- as.numeric(ig$page_rank(gk, damping = 0.85)$vector)
out$k_transitivity    <- ig$transitivity(gk, type = "global")
out$k_avg_clustering  <- ig$transitivity(gk, type = "average", isolates = "zero")
out$k_assortativity   <- ig$assortativity_degree(gk)
out$k_density         <- ig$edge_density(gk)
out$k_diameter        <- ig$diameter(gk)
out$k_avg_path        <- ig$mean_distance(gk)
out$k_bonpow_igraph   <- as.numeric(ig$power_centrality(gk, exponent = 0.1))
out$k_bonpow_sna      <- as.numeric(sna::bonpow(K, exponent = 0.1))
out$k_alpha_01        <- as.numeric(ig$alpha_centrality(gk, alpha = 0.1))
out$k_modularity_half <- ig$modularity(gk, ifelse(seq_len(34) <= 17, 1, 2))
fg <- ig$cluster_fast_greedy(gk)
out$k_fastgreedy_membership <- as.integer(ig$membership(fg))
out$k_fastgreedy_Q <- ig$modularity(fg)
out$k_louvain_Q <- sapply(1:200, function(s) { set.seed(s); ig$modularity(ig$cluster_louvain(gk)) })

# ---- directed -------------------------------------------------------------
D <- rd("network_directed.csv")
gd <- ig$graph_from_adjacency_matrix(D, mode = "directed")
out$d_reciprocity     <- ig$reciprocity(gd)
out$d_pagerank        <- as.numeric(ig$page_rank(gd, damping = 0.85)$vector)
hs <- ig$hits_scores(gd)
out$d_hub_maxscaled       <- as.numeric(hs$hub)
out$d_authority_maxscaled <- as.numeric(hs$authority)
out$d_betweenness_raw <- as.numeric(ig$betweenness(gd, directed = TRUE, normalized = FALSE))
out$d_evcent_sna <- abs(as.numeric(sna::evcent(D, gmode = "digraph")))
out$d_degree_in  <- as.numeric(ig$degree(gd, mode = "in"))
out$d_degree_out <- as.numeric(ig$degree(gd, mode = "out"))
out$d_degree_all <- as.numeric(ig$degree(gd, mode = "all"))
out$d_ncomp_weak   <- ig$components(gd, mode = "weak")$no
out$d_ncomp_strong <- ig$components(gd, mode = "strong")$no

# ---- disconnected: Wasserman-Faust closeness from igraph distances --------
A <- rd("network_disconnected.csv"); n <- nrow(A)
ga <- ig$graph_from_adjacency_matrix(A, mode = "undirected")
dist <- ig$distances(ga)
reach <- rowSums(is.finite(dist)) - 1
sumd  <- apply(dist, 1, function(r) sum(r[is.finite(r) & r > 0]))
out$disc_wf_closeness <- as.numeric(ifelse(sumd > 0, (reach / (n - 1)) * (reach / sumd), 0))
cp <- ig$components(ga)
out$disc_ncomp <- cp$no
out$disc_comp_sizes <- as.integer(sort(cp$csize))
out$disc_avg_clustering <- ig$transitivity(ga, type = "average", isolates = "zero")
out$disc_local_clustering <- as.numeric(ig$transitivity(ga, type = "local", isolates = "zero"))

# ---- QAP regressions ------------------------------------------------------
Y <- rd("network_qap_Y.csv"); Yb <- rd("network_qap_Yb.csv")
X1 <- rd("network_qap_X1.csv"); X2 <- rd("network_qap_X2.csv")
set.seed(1)
out$netlm_coef <- as.numeric(sna::netlm(Y, list(X1, X2), mode = "digraph",
                                        nullhyp = "qapspp", reps = 20)$coefficients)
out$netlm_graph_coef <- as.numeric(sna::netlm(Y + t(Y), list(X1 + t(X1), X2 + t(X2)),
                                              mode = "graph", nullhyp = "qapspp",
                                              reps = 20)$coefficients)
out$netlogit_coef <- as.numeric(sna::netlogit(Yb, list(X1, X2), mode = "digraph",
                                              nullhyp = "qapspp", reps = 20)$coefficients)

# ---- ERGM by maximum pseudo-likelihood ------------------------------------
at <- read.csv(file.path(OUT, "network_karate_attrs.csv"))
nw <- network(K, directed = FALSE); nw %v% "grp" <- at$grp; nw %v% "x" <- at$x
nd <- network(D, directed = TRUE)
fit_u <- ergm(nw ~ edges + triangle + nodematch("grp") + nodecov("x") + absdiff("x"),
              estimate = "MPLE")
fit_d <- ergm(nd ~ edges + mutual, estimate = "MPLE")
pack <- function(f) list(coef = as.numeric(coef(f)), se = as.numeric(sqrt(diag(vcov(f)))))
out$ergm_undirected <- pack(fit_u)
out$ergm_directed   <- pack(fit_d)

# ---- dyadic-robust OLS ----------------------------------------------------
for (tag in c("und", "dir")) {
  d <- read.csv(file.path(OUT, paste0("network_dyad_", tag, ".csv")))
  d$i1 <- d$i + 1L; d$j1 <- d$j + 1L
  stopifnot(identical(unique(c(d$i1, d$j1)), seq_len(max(d$i1, d$j1))))
  f <- lm(y ~ x, data = d)
  r <- suppressWarnings(dyadRobust(f, dat = d, dyadid = "pair", egoid = "i1", alterid = "j1"))
  out[[paste0("dyad_", tag)]] <- list(coef = as.numeric(r$bhat), se = as.numeric(r$sehat))
}

out$provenance <- list(
  r = R.version.string,
  igraph = as.character(packageVersion("igraph")),
  sna = as.character(packageVersion("sna")),
  ergm = as.character(packageVersion("ergm")),
  network = as.character(packageVersion("network")),
  dyadRobust = as.character(packageVersion("dyadRobust"))
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           file.path(OUT, "network_R.json"))
cat("wrote network_R.json\n")
