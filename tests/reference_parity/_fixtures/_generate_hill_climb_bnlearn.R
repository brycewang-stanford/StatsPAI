# ---------------------------------------------------------------------------
# Reference for tests/reference_parity/test_hill_climb_bnlearn_parity.py
#
# bnlearn: the BIC of fixed graphs (score(type = "bic") for categorical data,
# "bic-g" for continuous data), and the graph and score hc() ends at.
#
# Requires: bnlearn (5.2.1), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_hill_climb_bnlearn.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({library(bnlearn); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
dd <- read.csv(file.path(here, "hill_climb_discrete.csv"), stringsAsFactors = TRUE)
dg <- read.csv(file.path(here, "hill_climb_gaussian.csv"))
graphs_d <- c("[a][b][c][d][e]", "[a][b][c|a:b][d|c][e]", "[a][b|a][c|a:b][d|c:b][e|d]",
              "[c][a|c][b|c][d|c][e|a:b]")
graphs_g <- c("[x1][x2][x3][x4][x5][x6]", "[x1][x2][x3|x1:x2][x4|x3][x5|x1][x6]",
              "[x1][x2|x1][x3|x1:x2][x4|x3:x1][x5|x1:x4][x6|x5]", "[x4][x3|x4][x1|x3][x2|x3:x1][x5][x6|x2]")
sc <- function(gs, data, type) lapply(gs, function(g) list(graph = g, score = score(model2network(g), data, type = type)))
hd <- hc(dd); hg <- hc(dg)
arcs_of <- function(h) lapply(seq_len(nrow(arcs(h))), function(i) unname(arcs(h)[i, ]))
write_json(list(
  bnlearn = as.character(packageVersion("bnlearn")), R = R.version.string,
  discrete = list(fixed = sc(graphs_d, dd, "bic"), hc_score = score(hd, dd, type = "bic"), hc_arcs = arcs_of(hd)),
  gaussian = list(fixed = sc(graphs_g, dg, "bic-g"), hc_score = score(hg, dg, type = "bic-g"), hc_arcs = arcs_of(hg))),
  file.path(here, "hill_climb_bnlearn_R.json"), digits = NA, auto_unbox = TRUE)
