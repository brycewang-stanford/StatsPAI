# Reference trees for sp.policy_tree multi-action (reward-matrix) search;
# checked in tests/reference_parity/test_policytree_multi_action_parity.py.
# Run from this folder:  Rscript _generate_policytree_multi_R.R
# policytree is MIT. policytree_multi.csv was written by numpy
# (default_rng(20261019), n = 400; x4 rounded to 0.1 and x3 integer so
# ties are exercised). split.step > 1 is not compared: policytree thins
# the split grid by sample, sp.policy_tree by distinct value, so the two
# approximations differ (they agree at split.step = 1).
suppressPackageStartupMessages({ library(policytree); library(jsonlite) })
d <- read.csv("policytree_multi.csv")
X <- as.matrix(d[, c("x1", "x2", "x3", "x4")])
cases <- list(
  a3_d1 = list(G = c("g0", "g1", "g2"), depth = 1, mns = 1, step = 1),
  a3_d2 = list(G = c("g0", "g1", "g2"), depth = 2, mns = 1, step = 1),
  a4_d2 = list(G = c("g0", "g1", "g2", "g3"), depth = 2, mns = 1, step = 1),
  a4_d2_mns20 = list(G = c("g0", "g1", "g2", "g3"), depth = 2, mns = 20, step = 1),
  a3_d2_mns10 = list(G = c("g0", "g1", "g2"), depth = 2, mns = 10, step = 1)
)
nodes <- function(tr) lapply(tr$nodes, function(nd) nd[c("is_leaf", "split_variable", "split_value", "action", "left_child", "right_child")])
out <- list()
for (nm in names(cases)) {
  a <- cases[[nm]]
  Gm <- as.matrix(d[, a$G])
  tr <- policy_tree(X, Gm, depth = a$depth, min.node.size = a$mns, split.step = a$step)
  out[[nm]] <- c(a, list(actions = as.integer(predict(tr, X)), nodes = nodes(tr)))
}
out$versions <- list(R = R.version.string, policytree = as.character(packageVersion("policytree")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), null = "null"), "policytree_multi_R.json")
