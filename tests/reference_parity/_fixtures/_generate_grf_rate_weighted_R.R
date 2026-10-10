#!/usr/bin/env Rscript
# R grf reference for the *weighted* RATE operator.
#
# sp.rate() evaluates AUTOC / QINI on doubly-robust scores with the
# forest's observation weights (equalize_cluster_weights=True gives every
# cluster the same total weight). The map
#
#   (scores, priorities, sample weights) -> AUTOC, QINI, TOC curve
#
# is deterministic, so it can be pinned against
# grf::rank_average_treatment_effect.fit(DR.scores, priorities, target=,
# sample.weights=, clusters=), called here as a black box on fixed inputs
# (grf is GPL-3: only its exported function is run, its source is not
# consulted).  The standard error on the R side is a half-sample
# bootstrap and is recorded for a loose comparison only.
#
# Cases
#   continuous        distinct priorities, positive random weights
#   tied              12 priority levels, positive random weights
#   tied_permuted     the rows of `tied` in another order (the estimate
#                     must not depend on row order within tie groups)
#   tied_clustered    12 priority levels, weights 1 / cluster size, clusters
#   tied_unit         12 priority levels, unit weights (the unweighted
#                     operator, as a consistency check)
#
#   Rscript tests/reference_parity/_fixtures/_generate_grf_rate_weighted_R.R

suppressMessages({
  library(grf)
  library(jsonlite)
})

set.seed(20261010)
n <- 240
x <- rnorm(n)
scores <- 1 + 1.5 * x + rnorm(n, sd = 2)
prio_cont <- x + rnorm(n, sd = 0.5)
prio_tied <- as.numeric(cut(prio_cont, quantile(prio_cont, 0:12 / 12),
                            include.lowest = TRUE, labels = FALSE))
w_rand <- round(runif(n, 0.2, 3), 6)
sizes <- c(rep(2, 20), rep(5, 16), rep(10, 12))
clusters <- sample(rep(seq_along(sizes), sizes))
w_clus <- 1 / as.numeric(table(clusters)[as.character(clusters)])
perm <- sample(n)
q <- seq(0.05, 1, by = 0.05)

run <- function(s, p, w, cl = NULL) {
  out <- list(scores = s, priorities = p, weights = w)
  if (!is.null(cl)) out$clusters <- cl
  for (target in c("AUTOC", "QINI")) {
    set.seed(1)
    r <- rank_average_treatment_effect.fit(
      s, p, target = target, q = q, R = 2000,
      sample.weights = w, clusters = cl
    )
    out[[target]] <- list(
      estimate = unname(r$estimate),
      std_err = unname(r$std.err),
      toc_q = r$TOC$q,
      toc = r$TOC$estimate
    )
  }
  out
}

out <- list(
  meta = list(
    R_version = R.version.string,
    grf_version = as.character(packageVersion("grf")),
    R_bootstrap = 2000L,
    note = paste(
      "rank_average_treatment_effect.fit on fixed scores, priorities and",
      "sample weights; estimates and TOC are deterministic, std_err is a",
      "half-sample bootstrap (whole clusters when clusters are given)."
    )
  ),
  q = q,
  cases = list(
    continuous = run(scores, prio_cont, w_rand),
    tied = run(scores, prio_tied, w_rand),
    tied_permuted = run(scores[perm], prio_tied[perm], w_rand[perm]),
    tied_clustered = run(scores, prio_tied, w_clus, clusters),
    tied_unit = run(scores, prio_tied, rep(1, n))
  )
)

path <- "tests/reference_parity/_fixtures/grf_rate_weighted_R.json"
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
cat("wrote", path, "\n")
