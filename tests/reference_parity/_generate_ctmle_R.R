# Reference values for tests/reference_parity/test_ctmle_R_parity.py
#
# ctmle::ctmleDiscrete (greedy C-TMLE) on _fixtures/ctmle_data.csv: three
# simulated data sets (rep 0..2), each with two supplied initial outcome
# fits, a correctly specified one (q0c, q1c) and one that omits the
# confounder x1 (q0w, q1w). Records the order in which covariates enter
# the propensity model and the estimate at every step of the sequence.
#
# Run from tests/reference_parity:  Rscript _generate_ctmle_R.R
suppressMessages({library(ctmle); library(jsonlite)})
d <- read.csv("_fixtures/ctmle_data.csv")
out <- list(ctmle_version = as.character(packageVersion("ctmle")),
            r_version = R.version.string)
for (r in 0:2) for (q in c("c", "w")) {
  x <- d[d$rep == r, ]
  W <- as.matrix(x[, c("x1", "x2", "x3")])
  f <- ctmleDiscrete(Y = x$y, A = x$a, W = W,
                     Q = cbind(x[[paste0("q0", q)]], x[[paste0("q1", q)]]),
                     preOrder = FALSE, detailed = TRUE,
                     folds = split(seq_len(nrow(x)), x$fold))
  out[[paste0("rep", r, "_", q)]] <- list(
    terms = f$candidates$terms,
    candidate_estimates = as.numeric(unlist(f$candidates$results.all$est)),
    best_k = f$best_k, est = f$est, var = f$var.psi)
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "_fixtures/ctmle_R.json")
cat("wrote _fixtures/ctmle_R.json\n")
