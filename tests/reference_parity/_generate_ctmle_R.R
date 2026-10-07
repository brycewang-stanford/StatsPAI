# Reference values for tests/reference_parity/test_ctmle_R_parity.py
#
# ctmle::ctmleDiscrete (greedy C-TMLE) on _fixtures/ctmle_data.csv: three
# simulated data sets (rep 0..2), each with two supplied initial outcome
# fits, a correctly specified one (q0c, q1c) and one that omits the
# confounder x1 (q0w, q1w). Records the order in which covariates enter
# the propensity model and the estimate at every step of the sequence, for
# the greedy search and for six pre-ordered sequences.
#
# ctmleDiscrete ignores its folds argument (its results depend on the seed
# only), so best_k is recorded for information and not compared.
#
# Run from tests/reference_parity:  Rscript _generate_ctmle_R.R
suppressMessages({library(ctmle); library(jsonlite)})
d <- read.csv("_fixtures/ctmle_data.csv")
out <- list(ctmle_version = as.character(packageVersion("ctmle")),
            r_version = R.version.string)
for (r in 0:2) for (q in c("c", "w")) {
  set.seed(1)
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
# Pre-ordered sequences (preOrder = TRUE) on the first data set.
x <- d[d$rep == 0, ]
W <- as.matrix(x[, c("x1", "x2", "x3")])
pre <- list()
for (q in c("c", "w")) for (ord in list(c(1, 2, 3), c(2, 3, 1), c(3, 1, 2))) {
  # In this mode ctmleDiscrete stops extending the sequence once its
  # (random) cross-validation says so; the seed fixes how far it goes.
  set.seed(1)
  f <- ctmleDiscrete(Y = x$y, A = x$a, W = W,
                     Q = cbind(x[[paste0("q0", q)]], x[[paste0("q1", q)]]),
                     preOrder = TRUE, order = ord, detailed = TRUE)
  pre[[paste0(q, "_", paste0("x", ord, collapse = "_"))]] <- list(
    terms = f$candidates$terms,
    candidate_estimates = as.numeric(unlist(f$candidates$results.all$est)))
}
out$preordered <- pre
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "_fixtures/ctmle_R.json")
cat("wrote _fixtures/ctmle_R.json\n")
