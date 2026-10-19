# Reference pairs for sp.match(method='cem', k2k=True) vs R MatchIt 4.7.2
# (GPL; black box, outputs only); checked in
# tests/reference_parity/test_cem_k2k_parity.py.
# Run from this folder:  Rscript _generate_cem_k2k_R.R
# Data: matchit_constraints.csv. cutpoints = 4 equal-width bins per
# covariate (sp.match n_bins=4 gives the same cells on this file).
suppressMessages({library(MatchIt); library(jsonlite)})
d <- read.csv("matchit_constraints.csv")
out <- list()
for (cv in list(c("x1", "x2"), c("x1", "x2", "x3"))) {
  f <- as.formula(paste("d ~", paste(cv, collapse = " + ")))
  m <- matchit(f, data = d, method = "cem", cutpoints = 4, k2k = TRUE)
  mm <- m$match.matrix
  md <- match.data(m)
  key <- paste(cv, collapse = "_")
  out[[key]] <- list(covariates = cv, treated_row = as.integer(rownames(mm)),
                     control_row = as.integer(mm[, 1]),
                     att = weighted.mean(md$y[md$d == 1], md$weights[md$d == 1]) -
                       weighted.mean(md$y[md$d == 0], md$weights[md$d == 0]))
}
out$versions <- list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt")))
write_json(out, "cem_k2k_R.json", digits = NA, auto_unbox = TRUE)
