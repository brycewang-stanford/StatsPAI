# Reference values for sp.match(sampling_weights=) vs R MatchIt's
# s.weights; checked in tests/reference_parity/test_match_sweights_parity.py.
# Run from this folder:  Rscript _generate_match_sweights_R.R
# MatchIt and sandwich are GPL: used as black boxes (outputs only).
# Data: matchit_constraints.csv; sampling weights sw = 0.4 + (row %% 5) * 0.3
# (rows numbered from 1), added here and on the Python side.
# Effect and SE as the MatchIt vignettes prescribe: lm(y ~ d, weights =
# match_data weights, which include s.weights); cluster-robust on subclass
# (sandwich::vcovCL defaults) without replacement, HC3 with replacement.
# k:1 matching without replacement is left out: MatchIt assigns the k-th
# controls in a later round than the first ones, sp.match all k at once,
# so the matched sets differ when controls run short (a property of the
# matching, not of the sampling weights).
suppressMessages({library(MatchIt); library(sandwich); library(jsonlite)})
d <- read.csv("matchit_constraints.csv")
d$sw <- 0.4 + (seq_len(nrow(d)) %% 5) * 0.3
calls <- list(
  nn1 = list(replace = FALSE, ratio = 1),
  nn2_replace = list(replace = TRUE, ratio = 2),
  nn1_replace = list(replace = TRUE, ratio = 1),
  nn1_caliper = list(replace = FALSE, ratio = 1, caliper = 0.1)
)
out <- list()
for (name in names(calls)) {
  a <- calls[[name]]
  m <- suppressWarnings(do.call(matchit, c(list(d ~ x1 + x2 + g, data = d,
                                                s.weights = ~sw), a)))
  md <- match_data(m)
  fit <- lm(y ~ d, data = md, weights = weights)
  V <- if (isTRUE(a$replace)) vcovHC(fit, type = "HC3") else vcovCL(fit, cluster = ~subclass)
  mm <- m$match.matrix
  out[[name]] <- c(a, list(
    est = unname(coef(fit)["d"]), se = sqrt(V["d", "d"]),
    distance = unname(m$distance),
    n_matched = sum(!is.na(mm[, 1])),
    treated_row = as.integer(rownames(mm)),
    control_rows = lapply(seq_len(nrow(mm)), function(i) as.integer(mm[i, ]))))
}
out$versions <- list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt")),
                     sandwich = as.character(packageVersion("sandwich")))
write_json(out, "match_sweights_R.json", digits = NA, auto_unbox = TRUE)
