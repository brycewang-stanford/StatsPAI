# Reference values for sp.match(design=) -- nearest-neighbour matching with
# the design weights as MatchIt s.weights and the survey linearisation of
# the matched-sample regression; checked in
# tests/reference_parity/test_match_design_parity.py.
# Run from this folder:  Rscript _generate_match_design_R.R
# MatchIt and survey are GPL: used as black boxes (documentation and outputs
# only).
# Data: matchit_constraints.csv with sw = 0.4 + (row %% 5) * 0.3 (as in
# _generate_match_sweights_R.R) and a design added here and on the Python
# side: stratum = (row - 1) %/% 100 + 1, psu = (row - 1) %/% 20 + 1,
# fpc = 40 PSUs per stratum.
# Recipe: matchit(s.weights = ~sw); the matched rows carry match_data's
# weights (matching weight x s.weights); svyglm(y ~ d) on the domain of
# matched rows of the full design (subset(), so every stratum and PSU keeps
# its place in the variance), treating the matching weights as fixed.
suppressMessages({library(MatchIt); library(survey); library(jsonlite)})
d <- read.csv("matchit_constraints.csv")
i <- seq_len(nrow(d))
d$sw <- 0.4 + (i %% 5) * 0.3
d$stratum <- (i - 1) %/% 100 + 1
d$psu <- (i - 1) %/% 20 + 1
d$fpc <- 40
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
  mw <- m$weights               # matching weights, 0 = unmatched
  matched <- mw > 0
  dd <- d
  dd$cw <- ifelse(matched, mw * d$sw, d$sw)
  dd$matched <- matched
  res <- list(n_matched_rows = sum(matched))
  for (nm in c("full", "nofpc", "psu_only")) {
    des <- switch(nm,
      full = svydesign(ids = ~psu, strata = ~stratum, fpc = ~fpc, weights = ~cw, data = dd),
      nofpc = svydesign(ids = ~psu, strata = ~stratum, weights = ~cw, data = dd),
      psu_only = svydesign(ids = ~psu, weights = ~cw, data = dd))
    fit <- svyglm(y ~ d, design = subset(des, matched))
    res[[nm]] <- list(est = unname(coef(fit)["d"]), se = unname(SE(fit)["d"]))
  }
  out[[name]] <- c(a, res)
}
out$versions <- list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt")),
                     survey = as.character(packageVersion("survey")))
write_json(out, "match_design_R.json", digits = NA, auto_unbox = TRUE)
