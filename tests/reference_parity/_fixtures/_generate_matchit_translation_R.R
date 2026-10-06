# Reference values for the MatchIt translations checked in
# tests/reference_parity/test_accounting_research_parity.py
# Run from this folder:  Rscript _generate_matchit_translation_R.R
# matchit_translation.csv is synthetic: 1,200 units, 328 treated, made by
#   rng = np.random.default_rng(77); x1, x2 = rng.normal(size=n) twice;
#   d = uniform < 1 / (1 + exp(1.2 - 0.9 x1 + 0.5 x2));
#   y = 1 + 0.5 d + 0.8 x1 - 0.4 x2 + normal
suppressMessages({library(MatchIt); library(jsonlite)})
d <- read.csv("matchit_translation.csv")
calls <- list(
  default = list(),
  caliper_sd = list(caliper = 0.2),
  caliper_raw = list(caliper = 0.03, std.caliper = FALSE),
  smallest = list(m.order = "smallest"),
  data = list(m.order = "data"),
  replace = list(replace = TRUE)
)
out <- list()
for (name in names(calls)) {
  m <- do.call(matchit, c(list(d ~ x1 + x2, data = d), calls[[name]]))
  md <- match.data(m)
  w <- md$weights
  out[[name]] <- list(
    att = weighted.mean(md$y[md$d == 1], w[md$d == 1]) -
      weighted.mean(md$y[md$d == 0], w[md$d == 0]),
    n_matched = sum(!is.na(m$match.matrix[, 1]))
  )
}
out$versions <- list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt")))
write_json(out, "matchit_translation_R.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
