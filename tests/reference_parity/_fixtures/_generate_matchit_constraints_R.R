# Reference values for sp.match(exact=, covariate_calipers=, mahvars=)
# checked in tests/reference_parity/test_matchit_constraints_parity.py.
# Run from this folder:  Rscript _generate_matchit_constraints_R.R
# matchit_constraints.csv is synthetic (1,000 units), made by
#   rng = np.random.default_rng(20261019); x1, x2 = rng.normal(size=n);
#   g = rng.integers(0, 3, n); x3 = rng.binomial(1, 0.4, n);
#   d = uniform < 1 / (1 + exp(0.9 - 0.8 x1 + 0.5 x2 - 0.4 g));
#   y = 1 + 0.5 d + 0.8 x1 - 0.4 x2 + 0.3 g + normal
suppressMessages({library(MatchIt); library(jsonlite)})
d <- read.csv("matchit_constraints.csv")
calls <- list(
  exact_ps = list(f = d ~ x1 + x2 + g, exact = ~ g),
  exact_ps_replace = list(f = d ~ x1 + x2 + g, exact = ~ g, replace = TRUE),
  exact_two_ps = list(f = d ~ x1 + x2 + g + x3, exact = ~ g + x3),
  exact_mahal = list(f = d ~ x1 + x2, exact = ~ g, distance = "mahalanobis"),
  covcal_ps = list(f = d ~ x1 + x2 + g, caliper = c(x1 = 0.1)),
  covcal_ps_raw = list(f = d ~ x1 + x2 + g, caliper = c(x2 = 0.05), std.caliper = FALSE),
  mahvars = list(f = d ~ x1 + x2 + g, mahvars = ~ x1 + x2, caliper = 0.25),
  mahvars_exact_replace = list(f = d ~ x1 + x2 + g, mahvars = ~ x1 + x2,
                               exact = ~ x3, caliper = 0.5, replace = TRUE)
)
out <- list()
for (name in names(calls)) {
  a <- calls[[name]]
  f <- a$f; a$f <- NULL
  m <- suppressWarnings(do.call(matchit, c(list(f, data = d), a)))
  md <- match.data(m)
  w <- md$weights
  mm <- m$match.matrix
  out[[name]] <- list(
    att = weighted.mean(md$y[md$d == 1], w[md$d == 1]) -
      weighted.mean(md$y[md$d == 0], w[md$d == 0]),
    n_matched = sum(!is.na(mm[, 1])),
    treated_row = as.integer(rownames(mm)),
    control_row = as.integer(mm[, 1])
  )
}
out$versions <- list(R = R.version.string, MatchIt = as.character(packageVersion("MatchIt")))
write_json(out, "matchit_constraints_R.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
