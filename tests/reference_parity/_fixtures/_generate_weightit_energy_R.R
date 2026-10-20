# Reference outputs for sp.weightit(method='energy')
# (tests/reference_parity/test_weightit_energy_parity.py).
#
# WeightIt 1.7.0 is GPL: used as a black box (documentation and outputs on
# this fixture only). Same data as _generate_weightit_cobalt_R.R. Run from
# this directory:
#   Rscript _generate_weightit_energy_R.R
suppressMessages({library(WeightIt); library(jsonlite)})
d <- read.csv("weightit_cobalt_data.csv"); d$treat <- factor(d$treat)
# a three-level grouping coded as two numeric indicators
d$g2 <- as.numeric(d$x1 > 0 & d$x2 <= 0.5 | d$x1 <= 0 & d$x2 > 0.5)
d$g3 <- as.numeric(d$x1 > 0 & d$x2 > 0.5)
fb <- bin ~ x1 + x2 + x3; fm <- treat ~ x1 + x2 + x3
out <- list(`_versions` = list(WeightIt = as.character(packageVersion("WeightIt")),
                               osqp = as.character(packageVersion("osqp"))))
case <- function(name, formula, estimand, focal = NULL, ...) {
  args <- list(...)
  W <- do.call(weightit, c(list(formula, data = d, method = "energy",
                                estimand = estimand, focal = focal), args))
  out$cases[[name]] <<- list(treat = all.vars(formula)[1], covs = all.vars(formula)[-1],
                             estimand = estimand, focal = focal, args = args,
                             w = unname(W$weights))
}
case("bin_ATE", fb, "ATE")
case("bin_ATE_standard", fb, "ATE", improved = FALSE)
case("bin_ATT", fb, "ATT")
case("bin_ATC", fb, "ATC")
case("bin_ATE_lambda1", fb, "ATE", lambda = 1)
case("bin_ATT_lambda1", fb, "ATT", lambda = 1)
case("bin_ATE_mahalanobis", fb, "ATE", dist.mat = "mahalanobis")
case("bin_ATE_euclidean", fb, "ATE", dist.mat = "euclidean")
case("bin_ATE_moments1", fb, "ATE", moments = 1)
case("bin_ATE_minw", fb, "ATE", min.w = 0.1)
case("multi_ATE", fm, "ATE")
case("multi_ATE_standard", fm, "ATE", improved = FALSE)
case("multi_ATT", fm, "ATT", focal = "b")
case("multi_ATE_moments2_int", fm, "ATE", moments = 2, int = TRUE)
case("multi_ATT_moments1_lambda", fm, "ATT", focal = "c", moments = 1, lambda = 0.5)
# a flat objective: OSQP stops where the weights are not yet determined
case("bin_ATE_flat", bin ~ x1 + g2 + g3 + x3, "ATE", dist.mat = "euclidean")
write_json(out, "weightit_energy_R.json", digits = NA, auto_unbox = TRUE, null = "null")
