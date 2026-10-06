# Reference values for tests/reference_parity/test_tmle_parameters_R_parity.py
#
# tmle::tmle on _fixtures/tmle_design_data.csv with the initial fits
# supplied (Q and g1W from stats::glm, written to
# _fixtures/tmle_parameters_nuisance.csv so both sides target the same
# numbers). Records every parameter tmle reports: the treatment-specific
# means, the additive effect, the risk ratio and odds ratio (binary
# outcome), and the effects among the treated and the controls.
#
# Run from tests/reference_parity:  Rscript _generate_tmle_parameters_R.R
suppressMessages({library(tmle); library(jsonlite)})
d <- read.csv("_fixtures/tmle_design_data.csv")
W <- d[, c("x1", "x2", "x3")]
g1 <- as.numeric(predict(glm(d ~ x1 + x2 + x3, data = d, family = binomial),
                         type = "response"))
nuis <- data.frame(g = g1)
out <- list(tmle_version = as.character(packageVersion("tmle")),
            r_version = R.version.string)
pack <- function(e) {
  x <- list(psi = e$psi)
  if (!is.null(e$var.psi)) x$var <- e$var.psi
  if (!is.null(e$var.log.psi)) x$var_log <- e$var.log.psi
  x$ci <- e$CI
  x$pvalue <- e$pvalue
  x
}
for (fam in c("gaussian", "binomial")) {
  yv <- if (fam == "gaussian") "y" else "yb"
  qf <- glm(as.formula(paste(yv, "~ d + x1 + x2 + x3")), data = d, family = fam)
  Q <- cbind(predict(qf, transform(d, d = 0), type = "response"),
             predict(qf, transform(d, d = 1), type = "response"))
  nuis[[paste0("Q0_", yv)]] <- Q[, 1]
  nuis[[paste0("Q1_", yv)]] <- Q[, 2]
  cases <- list(plain = list(), cluster = list(id = d$g),
                weights = list(obsWeights = d$w),
                weights_cluster = list(obsWeights = d$w, id = d$g))
  for (cs in names(cases)) {
    f <- do.call(tmle, c(list(Y = d[[yv]], A = d$d, W = W, Q = Q, g1W = g1,
                              family = fam, gbound = 0.025), cases[[cs]]))
    e <- f$estimates
    rec <- list()
    for (k in c("EY1", "EY0", "ATE", "RR", "OR"))
      if (!is.null(e[[k]])) rec[[k]] <- pack(e[[k]])
    if (cs == "plain") {
      for (k in c("ATT", "ATC")) {
        rec[[k]] <- pack(e[[k]])
        ic <- e$IC[[paste0("IC.", k)]]
        rec[[k]]$n_ic <- length(ic)
        # tmle adds min(Y) to the influence curve of a continuous outcome;
        # remove it so the mean is the unsolved part of the EIF equation.
        shift <- if (fam == "gaussian") min(d[[yv]]) else 0
        rec[[k]]$ic_mean <- mean(ic) - shift
        rec[[k]]$converged <- e[[k]]$converged
      }
    }
    out[[paste(fam, cs, sep = "_")]] <- rec
  }
}
write.csv(nuis, "_fixtures/tmle_parameters_nuisance.csv", row.names = FALSE)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "_fixtures/tmle_parameters_R.json")
cat("wrote _fixtures/tmle_parameters_R.json\n")
