#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_weakiv_meta_parity.py
#
# Requires: R 4.5 + ivmodel + car + metafor.
# Run:      Rscript _generate_weakiv_R.R    (from this directory)
#
# Three references, three reasons
# ------------------------------
# * ivmodel::AR.test  — the Anderson-Rubin statistic AND its confidence set.
#   The set matters on its own: StatsPAI computes it twice, analytically
#   inside `sp.anderson_rubin_test` and by grid inversion in
#   `sp.anderson_rubin_ci`, and until 1.27.0 the two disagreed by ~8e-3
#   because the grid version reported the extreme grid point still inside
#   the set rather than the boundary.
# * ivmodel::CLR — Moreira's conditional likelihood ratio set. ivmodel
#   integrates the conditional distribution; StatsPAI simulates it, so this
#   row is Monte Carlo (T3) by construction and the test asserts the error
#   SHRINKS with n_sim rather than pinning a tolerance.
# * car::vif and metafor::rma — deterministic, pinned at machine level.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(ivmodel); library(car); library(metafor); library(jsonlite)
})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

d <- read.csv(file.path(OUT, "weakiv_data.csv"))
iv <- ivmodel(Y = d$y, D = d$d, Z = cbind(d$z1, d$z2), X = matrix(d$w, ncol = 1))
ar <- AR.test(iv, alpha = 0.05)
clr <- CLR(iv, alpha = 0.05)

out <- list(
  AR = list(Fstat = unname(ar$Fstat[1]), p = unname(ar$p.value),
            df1 = unname(ar$df[1]), df2 = unname(ar$df[2]),
            ci_lower = as.numeric(ar$ci)[1], ci_upper = as.numeric(ar$ci)[2]),
  CLR = list(p = as.numeric(clr$p.value)[1],
             ci_lower = as.numeric(clr$ci)[1], ci_upper = as.numeric(clr$ci)[2]),
  vif = as.list(car::vif(lm(y ~ d + z1 + z2 + w, data = d)))  # car:: — ivmodel masks vif
)

m <- read.csv(file.path(OUT, "meta_data.csv"))
g <- function(fit) list(b = unname(fit$beta[1]), se = unname(fit$se),
                        ci_lb = unname(fit$ci.lb), ci_ub = unname(fit$ci.ub),
                        tau2 = unname(fit$tau2), QE = unname(fit$QE),
                        QEp = unname(fit$QEp), I2 = unname(fit$I2),
                        H2 = unname(fit$H2))
out$meta_FE <- g(rma(yi = m$eff, sei = m$se, method = "FE"))
out$meta_DL <- g(rma(yi = m$eff, sei = m$se, method = "DL"))

out$provenance <- list(r = R.version.string,
                       ivmodel = as.character(packageVersion("ivmodel")),
                       car = as.character(packageVersion("car")),
                       metafor = as.character(packageVersion("metafor")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           file.path(OUT, "weakiv_R.json"))
cat("wrote weakiv_R.json\n")
