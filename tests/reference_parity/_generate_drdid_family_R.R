#!/usr/bin/env Rscript
# Reference values for the COMPLETE R DRDID 1.2.3 estimator family.
#
# DRDID exports 14 estimator functions. The Python port most people reach
# for (d2cml-ai/DRDIDpy) ships 6 of them and returns bare
# `(att, influence_function)` tuples with no inference. StatsPAI reaches
# all 14 through one entry point, `sp.drdid`, and this script pins every
# one of them:
#
#   panel                     repeated cross-sections
#   ------------------------  --------------------------
#   drdid_panel               drdid_rc
#   drdid_imp_panel           drdid_imp_rc
#                             drdid_rc1
#                             drdid_imp_rc1
#   std_ipw_did_panel         std_ipw_did_rc
#   ipw_did_panel             ipw_did_rc
#   reg_did_panel             reg_did_rc
#   twfe_did_panel            twfe_did_rc
#
# Writes:
#   _fixtures/drdid_family_long.csv       800 units x 2 periods, long form
#   _fixtures/drdid_family_reference.csv  one row per estimator
#
# The design has a real propensity-score gradient in x1/x2 and an ATT of
# 1.5, so the estimators genuinely disagree: the reference spread runs
# from 1.451 (ipw_did_*) to 1.697 (twfe_did_*). A port that silently
# collapsed two variants onto one estimator could not pass.
#
# R 4.x, DRDID 1.2.3.

suppressPackageStartupMessages(library(DRDID))
set.seed(99)

fixtures <- "tests/reference_parity/_fixtures"
if (!dir.exists(fixtures)) stop("run from the repository root")

n <- 800
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4)
ps <- plogis(-0.3 + 0.6 * x1 + 0.4 * x2)
D <- rbinom(n, 1, ps)
y0 <- 1 + 0.8 * x1 + 0.5 * x2 + rnorm(n)
y1 <- y0 + 0.4 + 1.5 * D + 0.3 * x1 + rnorm(n, sd = 0.7)

long <- rbind(
  data.frame(id = 1:n, post = 0, D = D, x1 = x1, x2 = x2, y = y0),
  data.frame(id = 1:n, post = 1, D = D, x1 = x1, x2 = x2, y = y1)
)
write.csv(long, file.path(fixtures, "drdid_family_long.csv"), row.names = FALSE)

cov <- cbind(1, x1, x2)
covl <- cbind(1, long$x1, long$x2)
out <- list()
add <- function(nm, r) {
  out[[length(out) + 1]] <<- data.frame(
    fn = nm, att = r$ATT, se = r$se, lci = r$lci, uci = r$uci
  )
}

# ---- panel ----------------------------------------------------------
add("drdid_panel", drdid_panel(y1, y0, D, cov, inffunc = TRUE))
add("drdid_imp_panel", drdid_imp_panel(y1, y0, D, cov, inffunc = TRUE))
add("std_ipw_did_panel", std_ipw_did_panel(y1, y0, D, cov, inffunc = TRUE))
add("ipw_did_panel", ipw_did_panel(y1, y0, D, cov, inffunc = TRUE))
add("reg_did_panel", reg_did_panel(y1, y0, D, cov, inffunc = TRUE))
add("twfe_did_panel", twfe_did_panel(y1, y0, D, cov))

# ---- repeated cross-sections ----------------------------------------
add("drdid_rc", drdid_rc(long$y, long$post, long$D, covl, inffunc = TRUE))
add("drdid_imp_rc", drdid_imp_rc(long$y, long$post, long$D, covl, inffunc = TRUE))
add("drdid_rc1", drdid_rc1(long$y, long$post, long$D, covl, inffunc = TRUE))
add("drdid_imp_rc1", drdid_imp_rc1(long$y, long$post, long$D, covl, inffunc = TRUE))
add("std_ipw_did_rc", std_ipw_did_rc(long$y, long$post, long$D, covl, inffunc = TRUE))
add("ipw_did_rc", ipw_did_rc(long$y, long$post, long$D, covl, inffunc = TRUE))
add("reg_did_rc", reg_did_rc(long$y, long$post, long$D, covl, inffunc = TRUE))
add("twfe_did_rc", twfe_did_rc(long$y, long$post, long$D, covl))

reference <- do.call(rbind, out)
rownames(reference) <- NULL
write.csv(reference, file.path(fixtures, "drdid_family_reference.csv"),
          row.names = FALSE)
cat("wrote", nrow(reference), "reference rows\n")
print(reference, digits = 10)
