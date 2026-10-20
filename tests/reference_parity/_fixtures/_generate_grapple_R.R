#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_grapple_R_parity.py
#
# Requires: R 4.5 + jsonlite, and GRAPPLE 0.2.2 (GitHub jingshuw/GRAPPLE,
# GPL (>= 2)). GRAPPLE cannot be installed (its haploR import is archived),
# so its estimator file R/mr_fun_general.R at commit 317e837 is downloaded to
# a temporary file and source()d. It is used strictly as a black box: only
# the exported grappleRobustEst() is called and only its return value is
# recorded; the file is not read or edited. Set GRAPPLE_SRC to a local copy
# of that file to skip the download.
#
# Run from any directory; writes grapple_sim.csv and grapple_R.json next to
# itself, numbers at digits = 17. Runtime: about a minute.
#
# Data (grapple_sim.csv, simulated here, set.seed(20261020))
# ----------------------------------------------------------
# * hetero_tau0 : 40 variants, heterogeneous SEs, no pleiotropy
# * hetero_tau  : 40 variants, heterogeneous SEs, balanced pleiotropy
# * outliers    : 40 variants, balanced pleiotropy + 4 gross outliers
# * no_xerror   : 30 variants, exposure SE 1e-7 (no measurement error)
# * weak        : 50 variants, weak instruments, equal SEs
#
# Recorded per scenario and loss ("tukey", "huber", "l2"):
# * fit   : grappleRobustEst(tau2 = NULL): beta.hat, tau2.hat, sqrt(beta.var),
#           tau2.se, beta.p.value;
# * fixed : grappleRobustEst(tau2 = t) for t in {0, 1e-4}: beta.hat and
#           sqrt(beta.var) (the variance formula evaluated at a given tau2).
# ---------------------------------------------------------------------------
suppressPackageStartupMessages(library(jsonlite))

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) {
  a <- commandArgs(FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  dirname(normalizePath(f))
})

gsrc <- Sys.getenv("GRAPPLE_SRC", "")
if (!nzchar(gsrc)) {
  url <- paste0("https://raw.githubusercontent.com/jingshuw/GRAPPLE/",
                "317e837340a29129c52c9d9b22f5f8bae72e27d0/R/mr_fun_general.R")
  gsrc <- tempfile(fileext = ".R")
  download.file(url, gsrc, quiet = TRUE)
}
source(gsrc)

set.seed(20261020)
sim <- function(id, m, gam, sx, sy, beta, tau, n_out = 0) {
  bx <- rnorm(m, gam, sx)
  by <- rnorm(m, beta * gam + rnorm(m, 0, tau), sy)
  if (n_out > 0) by[seq_len(n_out)] <- by[seq_len(n_out)] + 0.3
  data.frame(id = id, bx = bx, by = by, sx = sx, sy = sy)
}
m <- 40
d <- rbind(
  sim("hetero_tau0", m, runif(m, 0.05, 0.2), runif(m, 0.01, 0.03), runif(m, 0.01, 0.03), 0.5, 0),
  sim("hetero_tau", m, runif(m, 0.05, 0.2), runif(m, 0.01, 0.03), runif(m, 0.01, 0.03), 0.5, 0.02),
  sim("outliers", m, runif(m, 0.05, 0.2), runif(m, 0.01, 0.03), runif(m, 0.01, 0.03), 0.5, 0.02, 4),
  sim("no_xerror", 30, runif(30, 0.05, 0.2), rep(1e-7, 30), runif(30, 0.01, 0.03), 0.4, 0.02),
  sim("weak", 50, rnorm(50, 0, 0.012), rep(0.01, 50), rep(0.01, 50), 0.5, 0.004)
)
write.csv(d, file.path(here, "grapple_sim.csv"), row.names = FALSE)
d <- read.csv(file.path(here, "grapple_sim.csv"))  # both sides read the same bytes

out <- list(versions = list(R = R.version.string, GRAPPLE = "0.2.2 (317e837)"))
for (id in unique(d$id)) {
  s <- d[d$id == id, ]
  gd <- data.frame(SNP = paste0("v", seq_len(nrow(s))), gamma_exp1 = s$bx,
                   se_exp1 = s$sx, gamma_out1 = s$by, se_out1 = s$sy)
  for (loss in c("tukey", "huber", "l2")) {
    g <- suppressWarnings(grappleRobustEst(gd, loss.function = loss,
                                           diagnosis = FALSE, plot.it = FALSE))
    rec <- list(beta = unname(g$beta.hat), tau2 = g$tau2.hat,
                se = sqrt(unname(g$beta.var)), tau2_se = g$tau2.se,
                p = unname(g$beta.p.value), fixed = list())
    for (t2 in c(0, 1e-4)) {
      gf <- suppressWarnings(grappleRobustEst(gd, tau2 = t2, loss.function = loss,
                                              diagnosis = FALSE, plot.it = FALSE))
      rec$fixed[[length(rec$fixed) + 1]] <- list(tau2 = t2, beta = unname(gf$beta.hat),
                                                se = sqrt(unname(gf$beta.var)))
    }
    out[[id]][[loss]] <- rec
  }
}
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, pretty = TRUE),
           file.path(here, "grapple_R.json"))
