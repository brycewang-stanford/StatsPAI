#!/usr/bin/env Rscript
# Frozen R references for the sp.breakdown_m part of
# tests/reference_parity/test_did_synth_R_parity.py.
#
#   HonestDiD 0.2.8 (Rambachan & Roth's own package), findOptimalFLCI: the
#   fixed-length confidence interval under Delta^SD(M) at a grid of M, and the
#   breakdown value M* = sup{M : 0 not in FLCI(M)}, found by uniroot() on the
#   CI bound that faces zero (HonestDiD exports no breakdown function; the
#   package's own documentation reads the breakdown off the sensitivity grid,
#   and uniroot is that same definition solved to tolerance instead of read
#   off a grid).
#
# Every quantity is computed twice:
#   default   HonestDiD as shipped. Its folded-normal quantile .qfoldednormal
#             is a Monte-Carlo estimate (10^6 normal draws at seed 0), which
#             is ~1e-3 off the exact quantile (1.96224 vs 1.95996 at mu = 0).
#   exactq    the same code with .qfoldednormal replaced -- via
#             assignInNamespace -- by the exact inverse of the folded-normal
#             CDF. Nothing else changes. This isolates the Monte-Carlo
#             quantile as the mechanism of the default-run gap.
#
# Input: _fixtures/did_synth_honest_es.csv (hand-crafted; written by
# _fixtures/_generate_did_synth_data.py): 3 pre- and 3 post-period
# coefficients (reference period omitted) and their covariance.
#
# Regenerate:  Rscript tests/reference_parity/_generate_did_synth_honest_R.R

suppressPackageStartupMessages({
  library(HonestDiD)
  library(jsonlite)
})

script_dir <- tryCatch(
  dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")

es <- read.csv(file.path(fixtures, "did_synth_honest_es.csv"))
b <- es$betahat
S <- as.matrix(es[, paste0("s", 0:5)])
NPRE <- 3L; NPOST <- 3L
M_GRID <- c(0, 0.005, 0.01, 0.02, 0.05)
ALPHA <- 0.05

flci <- function(M, l_vec) {
  suppressWarnings(findOptimalFLCI(b, S, M = M, numPrePeriods = NPRE,
                                   numPostPeriods = NPOST, l_vec = l_vec,
                                   alpha = ALPHA)$FLCI)
}

breakdown <- function(l_vec) {
  f <- function(M) { ci <- flci(M, l_vec); max(ci[1], -ci[2]) }
  if (f(0) <= 0) return(0)
  hi <- 0.01
  while (f(hi) > 0) hi <- 2 * hi
  uniroot(f, c(0, hi), tol = 1e-12)$root
}

run <- function() {
  out <- list()
  for (e in 0:2) {
    l_vec <- rep(0, NPOST); l_vec[e + 1] <- 1
    ci <- t(vapply(M_GRID, function(M) flci(M, l_vec), numeric(2)))
    out[[paste0("e", e)]] <- list(
      M = M_GRID, ci_lower = ci[, 1], ci_upper = ci[, 2],
      breakdown = breakdown(l_vec)
    )
  }
  out
}

default <- run()

exact_qfoldednormal <- function(p, mu = 0, sd = 1, numSims = 10^6, seed = 0) {
  vapply(mu, function(m) {
    m <- abs(m)
    uniroot(function(q) pnorm((q - m) / sd) - pnorm((-q - m) / sd) - p,
            c(0, m + 20 * sd), tol = 1e-15)$root
  }, numeric(1))
}
assignInNamespace(".qfoldednormal", exact_qfoldednormal, "HonestDiD")
exactq <- run()

res <- list(
  provenance = list(
    R = R.version.string,
    HonestDiD = as.character(packageVersion("HonestDiD")),
    CVXR = as.character(packageVersion("CVXR")),
    jsonlite = as.character(packageVersion("jsonlite")),
    alpha = ALPHA, n_pre = NPRE, n_post = NPOST
  ),
  default = default,
  exactq = exactq
)

writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(fixtures, "did_synth_honest_R.json"))
cat("wrote", file.path(fixtures, "did_synth_honest_R.json"), "\n")
