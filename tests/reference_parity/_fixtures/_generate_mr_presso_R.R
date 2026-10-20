#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_mr_presso_R_parity.py
#
# Requires: R 4.5 + MRPRESSO 1.0 (GitHub rondolab/MR-PRESSO, GPL-3) and
# jsonlite. MRPRESSO is run strictly as a black box: only its exported
# mr_presso() is called and only its return value is recorded.
# Run from any directory; writes mr_presso_sim.csv and mr_presso_R.json next
# to itself. Runtime: about 10-40 minutes (MRPRESSO refits every
# leave-one-out IVW regression in every simulated draw).
#
# Data
# ----
# * mr_ldl.csv: the 28-variant LDL-C -> CHD summary statistics written by
#   _generate_mr_R.R (already committed; read, not rewritten).
# * mr_presso_sim.csv: 15 simulated variants written by this script, with
#   a clean outcome column (by_clean) and the same column with two gross
#   pleiotropic variants (by_out).
#
# What is deterministic and what is not
# -------------------------------------
# Deterministic (pinned tightly by the test): the raw and outlier-corrected
# IVW fits (estimate, Sd, T-stat, P-value), the observed RSS (global and per
# variant), the outlier set where every flagged variant has a Monte Carlo
# p-value of 0, and the distortion coefficient.
# Monte Carlo (stochastic screen only): the global, outlier and distortion
# p-values. Each scenario is re-run under several seeds and every p-value is
# recorded, so the test can compare the two simulation distributions rather
# than single draws. p-values that MRPRESSO prints as "<x" are recorded as 0
# (the k / B convention with k = 0) and flagged.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(MRPRESSO); library(jsonlite) })
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

ldl <- read.csv(file.path(OUT, "mr_ldl.csv"))

set.seed(20261020)
J <- 15
sim <- data.frame(bx = runif(J, 0.05, 0.20), bxse = runif(J, 0.005, 0.015),
                  byse = runif(J, 0.010, 0.030))
sim$by_clean <- 0.4 * sim$bx + rnorm(J, 0, sim$byse)
sim$by_out <- sim$by_clean
sim$by_out[c(1, 2)] <- sim$by_out[c(1, 2)] + c(0.15, -0.12)
write.csv(sim, file.path(OUT, "mr_presso_sim.csv"), row.names = FALSE)

num_p <- function(x) {
  x <- as.character(x)
  lt <- grepl("^<", x)
  v <- suppressWarnings(as.numeric(ifelse(lt, "0", x)))
  list(value = v, below = lt)
}

run <- function(d, by, seed, B) {
  set.seed(seed)
  pr <- mr_presso(BetaOutcome = by, BetaExposure = "bx", SdOutcome = "byse",
                  SdExposure = "bxse", OUTLIERtest = TRUE, DISTORTIONtest = TRUE,
                  data = d, NbDistribution = B, SignifThreshold = 0.05)
  m <- pr$`Main MR results`
  res <- pr$`MR-PRESSO results`
  gp <- num_p(res$`Global Test`$Pvalue)
  o <- list(seed = seed, B = B, components = names(res),
            raw = list(est = m$`Causal Estimate`[1], sd = m$Sd[1],
                       t = m$`T-stat`[1], p = m$`P-value`[1]),
            cor = list(est = m$`Causal Estimate`[2], sd = m$Sd[2],
                       t = m$`T-stat`[2], p = m$`P-value`[2]),
            rss_obs = as.numeric(res$`Global Test`$RSSobs),
            global_p = gp$value, global_p_below = gp$below)
  if (!is.null(res$`Outlier Test`)) {
    op <- num_p(res$`Outlier Test`$Pvalue)
    o$rss_obs_j <- as.numeric(res$`Outlier Test`$RSSobs)
    o$outlier_p <- op$value
    o$outlier_p_below <- op$below
  }
  if (!is.null(res$`Distortion Test`)) {
    dt <- res$`Distortion Test`
    o$outliers <- as.integer(dt$`Outliers Indices`)
    o$distortion_coef <- unname(as.numeric(dt$`Distortion Coefficient`))
    o$distortion_p <- as.numeric(dt$Pvalue)
  }
  o
}

scen <- list(ldl = list(d = ldl, by = "by"),
             sim_out = list(d = sim, by = "by_out"),
             sim_clean = list(d = sim, by = "by_clean"))
SEEDS <- 1:6
B <- 1000
out <- list()
for (nm in names(scen)) {
  s <- scen[[nm]]
  out[[nm]] <- lapply(SEEDS, function(sd) run(s$d, s$by, sd, B))
  cat("done", nm, "\n")
}
out$provenance <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  MRPRESSO = as.character(packageVersion("MRPRESSO")),
  license = packageDescription("MRPRESSO")$License,
  seeds = SEEDS, NbDistribution = B, SignifThreshold = 0.05)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"),
           file.path(OUT, "mr_presso_R.json"))
cat("wrote mr_presso_R.json\n")
