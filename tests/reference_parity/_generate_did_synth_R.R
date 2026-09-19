#!/usr/bin/env Rscript
# Frozen R references for tests/reference_parity/test_did_synth_R_parity.py
# (did_synth family of the phase-3 cross-language campaign).
#
#   synthdid 0.0.9 (Arkhangelsky, Athey, Hirshberg, Imbens & Wager's package)
#     synthdid_estimate / sc_estimate / did_estimate, unit + time weights,
#     jackknife SE, and the placebo / bootstrap SEs *draw for draw*.
#
# The placebo and bootstrap standard errors are Monte-Carlo quantities drawn
# from R's Mersenne-Twister stream, which no other language reproduces. They
# are pinned in two pieces, as the staggered-rollout suite does:
#   1. every draw's index vector (the permutation / bootstrap sample R drew)
#      is written out together with the estimate R computed on it, so the
#      Python side can replay the SAME draws and must agree per draw;
#   2. the end-to-end synthdid_se(method = "placebo", replications = 200) value
#      is recorded for a Monte-Carlo-error comparison only (T3).
# The per-draw `theta` closures below are copied verbatim from synthdid
# 0.0.9's placebo_se / bootstrap_sample (print(synthdid:::placebo_se)), with
# the random index injected instead of drawn inside.
#
# Regenerate (after _fixtures/_generate_did_synth_data.py):
#   Rscript tests/reference_parity/_generate_did_synth_R.R

suppressPackageStartupMessages({
  library(synthdid)
  library(jsonlite)
})

script_dir <- tryCatch(
  dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")
repo <- normalizePath(file.path(script_dir, "..", ".."))

N_DRAWS <- 40L

placebo_theta <- function(estimate, ind) {
  setup <- attr(estimate, "setup"); opts <- attr(estimate, "opts")
  weights <- attr(estimate, "weights")
  N1 <- nrow(setup$Y) - setup$N0
  N0 <- length(ind) - N1
  weights.boot <- weights
  weights.boot$omega <- synthdid:::sum_normalize(weights$omega[ind[1:N0]])
  as.numeric(do.call(synthdid_estimate, c(list(
    Y = setup$Y[ind, ], N0 = N0, T0 = setup$T0, X = setup$X[ind, , ],
    weights = weights.boot), opts)))
}

bootstrap_theta <- function(estimate, ind) {
  setup <- attr(estimate, "setup"); opts <- attr(estimate, "opts")
  weights <- attr(estimate, "weights")
  if (all(ind <= setup$N0) || all(ind > setup$N0)) return(NA_real_)
  weights.boot <- weights
  weights.boot$omega <- synthdid:::sum_normalize(weights$omega[sort(ind[ind <= setup$N0])])
  as.numeric(do.call(synthdid_estimate, c(list(
    Y = setup$Y[sort(ind), ], N0 = sum(ind <= setup$N0), T0 = setup$T0,
    X = setup$X[sort(ind), , ], weights = weights.boot), opts)))
}

one_method <- function(panel, method, seed) {
  est <- switch(method,
    sdid = synthdid_estimate(panel$Y, panel$N0, panel$T0),
    sc = sc_estimate(panel$Y, panel$N0, panel$T0),
    did = did_estimate(panel$Y, panel$N0, panel$T0))
  w <- attr(est, "weights"); opts <- attr(est, "opts")
  N0 <- panel$N0; N <- nrow(panel$Y)

  jk <- as.numeric(synthdid_se(est, method = "jackknife"))

  # Placebo: R draws sample(1:N0) per replication. Record the draws.
  set.seed(seed)
  pl_ind <- replicate(N_DRAWS, sample(1:N0), simplify = FALSE)
  pl_theta <- vapply(pl_ind, function(ind) placebo_theta(est, ind), numeric(1))
  pl_se_draws <- sqrt((N_DRAWS - 1) / N_DRAWS) * sd(pl_theta)

  # Bootstrap: sample(1:N, replace = TRUE), rejecting all-treated/all-control.
  # synthdid's bootstrap_sample returns NA outright with one treated unit
  # (N0 == N - 1), so no draws are recorded then.
  set.seed(seed + 1L)
  bs_ind <- list(); bs_theta <- numeric(0)
  while (N0 < N - 1 && length(bs_theta) < N_DRAWS) {
    ind <- sample(1:N, replace = TRUE)
    th <- bootstrap_theta(est, ind)
    if (!is.na(th)) { bs_ind[[length(bs_ind) + 1L]] <- ind; bs_theta <- c(bs_theta, th) }
  }
  bs_se_draws <- if (length(bs_theta) > 1) sqrt((N_DRAWS - 1) / N_DRAWS) * sd(bs_theta) else NA

  # End-to-end synthdid_se (Monte-Carlo; compared within simulation error).
  set.seed(seed + 2L)
  pl_se_200 <- as.numeric(synthdid_se(est, method = "placebo", replications = 200))

  list(
    estimate = as.numeric(est),
    omega = unname(as.numeric(w$omega)),
    lambda = unname(as.numeric(w$lambda)),
    zeta_omega = opts$zeta.omega, zeta_lambda = opts$zeta.lambda,
    min_decrease = opts$min.decrease,
    se_jackknife = jk,
    placebo_ind = pl_ind, placebo_theta = pl_theta, placebo_se_draws = pl_se_draws,
    bootstrap_ind = bs_ind, bootstrap_theta = bs_theta, bootstrap_se_draws = bs_se_draws,
    placebo_se_200 = pl_se_200
  )
}

one_dataset <- function(path, unit, time, outcome, treatment, seed) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  panel <- panel.matrices(df, unit = unit, time = time, outcome = outcome,
                          treatment = treatment)
  out <- list(
    units = rownames(panel$Y), times = as.numeric(colnames(panel$Y)),
    N0 = panel$N0, T0 = panel$T0
  )
  for (m in c("sdid", "sc", "did")) out[[m]] <- one_method(panel, m, seed)
  out
}

res <- list(
  provenance = list(
    R = R.version.string,
    synthdid = as.character(packageVersion("synthdid")),
    jsonlite = as.character(packageVersion("jsonlite")),
    n_draws = N_DRAWS
  ),
  prop99 = one_dataset(file.path(repo, "tests", "r_parity", "data", "12_sdid.csv"),
                       "state", "year", "cigsale", "treated", 42L),
  panel = one_dataset(file.path(fixtures, "did_synth_sdid_panel.csv"),
                      "unit", "year", "y", "treated", 7L)
)

writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, na = "null", pretty = FALSE),
           file.path(fixtures, "did_synth_R.json"))
cat("wrote", file.path(fixtures, "did_synth_R.json"), "\n")
