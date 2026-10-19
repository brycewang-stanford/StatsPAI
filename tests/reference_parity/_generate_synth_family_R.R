#!/usr/bin/env Rscript
# Frozen R references for tests/reference_parity/test_synth_family_R_parity.py
# (wave-2 synth-family work, 2026-10-19).
#
#   sp.augsynth vs augsynth::augsynth (0.2.0, the Ben-Michael / Feller /
#       Rothstein package): several treated units sharing one date
#       (single_augsynth averages them), progfunc = "Ridge" / "None",
#       scm = TRUE / FALSE, fixedeff = TRUE, auxiliary covariates with the
#       default residualize = FALSE, ridge lambda fixed and by augsynth's own
#       time-holdout CV. augsynth's synth_qp hard-codes OSQP eps 1e-8; the
#       generator swaps in a copy with eps 1e-12 (assignInNamespace), so the
#       simplex weights are exact to ~1e-11 and the comparison can be tight.
#
#   sp.fect's diagnostic tests vs fect::fect (2.4.1, Liu / Wang / Xu):
#       placeboTest (placebo.period = c(-2, 0)), carryoverTest
#       (carryover.period = c(1, 2)) and loo = TRUE, method = "fe",
#       vartype = "jackknife" (leave one unit out: deterministic, so the
#       comparison is exact), tol 1e-12. loo with the jackknife needs
#       nboots = N (fect sizes pre.att.boot by nboots). test.out /
#       loo.test.out are fect:::diagtest, including its F statistic built
#       from cov() of the raw jackknife replicates.
#
# Regenerate (after _generate_synth_family_data.py), from the repo root:
#   Rscript tests/reference_parity/_generate_synth_family_R.R

suppressPackageStartupMessages({
  library(augsynth)
  library(fect)
  library(jsonlite)
})

fixtures <- file.path("tests", "reference_parity", "_fixtures")
num <- function(x) as.numeric(x)

tight_qp <- function(X1, X0, V) {
  Pmat <- X0 %*% V %*% t(X0)
  qvec <- -t(X1) %*% V %*% t(X0)
  n0 <- nrow(X0)
  A <- rbind(rep(1, n0), diag(n0))
  l <- c(1, numeric(n0))
  u <- c(1, rep(1, n0))
  settings <- osqp::osqpSettings(verbose = FALSE, eps_rel = 1e-12, eps_abs = 1e-12,
                                 max_iter = 200000L, polishing = TRUE)
  osqp::solve_osqp(P = Pmat, q = qvec, A = A, l = l, u = u, pars = settings)$x
}
assignInNamespace("synth_qp", tight_qp, ns = "augsynth")

# ---------------------------------------------------------------------------
# 1. augsynth
# ---------------------------------------------------------------------------
asc <- read.csv(file.path(fixtures, "synth_family_asc.csv"))
asc_one <- asc[!(asc$unit == 2), ]  # single treated unit (unit 1)

run_asc <- function(name, data, cov = FALSE, ...) {
  form <- if (cov) y ~ treated | x1 + x2 else y ~ treated
  fit <- suppressMessages(augsynth(form, unit, time, data = data, ...))
  att <- predict(fit, att = TRUE)
  tt <- as.numeric(names(att))
  list(name = name,
       att = mean(att[tt >= 14]),
       att_path = num(att),
       weights = num(fit$weights),
       donors = as.integer(rownames(fit$weights)),
       lambda = if (is.null(fit$lambda)) NA_real_ else num(fit$lambda))
}

augsynth_cases <- list(
  run_asc("two_ridge_fixed", asc, progfunc = "Ridge", scm = TRUE, lambda = 0.5),
  run_asc("two_ridge_cv", asc, progfunc = "Ridge", scm = TRUE),
  run_asc("two_none", asc, progfunc = "None", scm = TRUE),
  run_asc("one_fixedeff_cv", asc_one, progfunc = "Ridge", scm = TRUE, fixedeff = TRUE),
  run_asc("one_noscm_cv", asc_one, progfunc = "Ridge", scm = FALSE),
  run_asc("one_cov_fixed", asc_one, cov = TRUE, progfunc = "Ridge", scm = TRUE, lambda = 2),
  run_asc("two_cov_cv", asc, cov = TRUE, progfunc = "Ridge", scm = TRUE),
  run_asc("two_cov_none_fe", asc, cov = TRUE, progfunc = "None", scm = TRUE, fixedeff = TRUE)
)

# ---------------------------------------------------------------------------
# 2. fect diagnostic tests
# ---------------------------------------------------------------------------
fd <- read.csv(file.path(fixtures, "synth_family_fect.csv"))
run_fect <- function(...) {
  suppressMessages(suppressWarnings(fect(
    Y ~ D, data = fd, index = c("id", "time"), method = "fe", force = "two-way",
    se = TRUE, vartype = "jackknife", tol = 1e-12, max.iteration = 20000,
    parallel = FALSE, ...)))
}
row5 <- function(m) list(estimate = num(m[1]), se = num(m[2]), ci_lower = num(m[3]),
                         ci_upper = num(m[4]), pvalue = num(m[5]))
tests <- function(x) lapply(x, function(v) if (is.null(v)) NA_real_ else num(v))

fp <- run_fect(placeboTest = TRUE, placebo.period = c(-2, 0))
fc <- run_fect(carryoverTest = TRUE, carryover.period = c(1, 2))
fl <- run_fect(loo = TRUE, nboots = length(unique(fd$id)))
fect_out <- list(
  placebo = list(att_avg = num(fp$att.avg), test = row5(fp$est.placebo),
                 test_out = tests(fp$test.out), sigma2_fect = num(fp$sigma2.fect)),
  carryover = list(att_avg = num(fc$att.avg), test = row5(fc$est.carryover),
                   test_out = tests(fc$test.out), sigma2_fect = num(fc$sigma2.fect)),
  loo = list(att_avg = num(fl$att.avg),
             time = num(fl$time), att = num(fl$est.att[, 1]), se = num(fl$est.att[, 2]),
             pre_period = num(rownames(fl$pre.est.att)),
             pre_att = num(fl$pre.est.att[, 1]), pre_se = num(fl$pre.est.att[, 2]),
             pre_p = num(fl$pre.est.att[, 5]), pre_count = num(fl$pre.est.att[, 6]),
             test_out = tests(fl$test.out), loo_test_out = tests(fl$loo.test.out),
             n_units = length(unique(fd$id)))
)

out <- list(
  meta = list(R = R.version.string,
              augsynth = as.character(packageVersion("augsynth")),
              fect = as.character(packageVersion("fect")),
              osqp = as.character(packageVersion("osqp"))),
  augsynth = augsynth_cases,
  fect = fect_out
)
write_json(out, file.path(fixtures, "synth_family_R.json"), auto_unbox = TRUE,
           digits = NA, null = "null", na = "null")
cat("wrote", file.path(fixtures, "synth_family_R.json"), "\n")
