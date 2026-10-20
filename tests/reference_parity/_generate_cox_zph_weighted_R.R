#!/usr/bin/env Rscript
# Frozen reference for CoxResult.ph_test on weighted and counting-process
# fits: survival::cox.zph (score test, survival >= 3.0) on the fixtures of
# _generate_survival_weighted_R.R. survival is LGPL: used as a black box
# (documentation and outputs only).
#
# Regenerate with:
#   Rscript tests/reference_parity/_generate_cox_zph_weighted_R.R

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

here <- function(p) file.path("tests", "reference_parity", p)
df <- read.csv(here("_fixtures/survival_weighted_data.csv"))
cp <- read.csv(here("_fixtures/survival_counting_data.csv"))

tab <- function(z) {
  t <- as.data.frame(z$table)
  list(term = rownames(t), chisq = unname(t$chisq), df = unname(t$df),
       p = unname(t$p))
}
out <- list()
add <- function(tag, fit) {
  r <- list(coef = unname(coef(fit)))
  for (tr in c("km", "identity", "rank", "log")) {
    z <- cox.zph(fit, transform = tr)
    r[[tr]] <- tab(z)
    if (tr == "km") {
      r$km_x <- unname(z$x)
      r$km_time <- unname(z$time)
      r$km_y <- unname(as.matrix(z$y))
    }
  }
  out[[tag]] <<- r
}
for (ties in c("efron", "breslow")) {
  add(paste0("w_model_", ties), coxph(Surv(time, event) ~ x1 + x2 + a,
      data = df, weights = w, ties = ties, robust = FALSE))
  add(paste0("w_robust_", ties), coxph(Surv(time, event) ~ x1 + x2 + a,
      data = df, weights = w, ties = ties, robust = TRUE))
  add(paste0("entry_", ties), coxph(Surv(entry, time, event) ~ x1 + x2 + a,
      data = df, ties = ties))
  add(paste0("entry_w_", ties), coxph(Surv(entry, time, event) ~ x1 + x2 + a,
      data = df, ties = ties, weights = w, robust = TRUE, id = id))
  add(paste0("cp_id_", ties), coxph(Surv(tstart, tstop, event) ~ z + x2 + a,
      data = cp, ties = ties, cluster = id))
  add(paste0("cp_strata_", ties), coxph(Surv(tstart, tstop, event) ~ z + a +
      strata(x2), data = cp, ties = ties))
  add(paste0("cp_w_", ties), coxph(Surv(tstart, tstop, event) ~ z + x2 + a,
      data = cp, ties = ties, weights = w, cluster = id))
}
out$version <- as.character(packageVersion("survival"))
write_json(out, here("_fixtures/cox_zph_weighted_R.json"), digits = NA,
           auto_unbox = TRUE)
