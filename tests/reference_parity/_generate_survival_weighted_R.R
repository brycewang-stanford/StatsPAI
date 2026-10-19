#!/usr/bin/env Rscript
# Frozen-reference generator for weighted / counting-process sp.cox and
# sp.kaplan_meier (case weights, robust and clustered variances,
# Surv(start, stop, event), infinitesimal-jackknife KM variance).
#
# Inputs (committed): _fixtures/survival_weighted_data.csv (one row per
# subject, IPT weights `w`, integer times with ties, delayed entry `entry`,
# cluster `cl`) and _fixtures/survival_counting_data.csv (the same subjects
# split at t = 3 with a time-varying covariate `z`).
#
# Regenerate with:
#   Rscript tests/reference_parity/_generate_survival_weighted_R.R

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

here <- function(p) file.path("tests", "reference_parity", p)
df <- read.csv(here("_fixtures/survival_weighted_data.csv"))
cp <- read.csv(here("_fixtures/survival_counting_data.csv"))

cox <- list()
add <- function(tag, fit) {
  cox[[tag]] <<- list(
    coef = unname(coef(fit)), se = unname(sqrt(diag(vcov(fit)))),
    loglik = fit$loglik[2], concordance = unname(fit$concordance["concordance"])
  )
}
for (ties in c("efron", "breslow")) {
  add(paste0("w_model_", ties), coxph(Surv(time, event) ~ x1 + x2 + a, data = df,
      weights = w, ties = ties, robust = FALSE))
  add(paste0("w_robust_", ties), coxph(Surv(time, event) ~ x1 + x2 + a, data = df,
      weights = w, ties = ties, robust = TRUE))
  add(paste0("w_cluster_", ties), coxph(Surv(time, event) ~ x1 + x2 + a, data = df,
      weights = w, ties = ties, cluster = cl))
  add(paste0("entry_", ties), coxph(Surv(entry, time, event) ~ x1 + x2 + a,
      data = df, ties = ties))
  add(paste0("entry_w_robust_", ties), coxph(Surv(entry, time, event) ~ x1 + x2 + a,
      data = df, ties = ties, weights = w, robust = TRUE, id = id))
  add(paste0("cp_id_", ties), coxph(Surv(tstart, tstop, event) ~ z + x2 + a,
      data = cp, ties = ties, cluster = id))
  add(paste0("cp_strata_", ties), coxph(Surv(tstart, tstop, event) ~ z + a + strata(x2),
      data = cp, ties = ties))
  add(paste0("unw_robust_", ties), coxph(Surv(time, event) ~ x1 + x2 + a, data = df,
      ties = ties, robust = TRUE))
}

km <- list()
g <- function(tag, s) {
  ss <- summary(s, censored = TRUE)
  ev <- s$n.event > 0
  km[[tag]] <<- list(time = s$time[ev], surv = s$surv[ev], se = ss$std.err[ev],
                     n_risk = s$n.risk[ev], lower = s$lower[ev], upper = s$upper[ev])
}
g("w_ij", survfit(Surv(time, event) ~ 1, data = df[df$a == 1, ], weights = w,
                  conf.type = "log-log"))
g("w_greenwood", survfit(Surv(time, event) ~ 1, data = df[df$a == 1, ], weights = w,
                         robust = FALSE, conf.type = "log"))
g("unw_ij", survfit(Surv(time, event) ~ 1, data = df, robust = TRUE, id = id,
                    conf.type = "plain"))
g("entry_greenwood", survfit(Surv(entry, time, event) ~ 1, data = df, conf.type = "log"))
g("cp_cluster", survfit(Surv(tstart, tstop, event) ~ 1, data = cp, weights = w,
                        cluster = id, conf.type = "log-log"))
g("w_cluster", survfit(Surv(time, event) ~ 1, data = df, weights = w, cluster = cl,
                       conf.type = "log-log"))

out <- list(
  cox = cox, km = km,
  provenance = list(R = R.version.string,
                    survival = as.character(packageVersion("survival")))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           here("_fixtures/survival_weighted_R.json"))
