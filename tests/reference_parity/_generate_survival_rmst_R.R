#!/usr/bin/env Rscript
# Frozen-reference generator for sp.rmst and sp.kaplan_meier(se_method =
# "xie_liu"), on the committed _fixtures/survival_weighted_data.csv:
#   * survRM2::rmst2 (unweighted RMST per arm, difference, RMST / RMTL ratios)
#   * survival: survfit(weights =) rmean and the infinitesimal-jackknife SE
#     sqrt(sum((w * residuals(fit, times = tau, type = "rmst"))^2)), per row
#     and summed within cluster first
#   * adjustedCurves::adjustedsurv(method = "iptw_km") with the weights given
#     (Xie-Liu variance)
#
# survRM2 and adjustedCurves are not in the default library here; install
# them into a private one and point R_LIBS_USER at it:
#   R_LIBS_USER=~/.statspai-rlib-surv Rscript tests/reference_parity/_generate_survival_rmst_R.R

suppressPackageStartupMessages({
  library(survival)
  library(survRM2)
  library(adjustedCurves)
  library(jsonlite)
})

here <- function(p) file.path("tests", "reference_parity", p)
df <- read.csv(here("_fixtures/survival_weighted_data.csv"))
tau <- 8

r <- rmst2(df$time, df$event, df$a, tau = tau)

ij <- function(arm, weighted, clustered = FALSE) {
  dd <<- df[df$a == arm, ]
  s <- if (weighted) {
    survfit(Surv(time, event) ~ 1, data = dd, weights = w, model = TRUE)
  } else {
    survfit(Surv(time, event) ~ 1, data = dd, model = TRUE)
  }
  rr <- residuals(s, times = tau, type = "rmst")
  v <- (if (weighted) dd$w else rep(1, nrow(dd))) * rr
  if (clustered) v <- rowsum(v, dd$cl)
  list(rmst = unname(summary(s, rmean = tau)$table["rmean"]), se = sqrt(sum(v^2)))
}

df$a_f <- factor(df$a)
adj <- adjustedsurv(data = df, variable = "a_f", ev_time = "time", event = "event",
                    method = "iptw_km", treatment_model = df$w, conf_int = TRUE,
                    stabilize = FALSE)
xl <- adj$adj

out <- list(
  tau = tau,
  survrm2 = list(arm1 = unname(r$RMST.arm1$rmst), arm0 = unname(r$RMST.arm0$rmst),
                 unadjusted = unname(r$unadjusted.result)),
  ij = list(w1 = ij(1, TRUE), w0 = ij(0, TRUE), u1 = ij(1, FALSE),
            w1_cluster = ij(1, TRUE, TRUE)),
  xie_liu = list(time = xl$time, group = as.character(xl$group), surv = xl$surv,
                 se = xl$se),
  provenance = list(R = R.version.string,
                    survival = as.character(packageVersion("survival")),
                    survRM2 = as.character(packageVersion("survRM2")),
                    adjustedCurves = as.character(packageVersion("adjustedCurves")))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           here("_fixtures/survival_rmst_R.json"))
