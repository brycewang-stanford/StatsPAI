#!/usr/bin/env Rscript
# Frozen-reference generator for sp.msm(censor=): ipw::ipwtm (1.3.0)
# treatment weights (type = "all") times censoring weights (exposure = the
# censoring indicator, family = "binomial", type = "first"), then the
# weighted lm of the MSM with sandwich::vcovCL(cluster = ~id, type = "HC1").
#
# Input: _fixtures/msm_censoring_data.csv, one row per (id, t) up to and
# including the period of censoring (cens = 1, a and y missing there).
#
# Regenerate with:
#   Rscript tests/reference_parity/_generate_msm_censoring_R.R

suppressPackageStartupMessages({
  library(ipw)
  library(sandwich)
  library(jsonlite)
})

here <- function(p) file.path("tests", "reference_parity", p)
full <- read.csv(here("_fixtures/msm_censoring_data.csv"))
full <- full[order(full$id, full$t), ]
lag0 <- function(x) c(0, head(x, -1))
full$a_lag <- ave(ifelse(is.na(full$a), 0, full$a), full$id, FUN = lag0)

wc <- ipwtm(exposure = cens, family = "binomial", link = "logit",
            numerator = ~ a_lag + v, denominator = ~ a_lag + v + l,
            id = id, timevar = t, type = "first", data = full)
full$wc <- wc$ipw.weights

unc <- full[full$cens == 0, ]
wt <- ipwtm(exposure = a, family = "binomial", link = "logit",
            numerator = ~ a_lag + v, denominator = ~ a_lag + v + l,
            id = id, timevar = t, type = "all", data = unc)
unc$wt <- wt$ipw.weights
unc$w <- unc$wt * unc$wc
unc$cum_a <- ave(unc$a, unc$id, FUN = cumsum)

fit <- lm(y ~ cum_a + v, data = unc, weights = w)
fit_nc <- lm(y ~ cum_a + v, data = unc, weights = wt)
out <- list(
  censoring_weights = unc$wc,
  treatment_weights = unc$wt,
  coef = unname(coef(fit)),
  se = unname(sqrt(diag(vcovCL(fit, cluster = ~id, type = "HC1")))),
  coef_no_censoring_weights = unname(coef(fit_nc)),
  provenance = list(R = R.version.string,
                    ipw = as.character(packageVersion("ipw")),
                    sandwich = as.character(packageVersion("sandwich")))
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE),
           here("_fixtures/msm_censoring_R.json"))
