#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_medsens_R_parity.py
# (sp.mediate_sensitivity), extending the single configuration in
# misc_sens_R.json. mediation (GPL >= 2) is run as a black box: only its
# documented arguments and returned values are used; its source was not
# read. Run from the repository root after _generate_cbps_medsens_data.py:
#   Rscript tests/reference_parity/_fixtures/_generate_medsens_R.R
# Writes _fixtures/medsens_R.json (digits = 17).
#
# Cases on medsens_sim.csv (linear mediator and outcome models):
#   bin_cov     binary t, covariates x1 + x2, rho.by = 0.05, default eps
#   bin_cov_fp  the same with eps = 1e-26 (the FGLS fixed point)
#   bin_nocov   binary t, no covariates, rho.by = 0.1
#   cont        continuous tc (control.value = 0, treat.value = 1),
#               covariates x1 + x2, rho.by = 0.1
# For each: rho, d0, d1, lower/upper.d0, err.cr.d, r.square.m / r.square.y,
# R2star.prod / R2tilde.prod (curves over rho), the thresholds, and the
# closed-form crossing corr(resid(M ~ T + X), resid(Y ~ T + X)) computed
# here with lm (Imai, Keele & Yamamoto 2010, Theorem 4).
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(mediation); library(jsonlite) })
FIX <- "tests/reference_parity/_fixtures"
df <- read.csv(file.path(FIX, "medsens_sim.csv"))
out <- list(versions = list(R = paste(R.version$major, R.version$minor, sep = "."),
                            mediation = as.character(packageVersion("mediation"))))
run <- function(fm, fy, treat, med, rho.by, eps = sqrt(.Machine$double.eps), ...) {
  set.seed(1)
  me <- mediate(fm, fy, treat = treat, mediator = med, sims = 50, ...)
  ms <- medsens(me, rho.by = rho.by, effect.type = "indirect", eps = eps)
  tot <- update(fy, as.formula(paste(". ~ . -", med)))
  list(rho = ms$rho, d0 = as.vector(ms$d0), d1 = as.vector(ms$d1),
       lower = as.vector(ms$lower.d0), upper = as.vector(ms$upper.d0),
       err_cr = ms$err.cr.d, r2_m = ms$r.square.m, r2_y = ms$r.square.y,
       R2star_prod = as.vector(ms$R2star.prod), R2tilde_prod = as.vector(ms$R2tilde.prod),
       R2star_thresh = ms$R2star.d.thresh, R2tilde_thresh = ms$R2tilde.d.thresh,
       rho_tilde = cor(resid(fm), resid(tot)))
}
fm <- lm(m ~ t + x1 + x2, data = df); fy <- lm(y ~ t + m + x1 + x2, data = df)
out$bin_cov <- run(fm, fy, "t", "m", 0.05)
out$bin_cov_fp <- run(fm, fy, "t", "m", 0.05, eps = 1e-26)
fm0 <- lm(m ~ t, data = df); fy0 <- lm(y ~ t + m, data = df)
out$bin_nocov <- run(fm0, fy0, "t", "m", 0.1)
fmc <- lm(mc ~ tc + x1 + x2, data = df); fyc <- lm(yc ~ tc + mc + x1 + x2, data = df)
out$cont <- run(fmc, fyc, "tc", "mc", 0.1, control.value = 0, treat.value = 1)
writeLines(toJSON(out, digits = 17, auto_unbox = TRUE, pretty = TRUE),
           file.path(FIX, "medsens_R.json"))
cat("wrote", file.path(FIX, "medsens_R.json"), "\n")
