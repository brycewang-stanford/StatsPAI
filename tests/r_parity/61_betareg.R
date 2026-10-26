# StatsPAI beta regression parity (R side) -- Module 61.
#
# link.phi='log' aligns betareg's precision equation with the log-link
# parameterisation used by sp.betareg and Stata betareg, so the ln_phi
# row (and its SE) is directly comparable across all three sides.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) dirname(normalizePath(sub("^--file=", "", .file_arg[1]))) else getwd()
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({ library(betareg) })

MODULE <- "61_betareg"
df <- read_csv_strict(MODULE)

# SE convention: betareg reports expected-(Fisher-)information SEs
# (Ferrari & Cribari-Neto 2004 convention); sp.betareg and Stata
# betareg (vce(oim)) report observed-information SEs. Point estimates
# are like-for-like at machine level; the SE columns carry a documented
# <=0.7% convention gap on this fixture (numDeriv observed-info SEs at
# the betareg optimum reproduce the sp/Stata SEs).
fit <- betareg(y ~ x1 + x2, data = df, link = "logit", link.phi = "log",
               control = betareg.control(reltol = 1e-14, maxit = 5000))
co <- coef(fit); se <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_intercept", estimate = unname(co["(Intercept)"]), se = unname(se["(Intercept)"]), n = nrow(df)),
  parity_row(MODULE, "beta_x1",        estimate = unname(co["x1"]),          se = unname(se["x1"]),          n = nrow(df)),
  parity_row(MODULE, "beta_x2",        estimate = unname(co["x2"]),          se = unname(se["x2"]),          n = nrow(df)),
  parity_row(MODULE, "ln_phi",         estimate = unname(co["(phi)_(Intercept)"]), se = unname(se["(phi)_(Intercept)"]), n = nrow(df))
)
# The default vcov is the analytic expected information. hessian = TRUE
# replaces it with the observed information from optim's numerical Hessian
# at the optimum, the convention of sp.betareg and Stata betareg; it agrees
# with them to ~1e-7, the accuracy of that numerical Hessian.
fit_oim <- betareg(y ~ x1 + x2, data = df, link = "logit", link.phi = "log",
                   control = betareg.control(reltol = 1e-14, maxit = 5000,
                                             hessian = TRUE))
co_o <- coef(fit_oim); se_o <- sqrt(diag(vcov(fit_oim)))
oim_names <- c(beta_intercept = "(Intercept)", beta_x1 = "x1", beta_x2 = "x2",
               ln_phi = "(phi)_(Intercept)")
for (lab in names(oim_names)) {
  nm <- oim_names[[lab]]
  rows[[length(rows) + 1L]] <- parity_row(MODULE, paste0(lab, "_oim"),
    estimate = unname(co_o[nm]), se = unname(se_o[nm]), n = nrow(df))
}
write_results(MODULE, rows,
              extra = list(package = "betareg::betareg",
                           link = "logit", phi_link = "log",
                           se_convention = "expected (Fisher) information"))
