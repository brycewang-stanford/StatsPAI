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

# hessian=TRUE reports observed-information SEs (the sp.betareg /
# Stata betareg oim convention) instead of betareg's default expected
# (Fisher) information; fsmaxit=0 keeps the BFGS optimum the Hessian is
# evaluated at.
fit <- betareg(y ~ x1 + x2, data = df, link = "logit", link.phi = "log",
               hessian = TRUE,
               control = betareg.control(reltol = 1e-14, maxit = 5000))
co <- coef(fit); se <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_intercept", estimate = unname(co["(Intercept)"]), se = unname(se["(Intercept)"]), n = nrow(df)),
  parity_row(MODULE, "beta_x1",        estimate = unname(co["x1"]),          se = unname(se["x1"]),          n = nrow(df)),
  parity_row(MODULE, "beta_x2",        estimate = unname(co["x2"]),          se = unname(se["x2"]),          n = nrow(df)),
  parity_row(MODULE, "ln_phi",         estimate = unname(co["(phi)_(Intercept)"]), se = unname(se["(phi)_(Intercept)"]), n = nrow(df))
)
write_results(MODULE, rows,
              extra = list(package = "betareg::betareg",
                           link = "logit", phi_link = "log"))
