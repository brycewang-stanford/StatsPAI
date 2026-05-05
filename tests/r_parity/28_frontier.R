# StatsPAI cross-section frontier parity (R side) -- Module 28.
#
# Reads data/28_frontier.csv and runs sfaR::sfacross with half-
# normal inefficiency. Tolerance: rel < 1e-2.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(sfaR)
})

MODULE <- "28_frontier"

df <- read_csv_strict(MODULE)

set.seed(PARITY_SEED)
fit <- sfaR::sfacross(
  formula = lny ~ lnk + lnl,
  data    = df,
  udist   = "hnormal",
  S       = 1L              # production frontier (S = -1 for cost)
)

# sfaR returns coefficients in `mlParam`; the frontier coefs come
# first, then the variance parameters in log-form.
co <- coef(fit)
# Standard errors:
se <- sqrt(diag(vcov(fit)))

# Variance components: sfaR exposes sigmaSq_u, sigmaSq_v.
em <- summary(fit)$mlParamMatrix
# Extract sigma_u, sigma_v from the log-parametrisation.
ln_sigma_u_sq <- co["Wu_(Intercept)"]
ln_sigma_v_sq <- co["Vu_(Intercept)"]
sigma_u <- exp(0.5 * ln_sigma_u_sq)
sigma_v <- exp(0.5 * ln_sigma_v_sq)
lambda  <- sigma_u / sigma_v

# Mean technical efficiency (Battese-Coelli).
eff <- sfaR::efficiencies(fit)
mean_eff <- mean(eff[, "teJLMS"])  # Jondrow-Lovell-Materov-Schmidt

rows <- list(
  parity_row(MODULE, "beta_intercept",
             estimate = unname(co["(Intercept)"]),
             se = unname(se["(Intercept)"]), n = nrow(df)),
  parity_row(MODULE, "beta_lnk",
             estimate = unname(co["lnk"]),
             se = unname(se["lnk"]), n = nrow(df)),
  parity_row(MODULE, "beta_lnl",
             estimate = unname(co["lnl"]),
             se = unname(se["lnl"]), n = nrow(df)),
  parity_row(MODULE, "sigma_u", estimate = sigma_u, n = nrow(df)),
  parity_row(MODULE, "sigma_v", estimate = sigma_v, n = nrow(df)),
  parity_row(MODULE, "lambda", estimate = lambda, n = nrow(df)),
  parity_row(MODULE, "mean_efficiency",
             estimate = mean_eff, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(distribution = "half-normal",
                           S = 1, package = "sfaR::sfacross"))
