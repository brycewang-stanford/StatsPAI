# StatsPAI panel SFA parity (R side) -- Module 29.
#
# Reads data/29_panel_sfa.csv and runs frontier::sfa with the
# time-invariant Pitt-Lee 1981 specification.
# Tolerance: rel < 1e-2.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(frontier)
  library(plm)
})

MODULE <- "29_panel_sfa"

df <- read_csv_strict(MODULE)

# frontier::sfa expects a pdata.frame with index=c(unit, time).
pdf <- plm::pdata.frame(df, index = c("unit", "year"))

fit <- frontier::sfa(
  formula = lny ~ lnk + lnl,
  data    = pdf,
  ineffDecrease = TRUE,        # production frontier
  truncNorm = FALSE,           # half-normal inefficiency
  timeEffect = FALSE           # time-invariant
)

co <- coef(fit)
se <- sqrt(diag(vcov(fit)))

# frontier::sfa parameterises sigmaSq = sigmaSq_v + sigmaSq_u and
# gamma = sigmaSq_u / sigmaSq.
sigma2  <- unname(co["sigmaSq"])
gamma   <- unname(co["gamma"])
sigma2_u <- gamma * sigma2
sigma2_v <- (1 - gamma) * sigma2
sigma_u <- sqrt(sigma2_u)
sigma_v <- sqrt(sigma2_v)

rows <- list(
  parity_row(MODULE, "beta_intercept",
             estimate = unname(co["(Intercept)"]),
             se = unname(se["(Intercept)"]),
             n = nrow(df)),
  parity_row(MODULE, "beta_lnk",
             estimate = unname(co["lnk"]),
             se = unname(se["lnk"]),
             n = nrow(df)),
  parity_row(MODULE, "beta_lnl",
             estimate = unname(co["lnl"]),
             se = unname(se["lnl"]),
             n = nrow(df)),
  parity_row(MODULE, "sigma_u", estimate = sigma_u, n = nrow(df)),
  parity_row(MODULE, "sigma_v", estimate = sigma_v, n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(distribution = "half-normal",
                           timeEffect = FALSE,
                           package = "frontier::sfa"))
