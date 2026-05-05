# StatsPAI Cox PH parity (R side) -- Module 24.
#
# Reads data/24_coxph.csv and runs survival::coxph with Efron ties.
# Tolerance: rel < 1e-3.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(survival)
})

MODULE <- "24_coxph"

df <- read_csv_strict(MODULE)

fit <- survival::coxph(
  Surv(time, event) ~ x1 + x2,
  data   = df,
  ties   = "efron"
)

co  <- coef(fit)
ses <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_x1",
             estimate = unname(co["x1"]),
             se = unname(ses["x1"]),
             ci_lo = unname(co["x1"]) - qnorm(0.975) * unname(ses["x1"]),
             ci_hi = unname(co["x1"]) + qnorm(0.975) * unname(ses["x1"]),
             n = nrow(df)),
  parity_row(MODULE, "beta_x2",
             estimate = unname(co["x2"]),
             se = unname(ses["x2"]),
             ci_lo = unname(co["x2"]) - qnorm(0.975) * unname(ses["x2"]),
             ci_hi = unname(co["x2"]) + qnorm(0.975) * unname(ses["x2"]),
             n = nrow(df)),
  parity_row(MODULE, "concordance",
             estimate = unname(summary(fit)$concordance["C"]),
             n = nrow(df))
)

write_results(MODULE, rows, extra = list(ties = "efron"))
