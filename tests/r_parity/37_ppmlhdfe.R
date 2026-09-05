# StatsPAI Poisson PML with HDFE parity (R side) -- Module 37.
#
# Reads data/37_ppmlhdfe.csv and runs fixest::fepois with HC1 SEs.
# Tolerance: rel < 1e-3 on coefficients; rel < 1e-2 on SEs.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(fixest)
})

# Rows are named beta_<x>__fixestK: fixest::fepois's heteroskedasticity-
# robust vcov applies N/(N-K) with K = slopes + absorbed FE levels
# (minus one per additional FE dimension); the Python side emits the
# same-convention rows under this suffix (ssc = "fixest") and keeps the
# Stata ppmlhdfe N/(N-1) rows unsuffixed for the Stata side.
MODULE <- "37_ppmlhdfe"

df <- read_csv_strict(MODULE)

fit <- fixest::fepois(
  y ~ x1 + x2 | origin,
  data = df,
  vcov = "hetero"   # HC1
)

co <- coef(fit)
se <- sqrt(diag(vcov(fit)))

rows <- list(
  parity_row(MODULE, "beta_x1__fixestK",
             estimate = unname(co["x1"]),
             se = unname(se["x1"]),
             n = nrow(df)),
  parity_row(MODULE, "beta_x2__fixestK",
             estimate = unname(co["x2"]),
             se = unname(se["x2"]),
             n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(
                fe = "origin",
                vcov = "HC1",
                package = "fixest::fepois"
              ))
