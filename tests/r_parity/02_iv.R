# StatsPAI 2SLS parity (R side) -- Module 02.
#
# Reads data/02_iv.csv and runs AER::ivreg with HC1 robust SE.
# Tolerance: rel < 1e-6.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(AER)
  library(sandwich)
})

MODULE  <- "02_iv"
# Reduced form: lwage ~ exper + expersq + black + south + smsa + educ
# Instruments: exper + expersq + black + south + smsa + nearc4
FORMULA <- lwage ~ exper + expersq + black + south + smsa + educ |
                   exper + expersq + black + south + smsa + nearc4

df <- read_csv_strict(MODULE)

fit <- AER::ivreg(FORMULA, data = df)
vc  <- sandwich::vcovHC(fit, type = "HC1")
se  <- sqrt(diag(vc))

rows <- list()
for (name in names(coef(fit))) {
  beta <- unname(coef(fit)[name])
  s    <- unname(se[name])
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = paste0("beta_", name),
    estimate  = beta,
    se        = s,
    ci_lo     = beta - qnorm(0.975) * s,
    ci_hi     = beta + qnorm(0.975) * s,
    n         = nrow(df)
  )
}

write_results(MODULE, rows, extra = list(formula = deparse(FORMULA), vcov = "HC1"))
