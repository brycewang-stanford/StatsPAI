# StatsPAI OLS parity (R side) -- Module 01.
#
# Reads data/01_ols.csv (written by 01_ols.py) and runs lm() with
# sandwich::vcovHC(type = "HC1"). Tolerance: rel < 1e-6.

# Resolve _common.R relative to this script (works under Rscript and
# under interactive source() with chdir = TRUE).
.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(sandwich)
})

MODULE  <- "01_ols"
FORMULA <- lwage ~ educ + exper + expersq + black + south + smsa

df <- read_csv_strict(MODULE)

fit <- lm(FORMULA, data = df)
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
