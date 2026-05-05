# StatsPAI OLS cluster-robust SE parity (R side) -- Module 14.
#
# Reads data/14_ols_cluster.csv and runs lm + sandwich::vcovCL with
# cluster = countyreal. Tolerance: rel < 1e-3.

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

MODULE  <- "14_ols_cluster"
FORMULA <- lemp ~ treat + year

df <- read_csv_strict(MODULE)

fit <- lm(FORMULA, data = df)
vc <- sandwich::vcovCL(fit, cluster = ~ countyreal, type = "HC1")
se <- sqrt(diag(vc))

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

write_results(MODULE, rows,
              extra = list(formula = deparse(FORMULA),
                           vcov = "cluster (CR1)",
                           cluster_var = "countyreal"))
