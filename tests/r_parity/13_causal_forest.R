# StatsPAI Causal Forest parity (R side) -- Module 13.
#
# Reads data/13_causal_forest.csv (NSW-DW replica) and runs
# grf::causal_forest with grf::average_treatment_effect for the
# treated sample. Tolerance: rel < 1e-2 on the ATT.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(grf)
})

MODULE <- "13_causal_forest"

df <- read_csv_strict(MODULE)
covariates <- c("age", "education", "black", "hispanic", "married", "re74", "re75")

X <- as.matrix(df[, covariates])
Y <- df$re78
W <- df$treat

set.seed(PARITY_SEED)
cf <- grf::causal_forest(X = X, Y = Y, W = W, num.trees = 2000,
                          seed = PARITY_SEED)

att_obj <- grf::average_treatment_effect(cf, target.sample = "treated")
ate_obj <- grf::average_treatment_effect(cf, target.sample = "all")

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "att_causal_forest",
    estimate  = unname(att_obj["estimate"]),
    se        = unname(att_obj["std.err"]),
    ci_lo     = unname(att_obj["estimate"] - qnorm(0.975) * att_obj["std.err"]),
    ci_hi     = unname(att_obj["estimate"] + qnorm(0.975) * att_obj["std.err"]),
    n         = nrow(df)
  ),
  parity_row(
    module    = MODULE,
    statistic = "ate_causal_forest",
    estimate  = unname(ate_obj["estimate"]),
    se        = unname(ate_obj["std.err"]),
    ci_lo     = unname(ate_obj["estimate"] - qnorm(0.975) * ate_obj["std.err"]),
    ci_hi     = unname(ate_obj["estimate"] + qnorm(0.975) * ate_obj["std.err"]),
    n         = nrow(df)
  )
)

write_results(MODULE, rows,
              extra = list(num.trees = 2000,
                           seed = PARITY_SEED))
