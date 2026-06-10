# StatsPAI Causal Forest parity (R side) -- Module 13.
#
# Reads data/13_causal_forest.csv (clean-overlap DGP written by the
# Python side at %.16g precision) and runs grf::causal_forest with
# grf::average_treatment_effect for the "all" (ATE) and "treated"
# (ATT) targets -- the doubly-robust AIPW estimands, like-for-like
# with sp.causal_forest.average_treatment_effect.
#
# See 13_causal_forest.py for why the clean-overlap DGP replaces the
# former NSW-DW sample. Registered tolerance: rel_est < 0.005 on ATE/ATT.

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
covariates <- c("x1", "x2", "x3", "x4", "x5")

X <- as.matrix(df[, covariates])
Y <- df$Y
W <- df$T

set.seed(PARITY_SEED)
cf <- grf::causal_forest(X = X, Y = Y, W = W, num.trees = 2000,
                         seed = PARITY_SEED)

ate_obj <- grf::average_treatment_effect(cf, target.sample = "all")
att_obj <- grf::average_treatment_effect(cf, target.sample = "treated")

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "ate_causal_forest",
    estimate  = unname(ate_obj["estimate"]),
    se        = unname(ate_obj["std.err"]),
    ci_lo     = unname(ate_obj["estimate"] - qnorm(0.975) * ate_obj["std.err"]),
    ci_hi     = unname(ate_obj["estimate"] + qnorm(0.975) * ate_obj["std.err"]),
    n         = nrow(df)
  ),
  parity_row(
    module    = MODULE,
    statistic = "att_causal_forest",
    estimate  = unname(att_obj["estimate"]),
    se        = unname(att_obj["std.err"]),
    ci_lo     = unname(att_obj["estimate"] - qnorm(0.975) * att_obj["std.err"]),
    ci_hi     = unname(att_obj["estimate"] + qnorm(0.975) * att_obj["std.err"]),
    n         = nrow(df)
  )
)

write_results(MODULE, rows,
              extra = list(num.trees = 2000,
                           seed = PARITY_SEED,
                           estimator = "AIPW doubly-robust (grf)"))
