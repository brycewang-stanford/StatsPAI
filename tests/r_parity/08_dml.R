# StatsPAI DML PLR parity (R side) -- Module 08.
#
# Reads data/08_dml.csv (the StatsPAI Card 1995 replica) and runs
# DoubleML::DoubleMLPLR with mlr3::regr.lm nuisance learners. Five
# folds, single repetition. Tolerance: rel < 1e-3.
#
# Note: cross-fitting fold splits are seeded so that R sklearn-style
# KFold and mlr3 ResamplingCV(folds=5) DO NOT produce bit-equal
# splits. The gap on theta_DML_PLR therefore reflects fold-noise
# alone (the score function is identical on both sides).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)
  library(data.table)
})

MODULE <- "08_dml"

df <- read_csv_strict(MODULE)

# DoubleML wants a DoubleMLData object built from a data.table.
dt <- as.data.table(df)
covariates <- c("exper", "expersq", "black", "south", "smsa")

dml_data <- DoubleML::DoubleMLData$new(
  data = dt, y_col = "lwage", d_cols = "educ", x_cols = covariates
)

# Linear regression nuisance learners (closed-form, no MC noise).
ml_g <- mlr3::lrn("regr.lm")
ml_m <- mlr3::lrn("regr.lm")

set.seed(PARITY_SEED)
dml_obj <- DoubleML::DoubleMLPLR$new(
  data    = dml_data,
  ml_l    = ml_g,
  ml_m    = ml_m,
  n_folds = 5L,
  n_rep   = 1L,
  score   = "partialling out",
  dml_procedure = "dml2"
)
dml_obj$fit()

theta <- as.numeric(dml_obj$coef["educ"])
se    <- as.numeric(dml_obj$se["educ"])
ci    <- dml_obj$confint(joint = FALSE, level = 0.95)["educ", ]

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "theta_DML_PLR",
    estimate  = theta,
    se        = se,
    ci_lo     = unname(ci[1]),
    ci_hi     = unname(ci[2]),
    n         = nrow(df)
  )
)

write_results(MODULE, rows,
              extra = list(dml_model = "PLR",
                           n_folds = 5L,
                           ml_g = "regr.lm",
                           ml_m = "regr.lm",
                           score = "partialling out",
                           dml_procedure = "dml2"))
