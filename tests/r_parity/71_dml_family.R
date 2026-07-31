# StatsPAI DML family parity (R side) -- Module 71.
#
# Reads data/71_dml_family.csv (written by 71_dml_family.py at %.16g
# precision) and runs the three DoubleML model classes that module 08
# does not cover:
#
#   DoubleML::DoubleMLIRM   -- binary D, ATE
#   DoubleML::DoubleMLPLIV  -- endogenous continuous D, instrument z_c
#   DoubleML::DoubleMLIIVM  -- binary D and binary Z, LATE
#
# All three consume the deterministic fold_id column through
# set_sample_splitting(), the same partition the StatsPAI side uses via
# fold_indices=. With the split fixed by the data, cross-fitting
# contributes no Monte Carlo term and the residual gap is the estimator.
#
# Nuisance learners are closed-form counterparts of the sklearn side:
# regr.lm for regressions, classif.log_reg (stats::glm) for the binary
# nuisances. trimming_threshold is set to 1e-12 on both sides (DoubleML's
# default); the DGP keeps propensities inside (0.2, 0.8), so no unit is
# trimmed either way.
#
# Registered tolerance: rel_est < 1e-6 (PLIV) / 1e-4 (IRM, IIVM).
# See 71_dml_family.py.

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

MODULE <- "71_dml_family"
N_FOLDS <- 5L
TRIMMING <- 1e-12

df <- read_csv_strict(MODULE)
dt <- as.data.table(df)
covariates <- c("x1", "x2", "x3", "x4", "x5")
n <- nrow(dt)

# The shared partition, expressed the way DoubleML wants it.
fold_levels <- sort(unique(dt$fold_id))
test_ids <- lapply(fold_levels, function(k) which(dt$fold_id == k))
train_ids <- lapply(fold_levels, function(k) which(dt$fold_id != k))
smpls <- list(list(train_ids = train_ids, test_ids = test_ids))

reg <- function() mlr3::lrn("regr.lm")
clf <- function() mlr3::lrn("classif.log_reg", predict_type = "prob")

collect <- function(obj, treat_col) {
  ci <- obj$confint(joint = FALSE, level = 0.95)[treat_col, ]
  list(
    estimate = as.numeric(obj$coef[treat_col]),
    se = as.numeric(obj$se[treat_col]),
    ci_lo = unname(ci[1]),
    ci_hi = unname(ci[2])
  )
}

# --- IRM: binary treatment, ATE -------------------------------------
# DoubleML requires a factor outcome for the classification nuisance;
# the treatment column itself is passed as the binary d_col.
irm_data <- DoubleML::DoubleMLData$new(
  data = dt, y_col = "y_irm", d_cols = "d_bin", x_cols = covariates
)
set.seed(PARITY_SEED)
irm <- DoubleML::DoubleMLIRM$new(
  data = irm_data,
  ml_g = reg(),
  ml_m = clf(),
  n_folds = N_FOLDS,
  n_rep = 1L,
  score = "ATE",
  trimming_rule = "truncate",
  trimming_threshold = TRIMMING,
  dml_procedure = "dml2",
  draw_sample_splitting = FALSE
)
irm$set_sample_splitting(smpls)
irm$fit()
irm_out <- collect(irm, "d_bin")

# --- PLIV: endogenous continuous D, continuous instrument -----------
pliv_data <- DoubleML::DoubleMLData$new(
  data = dt, y_col = "y_pliv", d_cols = "d_cont",
  x_cols = covariates, z_cols = "z_c"
)
set.seed(PARITY_SEED)
pliv <- DoubleML::DoubleMLPLIV$new(
  data = pliv_data,
  ml_l = reg(),
  ml_m = reg(),
  ml_r = reg(),
  n_folds = N_FOLDS,
  n_rep = 1L,
  score = "partialling out",
  dml_procedure = "dml2",
  draw_sample_splitting = FALSE
)
pliv$set_sample_splitting(smpls)
pliv$fit()
pliv_out <- collect(pliv, "d_cont")

# --- IIVM: binary D, binary Z, LATE ---------------------------------
iivm_data <- DoubleML::DoubleMLData$new(
  data = dt, y_col = "y_iivm", d_cols = "d_iv",
  x_cols = covariates, z_cols = "z_b"
)
set.seed(PARITY_SEED)
iivm <- DoubleML::DoubleMLIIVM$new(
  data = iivm_data,
  ml_g = reg(),
  ml_m = clf(),
  ml_r = clf(),
  n_folds = N_FOLDS,
  n_rep = 1L,
  score = "LATE",
  trimming_rule = "truncate",
  trimming_threshold = TRIMMING,
  dml_procedure = "dml2",
  draw_sample_splitting = FALSE
)
iivm$set_sample_splitting(smpls)
iivm$fit()
iivm_out <- collect(iivm, "d_iv")

mk <- function(stat, o) {
  parity_row(MODULE, stat, estimate = o$estimate, se = o$se,
             ci_lo = o$ci_lo, ci_hi = o$ci_hi, n = n)
}

rows <- list(
  mk("theta_DML_IRM", irm_out),
  mk("theta_DML_PLIV", pliv_out),
  mk("theta_DML_IIVM", iivm_out)
)

write_results(MODULE, rows,
              extra = list(
                n_folds = N_FOLDS,
                fold_source = "user",
                fold_column = "fold_id",
                trimming_threshold = TRIMMING,
                ml_regression = "regr.lm",
                ml_classification = "classif.log_reg",
                dml_procedure = "dml2",
                scores = list(irm = "ATE", pliv = "partialling out",
                              iivm = "LATE")
              ))
