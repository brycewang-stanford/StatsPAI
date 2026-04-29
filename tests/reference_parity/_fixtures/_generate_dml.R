#!/usr/bin/env Rscript
# Generate R DoubleML reference values for the PLR + IRM specifications.
#
# This script is run ONCE — its output (dml_R.json) is committed as the
# golden fixture against which sp.dml is validated.  Re-run only when:
#   • the underlying DGP changes (see _generate_dml_data.py), or
#   • an R DoubleML version bump shifts the numerical answer (in which
#     case the version metadata in the JSON should be bumped too).
#
# Numerical seed is fixed via DoubleML's set.seed() pattern; the same
# seed is used in the Python test for paired comparisons.

suppressMessages({
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)
  library(jsonlite)
  lgr::get_logger("mlr3")$set_threshold("warn")
})

# Read the same data the Python test uses
csv_path <- "tests/reference_parity/_fixtures/dml_data.csv"
df <- read.csv(csv_path)

# Cast to data.table for DoubleML
suppressMessages(library(data.table))
dt <- as.data.table(df)

x_cols <- grep("^x", names(dt), value = TRUE)

# ── Partially Linear Regression (PLR) ──────────────────────────────────
set.seed(42)
data_plr <- DoubleMLData$new(
  dt, y_col = "y", d_cols = "d", x_cols = x_cols
)
ml_l <- lrn("regr.cv_glmnet")
ml_m <- lrn("regr.cv_glmnet")

set.seed(42)
plr <- DoubleMLPLR$new(
  data_plr, ml_l = ml_l, ml_m = ml_m,
  n_folds = 5, n_rep = 1, score = "partialling out"
)
plr$fit()
plr_coef <- plr$coef[["d"]]
plr_se   <- plr$se[["d"]]

# ── Interactive Regression Model (IRM, ATE) ────────────────────────────
set.seed(42)
data_irm <- DoubleMLData$new(
  dt, y_col = "y", d_cols = "d_bin", x_cols = x_cols
)
ml_g <- lrn("regr.cv_glmnet")
ml_p <- lrn("classif.cv_glmnet")
set.seed(42)
irm <- DoubleMLIRM$new(
  data_irm, ml_g = ml_g, ml_m = ml_p,
  n_folds = 5, n_rep = 1, score = "ATE"
)
irm$fit()
irm_coef <- irm$coef[["d_bin"]]
irm_se   <- irm$se[["d_bin"]]

# ── Save fixture ───────────────────────────────────────────────────────
out <- list(
  meta = list(
    R_version       = R.version.string,
    DoubleML_version = as.character(packageVersion("DoubleML")),
    mlr3_version    = as.character(packageVersion("mlr3")),
    mlr3learners_version = as.character(packageVersion("mlr3learners")),
    seed            = 42L,
    n_folds         = 5L,
    n_rep           = 1L,
    learner         = "cv_glmnet"
  ),
  plr = list(coef = plr_coef, se = plr_se),
  irm = list(coef = irm_coef, se = irm_se)
)
out_path <- "tests/reference_parity/_fixtures/dml_R.json"
write_json(out, out_path, pretty = TRUE, auto_unbox = TRUE,
           digits = NA, na = "null")
cat("Wrote", out_path, "\n")
cat(sprintf("  PLR: coef=%.6f  se=%.6f\n", plr_coef, plr_se))
cat(sprintf("  IRM: coef=%.6f  se=%.6f\n", irm_coef, irm_se))
