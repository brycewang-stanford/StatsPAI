#!/usr/bin/env Rscript
# =====================================================================
#  R DoubleML under repeated cross-fitting (n_rep = 3), for
#  tests/reference_parity/test_dml_options_doubleml_parity.py.
#
#  Reads dml_options_data.csv (written by _generate_dml_options_doubleml.py)
#  and its fold columns rep0_kfold, rep1_kfold, rep2_kfold -- the folds
#  sp.dml(n_rep=3) draws itself -- and runs DoubleMLPLR with
#  set_sample_splitting(), OLS nuisances (regr.lm), both scores.
#
#  Writes dml_options_R.json: every repetition's estimate and SE
#  (all_coef, all_se) and the aggregated coef and se. The repetitions are
#  the parity quantities; the aggregated SE documents R DoubleML's rule,
#      se^2 = median(n * se_r^2 + (theta_r - median theta)^2) / n,
#  which is not the rule sp.dml or Python DoubleML uses.
#
#  Environment: R 4.5.2 / DoubleML 1.0.2 / mlr3 1.6.0 / mlr3learners 0.14.0
# =====================================================================
suppressPackageStartupMessages({
  library(DoubleML); library(mlr3); library(mlr3learners)
  library(data.table); library(jsonlite)
})
lgr::get_logger("mlr3")$set_threshold("warn")
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

dt <- as.data.table(read.csv(file.path(OUT, "dml_options_data.csv")))
covariates <- c("x1", "x2", "x3", "x4")
n <- nrow(dt)

smpls <- lapply(0:2, function(r) {
  f <- dt[[sprintf("rep%d_kfold", r)]]
  lv <- sort(unique(f))
  list(train_ids = lapply(lv, function(k) which(f != k)),
       test_ids  = lapply(lv, function(k) which(f == k)))
})

reg <- function() mlr3::lrn("regr.lm")
data <- DoubleML::DoubleMLData$new(data = dt, y_col = "y", d_cols = "d",
                                   x_cols = covariates)
out <- list()
for (score in c("partialling out", "IV-type")) {
  args <- list(data = data, ml_l = reg(), ml_m = reg(), n_folds = 5L,
               n_rep = 3L, score = score, draw_sample_splitting = FALSE)
  if (score == "IV-type") args$ml_g <- reg()
  obj <- do.call(DoubleML::DoubleMLPLR$new, args)
  obj$set_sample_splitting(smpls)
  obj$fit()
  key <- if (score == "IV-type") "plr_ivtype_rep3" else "plr_po_rep3"
  out[[key]] <- list(
    score = score,
    theta = unname(obj$coef[1]), se = unname(obj$se[1]),
    all_theta = as.numeric(obj$all_coef[1, ]),
    all_se = as.numeric(obj$all_se[1, ])
  )
}
out[["_meta"]] <- list(
  generated_by = "_generate_dml_options_R.R",
  r_version = R.version.string,
  doubleml_version = as.character(packageVersion("DoubleML")),
  mlr3_version = as.character(packageVersion("mlr3")),
  mlr3learners_version = as.character(packageVersion("mlr3learners")),
  n = n
)
write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(OUT, "dml_options_R.json"))
cat("[dml-options-R] wrote", length(out) - 1, "cases\n")
