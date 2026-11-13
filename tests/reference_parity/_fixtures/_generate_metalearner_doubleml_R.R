# Reference for the average effect that sp.metalearner / sp.compare_metalearners
# report: the cross-fitted AIPW estimate, which is DoubleML's interactive
# regression model (score "ATE") when both use the same folds and learners.
#
# DoubleMLIRM on metalearner_train.csv with the stored folds, least squares
# for the two arm outcome regressions (regr.lm), logistic regression for the
# propensity (classif.log_reg at epsilon = 1e-14), trimming at 0.01.
# Recorded: the estimate and DoubleML's standard error, sqrt(mean(psi^2) / n).
#
#   Rscript tests/reference_parity/_fixtures/_generate_metalearner_doubleml_R.R
suppressMessages({
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)
  library(data.table)
  library(jsonlite)
})
lgr::get_logger("mlr3")$set_threshold("warn")
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
d <- fread(file.path(here, "metalearner_train.csv"))
dml_data <- DoubleMLData$new(d, y_col = "y", d_cols = "d",
                             x_cols = c("x1", "x2", "x3"))
ml_m <- lrn("classif.log_reg", predict_type = "prob", epsilon = 1e-14, maxit = 100)
irm <- DoubleMLIRM$new(dml_data, ml_g = lrn("regr.lm"), ml_m = ml_m,
                       score = "ATE", trimming_threshold = 0.01,
                       draw_sample_splitting = FALSE)
folds <- sort(unique(d$fold))
irm$set_sample_splitting(list(list(
  train_ids = lapply(folds, function(k) which(d$fold != k)),
  test_ids = lapply(folds, function(k) which(d$fold == k))
)))
irm$fit()
res <- list(
  versions = list(R = R.version.string,
                  DoubleML = as.character(packageVersion("DoubleML")),
                  mlr3 = as.character(packageVersion("mlr3")),
                  mlr3learners = as.character(packageVersion("mlr3learners"))),
  n = nrow(d), ate = unname(irm$coef), se = unname(irm$se)
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "metalearner_doubleml_R.json"))
print(c(res$ate, res$se), digits = 15)
