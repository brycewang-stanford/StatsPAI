# Reference for sp.super_learner: SuperLearner::SuperLearner with the
# convex-combination least-squares meta-learner (method.CC_LS), deterministic
# learners and the validation folds stored in super_learner_data.csv.
#
# Library (regression): the mean, a linear model on all four covariates,
# a linear model on x1 alone, and a linear model on the full quadratic
# expansion. Library (classification): the mean and two logistic models.
# Recorded: the cross-validated predictions Z, the cross-validated risks,
# the weights, and the ensemble prediction on the first 25 rows.
#
# SuperLearner is GPL-3; black box.
#
#   Rscript _generate_super_learner_R.R
suppressMessages({
  library(SuperLearner)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
d <- read.csv(file.path(here, "super_learner_data.csv"))
X <- d[, c("x1", "x2", "x3", "x4")]

SL.lm.x1 <- function(Y, X, newX, family, obsWeights, ...) {
  fit <- glm(Y ~ x1, data = X, family = family, weights = obsWeights)
  list(pred = predict(fit, newdata = newX, type = "response"),
       fit = list(object = fit))
}
SL.lm.quad <- function(Y, X, newX, family, obsWeights, ...) {
  f <- Y ~ x1 + x2 + x3 + x4 + I(x1^2) + I(x2^2) + I(x3^2) + I(x4^2) +
    x1:x2 + x1:x3 + x1:x4 + x2:x3 + x2:x4 + x3:x4
  fit <- glm(f, data = X, family = family, weights = obsWeights)
  list(pred = predict(fit, newdata = newX, type = "response"),
       fit = list(object = fit))
}
SL.glm.tight <- function(Y, X, newX, family, obsWeights, ...) {
  fit <- glm(Y ~ ., data = X, family = family, weights = obsWeights,
             control = glm.control(epsilon = 1e-14, maxit = 100))
  list(pred = predict(fit, newdata = newX, type = "response"),
       fit = list(object = fit))
}
SL.glm.x12 <- function(Y, X, newX, family, obsWeights, ...) {
  fit <- glm(Y ~ x1 + x2, data = X, family = family, weights = obsWeights,
             control = glm.control(epsilon = 1e-14, maxit = 100))
  list(pred = predict(fit, newdata = newX, type = "response"),
       fit = list(object = fit))
}
predict.SL.lm.x1 <- predict.SL.lm.quad <- predict.SL.glm.tight <-
  predict.SL.glm.x12 <- function(object, newdata, ...) {
    predict(object$object, newdata = newdata, type = "response")
  }
for (cl in c("SL.lm.x1", "SL.lm.quad", "SL.glm.tight", "SL.glm.x12")) {
  environment(get(cl))  # keep in global env for SuperLearner's lookup
}

run <- function(Y, family, lib, fold, w = NULL) {
  rows <- lapply(1:5, function(k) which(fold == k))
  fit <- SuperLearner(
    Y = Y, X = X, newX = X[1:25, ], family = family, SL.library = lib,
    method = "method.CC_LS", obsWeights = w,
    cvControl = list(V = 5L, validRows = rows)
  )
  list(library = lib, weights = unname(fit$coef), cv_risk = unname(fit$cvRisk),
       Z = unname(fit$Z), pred = as.numeric(fit$SL.predict))
}
w <- 0.5 + (seq_len(nrow(d)) %% 4) * 0.5
res <- list(
  versions = list(R = R.version.string,
                  SuperLearner = as.character(packageVersion("SuperLearner"))),
  regression = run(d$y, gaussian(),
                   c("SL.mean", "SL.glm", "SL.lm.x1", "SL.lm.quad"), d$fold_reg),
  regression_weighted = run(d$y, gaussian(),
                            c("SL.mean", "SL.glm", "SL.lm.x1", "SL.lm.quad"),
                            d$fold_reg, w),
  classification = run(d$yb, binomial(),
                       c("SL.mean", "SL.glm.tight", "SL.glm.x12"), d$fold_clf)
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "super_learner_R.json"))
print(res$regression$weights); print(res$regression_weighted$weights)
print(res$classification$weights)
