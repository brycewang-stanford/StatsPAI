# ---------------------------------------------------------------------------
# Reference numbers for tests/reference_parity/test_glmnet_r_parity.py
#
# glmnet and cv.glmnet on four simulated designs: more rows than columns,
# more columns than rows, a binary outcome, and a low-signal design that
# separates the two early-stopping rules. Both sides read the CSV files this
# script writes. The convergence threshold is 1e-14 so that the reference is
# the minimiser, not glmnet's default approximation to it.
#
# Requires: glmnet (4.1-10), jsonlite.
# Run:      Rscript tests/reference_parity/_fixtures/_generate_glmnet.R
#           (from the repository root)
# ---------------------------------------------------------------------------
suppressMessages({library(glmnet); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
mk <- function(seed, n, p, fam, noise = 1) {
  set.seed(seed)
  X <- round(matrix(rnorm(n * p), n, p), 3); X[, 2] <- X[, 2] * 5 + 3
  beta <- c(3, -2, 0, 0, 1.5, rep(0, p - 5)) / noise
  eta <- as.numeric(X %*% beta) / (if (fam == "binomial") 3 else 1)
  y <- if (fam == "binomial") rbinom(n, 1, plogis(eta - mean(eta)))
       else round(eta + rnorm(n, sd = noise), 3)
  list(X = X, y = y, foldid = sample(rep(1:5, length.out = n)), family = fam)
}
cases <- list(tall = mk(123, 80, 10, "gaussian"), wide = mk(7, 40, 60, "gaussian"),
              binomial = mk(11, 300, 8, "binomial"), low = mk(5, 200, 6, "gaussian", 8),
              low_binomial = mk(6, 200, 6, "binomial", 6))
ctl <- list(thresh = 1e-14, maxit = 1e7)
out <- list(meta = list(R = R.version.string, glmnet = as.character(packageVersion("glmnet"))))
for (nm in names(cases)) {
  d <- cases[[nm]]; fam <- d$family
  write.csv(data.frame(y = d$y, d$X, foldid = d$foldid),
            file.path(here, paste0("glmnet_", nm, ".csv")), row.names = FALSE)
  res <- list(family = fam)
  for (al in c(1, 0.5, 0)) {
    f <- do.call(glmnet, c(list(d$X, d$y, family = fam, alpha = al), ctl))
    cv <- do.call(cv.glmnet, c(list(d$X, d$y, family = fam, alpha = al, foldid = d$foldid), ctl))
    idx <- unique(pmin(length(f$lambda), c(1, 2, 5, 20, 40, length(f$lambda))))
    res[[paste0("alpha", al)]] <- list(alpha = al, lambda = f$lambda, df = f$df,
      dev = f$dev.ratio, idx = idx, a0 = as.numeric(f$a0)[idx],
      beta = as.matrix(f$beta)[, idx, drop = FALSE], cvm = cv$cvm, cvsd = cv$cvsd,
      lambda_min = cv$lambda.min, lambda_1se = cv$lambda.1se,
      coef_1se = as.numeric(coef(cv, s = "lambda.1se")))
  }
  # penalty factors, one predictor unpenalised
  pf <- c(0, 2, rep(1, ncol(d$X) - 2))
  f <- do.call(glmnet, c(list(d$X, d$y, family = fam, alpha = 0.7, penalty.factor = pf), ctl))
  idx <- unique(pmin(length(f$lambda), c(1, 10, 30, length(f$lambda))))
  res$penalty_factor <- list(pf = pf, lambda = f$lambda, idx = idx,
    a0 = as.numeric(f$a0)[idx], beta = as.matrix(f$beta)[, idx, drop = FALSE])
  # penalties supplied by the user; without standardisation; cross-validated
  lam <- c(0.3, 0.05, 0.001)
  f <- do.call(glmnet, c(list(d$X, d$y, family = fam, alpha = 1, lambda = lam), ctl))
  g <- do.call(glmnet, c(list(d$X, d$y, family = fam, alpha = 1, lambda = lam, standardize = FALSE), ctl))
  lam6 <- f$lambda[1] * c(1, 0.5, 0.25, 0.1, 0.05, 0.01)
  cvu <- do.call(cv.glmnet, c(list(d$X, d$y, family = fam, alpha = 0.5, lambda = lam6, foldid = d$foldid), ctl))
  res$user <- list(lambda = lam, a0 = as.numeric(f$a0), beta = as.matrix(f$beta),
    a0_nostd = as.numeric(g$a0), beta_nostd = as.matrix(g$beta),
    cv_lambda = cvu$lambda, cvm = cvu$cvm, cvsd = cvu$cvsd, lambda_min = cvu$lambda.min, lambda_1se = cvu$lambda.1se)
  out[[nm]] <- res
}
# the adaptive lasso as the textbooks write it: ridge first, weights 1 / |b|
d <- cases$tall
ridge <- do.call(glmnet, c(list(d$X, d$y, alpha = 0, lambda = 0.1), ctl))
w <- 1 / abs(as.numeric(ridge$beta))
ad <- do.call(cv.glmnet, c(list(d$X, d$y, alpha = 1, penalty.factor = w, foldid = d$foldid), ctl))
out$adaptive <- list(ridge = as.numeric(ridge$beta), weights = w, lambda_min = ad$lambda.min,
  lambda_1se = ad$lambda.1se, coef_min = as.numeric(coef(ad, s = "lambda.min")),
  coef_1se = as.numeric(coef(ad, s = "lambda.1se")))
write_json(out, file.path(here, "glmnet_R.json"), digits = NA, auto_unbox = TRUE)
