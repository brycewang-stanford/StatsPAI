# Runs of the R package dbarts on the committed Friedman sample, the
# screen for sp.bart. Five seeds; test-sample error of the posterior mean,
# coverage and width of the 95 percent bands, and the error scale.
#   Rscript bart_reference.R bart_friedman.csv bart_R.json
suppressMessages({library(dbarts); library(jsonlite)})
args <- commandArgs(TRUE)
d <- read.csv(args[1])
xs <- paste0("x", 0:5)
tr <- d[d$test == 0, ]; te <- d[d$test == 1, ]
runs <- list()
for (s in 1:5) {
  set.seed(s)
  f <- bart(as.matrix(tr[, xs]), tr$y, as.matrix(te[, xs]), ntree = 200,
            ndpost = 1000, nskip = 1000, verbose = FALSE)
  m <- colMeans(f$yhat.test)
  lo <- apply(f$yhat.test, 2, quantile, 0.025)
  hi <- apply(f$yhat.test, 2, quantile, 0.975)
  runs[[s]] <- list(rmse = sqrt(mean((m - te$f)^2)),
                    coverage = mean(lo <= te$f & te$f <= hi),
                    width = mean(hi - lo), sigma = mean(f$sigma))
}
write_json(list(dbarts_version = as.character(packageVersion("dbarts")), runs = runs),
           args[2], digits = 8, auto_unbox = TRUE, pretty = TRUE)
