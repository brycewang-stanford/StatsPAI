# Long runs of the R package stochvol on the committed return series, the
# screen for sp.stochvol. Three seeds, 40,000 draws each.
#   Rscript stochvol_reference.R stochvol_returns.csv stochvol_R.json
suppressMessages({library(stochvol); library(jsonlite)})
args <- commandArgs(TRUE)
y <- read.csv(args[1])$y
y <- y - mean(y)
runs <- list()
for (s in 1:3) {
  set.seed(s)
  f <- svsample(y, draws = 40000, burnin = 5000, priormu = c(0, 10),
                priorphi = c(5, 1.5), priorsigma = 1, quiet = TRUE)
  p <- as.matrix(f$para[[1]])[, c("mu", "phi", "sigma")]
  v <- colMeans(exp(as.matrix(f$latent[[1]]) / 2))
  runs[[s]] <- list(mean = as.list(colMeans(p)), sd = as.list(apply(p, 2, sd)),
                    vol = unname(v[c(1, 500, 1000, 1500)]))
}
write_json(list(stochvol_version = as.character(packageVersion("stochvol")),
                runs = runs), args[2], digits = 10, auto_unbox = TRUE, pretty = TRUE)
