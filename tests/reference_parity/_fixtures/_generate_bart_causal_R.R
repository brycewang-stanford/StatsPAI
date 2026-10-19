# Seed-replicate reference for sp.bart_causal (stochastic screen, tier S;
# never parity). Checked in tests/reference_parity/test_bart_causal_seed_screen.py.
# Run from this folder:  Rscript _generate_bart_causal_R.R
# bcf 2.0.2 and bartCause 1.0.10 are GPL: used as black boxes (outputs
# only). Data: bart_causal_data.csv, the DGP of bcf's help-page example
# (n = 300, numpy default_rng(20261019)) with the true propensity `pi`
# supplied to both packages. Seeds 1000 + 100000 k (k = 1..8) so that
# neighbouring seeds do not share draws.
suppressPackageStartupMessages({ library(bcf); library(bartCause); library(jsonlite) })
d <- read.csv("bart_causal_data.csv")
X <- as.matrix(d[, c("x1", "x2", "x3")])
out <- list(bcf = list(), bart = list())
for (k in 1:8) {
  s <- 1000 + 100000 * k
  f <- bcf(d$y, d$z, X, X, d$pi, nburn = 1000, nsim = 1000, n_chains = 1,
           random_seed = s, no_output = TRUE, verbose = FALSE, n_threads = 1)
  ate <- rowMeans(f$tau)
  out$bcf[[k]] <- list(seed = s, ate_mean = mean(ate), ate_sd = sd(ate),
                       cate_mean = colMeans(f$tau))
  b <- bartc(d$y, d$z, X, method.rsp = "bart", method.trt = d$pi,
             estimand = "ate", n.chains = 1L, n.burn = 1000L, n.samples = 1000L,
             seed = s, verbose = FALSE)
  ate_b <- extract(b, type = "cate")  # mean over the sample of f(x, 1) - f(x, 0)
  out$bart[[k]] <- list(seed = s, ate_mean = mean(ate_b), ate_sd = sd(ate_b),
                        cate_mean = colMeans(extract(b, type = "icate")))
}
out$versions <- list(R = R.version.string, bcf = as.character(packageVersion("bcf")),
                     bartCause = as.character(packageVersion("bartCause")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17)), "bart_causal_R.json")
