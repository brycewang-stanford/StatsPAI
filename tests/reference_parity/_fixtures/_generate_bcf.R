#!/usr/bin/env Rscript
# R bcf reference: posterior mean ATE on the shared seed=42 DGP.
suppressMessages({
  library(bcf)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/bcf_data.csv")
X <- as.matrix(df[, paste0("X", 1:5)])
y <- df$y
W <- df$W

# Estimate propensity score externally (BCF needs it as input)
glm_fit <- glm(W ~ X, family = binomial())
pihat <- predict(glm_fit, type = "response")

set.seed(42)
m <- bcf::bcf(
  y = y, z = W, x_control = X, x_moderate = X,
  pihat = pihat,
  nburn = 1000, nsim = 2000, w = NULL,
  random_seed = 42, n_chains = 1
)
# Posterior ATE = average across posterior draws of average tau
tau_post <- m$tau                  # samples × n
ate_draws <- rowMeans(tau_post)
ate_mean <- mean(ate_draws)
ate_sd   <- sd(ate_draws)

out <- list(
  meta = list(
    R_version = R.version.string,
    bcf_version = as.character(packageVersion("bcf")),
    nburn = 1000, nsim = 2000, seed = 42L
  ),
  ate = list(mean = ate_mean, sd = ate_sd),
  n_obs = nrow(df)
)
write_json(out, "tests/reference_parity/_fixtures/bcf_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("BCF ATE: mean=%.6f  sd=%.6f\n", ate_mean, ate_sd))
