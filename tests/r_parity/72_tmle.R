# StatsPAI TMLE parity -- Module 72 (R side).
#
# Reads data/72_tmle.csv (written by 72_tmle.py at %.16g precision) and
# runs tmle::tmle on the *shared* initial fits carried in the Q0 / Q1 /
# g1W columns. Supplying Q and g1W bypasses the package's own
# SuperLearner stage, so both engines run only their targeting step on
# identical inputs and the row grades the fluctuation + plug-in rather
# than two different nuisance estimators.
#
# family = "binomial" because the fixture's outcome is binary; that
# removes the [0,1] rescaling both implementations apply to continuous
# outcomes from the comparison.
#
# gbound = 1e-8 disables propensity truncation. The DGP keeps g inside
# (0.25, 0.75), so no unit is near a bound; pinning the parameter on
# both sides removes it as a spurious convention difference.
#
# tmle::tmle fluctuates along two per-arm clever covariates and returns a
# 2-vector epsilon. The StatsPAI side therefore uses
# fluctuation="per_arm"; its documented default ("single", one clever
# covariate and a scalar epsilon) is a different submodel and is
# recorded in the Python side's extra block. See 72_tmle.py.
#
# Registered tolerance: rel_est < 1e-6 (machine tier).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(tmle)
})

MODULE <- "72_tmle"
GBOUND <- 1e-8

df <- read_csv_strict(MODULE)
covariates <- c("w1", "w2", "w3")
W <- as.matrix(df[, covariates])
n <- nrow(df)

fit <- tmle::tmle(
  Y = df$Y,
  A = df$A,
  W = W,
  Q = cbind(df$Q0, df$Q1),
  g1W = df$g1W,
  family = "binomial",
  gbound = GBOUND
)

psi <- as.numeric(fit$estimates$ATE$psi)
se <- sqrt(as.numeric(fit$estimates$ATE$var.psi))
ci <- as.numeric(fit$estimates$ATE$CI)

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "psi_tmle_ate",
    estimate  = psi,
    se        = se,
    ci_lo     = ci[1],
    ci_hi     = ci[2],
    n         = n
  )
)

write_results(MODULE, rows,
              extra = list(
                family = "binomial",
                gbound = GBOUND,
                epsilon = as.numeric(fit$epsilon),
                estimator = "tmle::tmle (per-arm fluctuation)",
                nuisance = "Q and g1W supplied; package SuperLearner bypassed"
              ))
