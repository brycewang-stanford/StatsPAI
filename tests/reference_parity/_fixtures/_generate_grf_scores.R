#!/usr/bin/env Rscript
# R grf reference: the *doubly-robust score vector itself*, plus the
# forest outputs it is built from, on the shared seed=42 DGP.
#
# Why this fixture exists
# -----------------------
# The module-13 / grf_R.json comparison grades sp.causal_forest's AIPW
# ATE against grf's. Two independently grown forests never produce the
# same tau.hat, so that comparison can only ever be graded against
# combined Monte Carlo error -- and the AIPW *standard error*, which
# depends on the forest's own predictions, carries a 50% relative band
# that is too wide to be called validation.
#
# The fix is to separate the two things being tested:
#
#   1. the forest        -- not pinnable across implementations; its
#                           calibration is evidenced by coverage.
#   2. the AIPW operator -- the map from (Y, W, tau.hat, Y.hat, W.hat)
#                           to the score vector, the point estimate and
#                           the influence-function SE. This is a closed
#                           form and IS exactly pinnable.
#
# This fixture materialises everything needed to pin (2): grf's own
# tau.hat / Y.hat / W.hat, grf's own get_scores() output, and grf's
# reported ATE/ATT estimate and std.err. StatsPAI's score construction
# is then fed grf's forest outputs and must reproduce grf's scores
# elementwise and its ATE/SE to the floating-point floor.
#
#   Rscript tests/reference_parity/_fixtures/_generate_grf_scores.R

suppressMessages({
  library(grf)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/grf_data.csv")
X <- as.matrix(df[, paste0("X", 1:5)])
y <- df$y
W <- df$W

set.seed(42)
cf <- causal_forest(X, y, W, num.trees = 2000, seed = 42)

# Forest outputs that the AIPW score is a deterministic function of.
tau_hat <- as.numeric(predict(cf)$predictions)
y_hat <- as.numeric(cf$Y.hat)
w_hat <- as.numeric(cf$W.hat)

# grf's own doubly-robust score vector (the ATE scores), and the
# aggregate it reports from them.
scores <- as.numeric(grf::get_scores(cf))
ate <- average_treatment_effect(cf, target.sample = "all")
att <- average_treatment_effect(cf, target.sample = "treated")

out <- list(
  meta = list(
    R_version = R.version.string,
    grf_version = as.character(packageVersion("grf")),
    num_trees = 2000L,
    seed = 42L,
    note = paste(
      "tau_hat / y_hat / w_hat are grf's forest outputs;",
      "scores is grf::get_scores(cf), the ATE doubly-robust score",
      "vector; ate/att are grf::average_treatment_effect. Together",
      "these pin the AIPW operator independently of the forest."
    )
  ),
  n_obs = nrow(df),
  y = y,
  W = W,
  tau_hat = tau_hat,
  y_hat = y_hat,
  w_hat = w_hat,
  scores = scores,
  ate = list(
    estimate = unname(ate["estimate"]),
    se = unname(ate["std.err"])
  ),
  att = list(
    estimate = unname(att["estimate"]),
    se = unname(att["std.err"])
  )
)
write_json(out, "tests/reference_parity/_fixtures/grf_scores_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("ATE %.10f (se %.10f);  ATT %.10f (se %.10f);  n=%d\n",
            ate["estimate"], ate["std.err"],
            att["estimate"], att["std.err"], nrow(df)))
cat(sprintf("mean(scores)=%.10f  sd(scores)/sqrt(n)=%.10f\n",
            mean(scores), sd(scores) / sqrt(length(scores))))
