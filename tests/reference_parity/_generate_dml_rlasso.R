#!/usr/bin/env Rscript
# Ground-truth for sp.dml(model='plr', ml_g='rlasso', ml_m='rlasso'):
# a manual Double-ML partially-linear-regression estimator whose nuisances
# E[Y|X] and E[D|X] are fit by hdm::rlasso (the reference rigorous Lasso),
# cross-fitted over a FIXED fold partition that Python reuses verbatim.
#
# Because the folds are shared (dumped in the `fold` column) and StatsPAI's
# rlasso engine is bit-exact with hdm, the partialling-out moment and its
# SE agree with sp.dml to ~1e-6 — a tight cross-ecosystem check, not a
# tolerance band.
#
# Writes:
#   _fixtures/dml_rlasso_data.csv   [y, d, x1..xp, fold]   (fold ∈ 0..K-1)
#   _fixtures/dml_rlasso_R.json     {theta, se, n_folds}
#
# Re-run only on contract change:
#   Rscript tests/reference_parity/_generate_dml_rlasso.R
#
# hdm reference: Chernozhukov, Hansen & Spindler (2016) [@chernozhukov2016hdm].
# DML score: Chernozhukov et al. (2018) [@chernozhukov2018double].

suppressMessages({
  library(hdm)
  library(jsonlite)
})

FIX <- "tests/reference_parity/_fixtures"
dir.create(FIX, showWarnings = FALSE, recursive = TRUE)

set.seed(7)
n <- 400L
p <- 20L
K <- 5L
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("x", seq_len(p))
gamma <- c(1.0, -0.7, 0.5, rep(0, p - 3))      # E[D|X] = X gamma  (sparse)
beta <- c(0.6, 0.0, -0.4, rep(0, p - 3))       # outcome controls
theta_true <- 1.5
d <- as.numeric(X %*% gamma + rnorm(n))
y <- theta_true * d + as.numeric(X %*% beta) + rnorm(n)

# Deterministic fold labels 0..K-1, reused verbatim by Python.
fold <- (seq_len(n) - 1L) %% K

write.csv(
  data.frame(y = y, d = d, X, fold = fold, check.names = FALSE),
  file.path(FIX, "dml_rlasso_data.csv"), row.names = FALSE
)

# Cross-fitted nuisances via hdm::rlasso (defaults: post=TRUE, intercept=TRUE,
# c=1.1, gamma=0.1/log(n_train), heteroskedastic) — identical to StatsPAI's
# RlassoRegressor defaults.
y_resid <- numeric(n)
d_resid <- numeric(n)
for (k in 0:(K - 1)) {
  tr <- which(fold != k)
  te <- which(fold == k)
  g <- rlasso(X[tr, , drop = FALSE], y[tr], post = TRUE, intercept = TRUE)
  m <- rlasso(X[tr, , drop = FALSE], d[tr], post = TRUE, intercept = TRUE)
  y_resid[te] <- y[te] - as.numeric(predict(g, newdata = X[te, , drop = FALSE]))
  d_resid[te] <- d[te] - as.numeric(predict(m, newdata = X[te, , drop = FALSE]))
}

# sp.dml PLR partialling-out moment + sandwich SE (plr.py).
denom <- sum(d_resid * d_resid)
theta <- sum(d_resid * y_resid) / denom
psi_score <- (y_resid - theta * d_resid) * d_resid
J <- -mean(d_resid^2)
sigma2 <- mean(psi_score^2)
se <- sqrt(sigma2 / (J^2 * n))

out <- list(
  meta = list(R_version = R.version.string,
              hdm_version = as.character(packageVersion("hdm"))),
  n = n, p = p, n_folds = K, theta_true = theta_true,
  theta = theta, se = se
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(FIX, "dml_rlasso_R.json"))
cat("DML-PLR rlasso reference: theta =", round(theta, 6),
    " se =", round(se, 6), " (truth", theta_true, ")\n")
