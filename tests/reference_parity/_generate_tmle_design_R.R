# R reference for tests/reference_parity/test_tmle_design_R_parity.py
# (sp.tmle(weights=, cluster=) against tmle::tmle(obsWeights=, id=)).
#
# Generates the data here and writes both sides' bytes:
#   _fixtures/tmle_design_data.csv, _fixtures/tmle_design_R.json
# Run from tests/reference_parity/:  Rscript _generate_tmle_design_R.R
#
# Options match test_teffects_R_parity.py's TMLE rows: SL.glm for both
# nuisances, gbound = 0.025, cvQinit = FALSE (initial Q fitted on the full
# sample), tmle's default two-covariate fluctuation. Clusters all have 12
# rows: tmle averages the influence function within id, which equals the
# usual cluster sum (with G/(G-1)) only when cluster sizes are equal.

suppressMessages({
  library(tmle)
  library(jsonlite)
})

set.seed(20260926)
G <- 80; m <- 12; n <- G * m
g <- rep(seq_len(G), each = m)
u <- rnorm(G)[g]
x1 <- rnorm(n) + 0.5 * u
x2 <- rnorm(n)
x3 <- runif(n)
w <- exp(0.4 * x2 + 0.3 * runif(n))
d <- rbinom(n, 1, plogis(-0.2 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3))
y <- 1 + (1.2 + 0.6 * x2) * d + 0.7 * x1 + 0.3 * x2 - 0.5 * x3 + u + rnorm(n)
yb <- rbinom(n, 1, plogis(-0.3 + (0.8 + 0.4 * x2) * d + 0.5 * x1 - 0.3 * x3 + 0.5 * u))
dat <- data.frame(g = g, x1 = x1, x2 = x2, x3 = x3, w = w, d = d, y = y, yb = yb)
write.csv(dat, "_fixtures/tmle_design_data.csv", row.names = FALSE)
dat <- read.csv("_fixtures/tmle_design_data.csv")  # the bytes Python reads

W <- dat[, c("x1", "x2", "x3")]
fit <- function(Y, family, id = NULL, obsWeights = NULL) {
  args <- list(Y = Y, A = dat$d, W = W, family = family,
               Q.SL.library = "SL.glm", g.SL.library = "SL.glm",
               gbound = 0.025, cvQinit = FALSE)
  if (!is.null(id)) args$id <- id
  if (!is.null(obsWeights)) args$obsWeights <- obsWeights
  f <- do.call(tmle, args)
  list(psi = f$estimates$ATE$psi, se = sqrt(f$estimates$ATE$var.psi))
}
out <- list()
for (fam in c("gaussian", "binomial")) {
  Y <- if (fam == "gaussian") dat$y else dat$yb
  out[[paste0(fam, "_plain")]] <- fit(Y, fam)
  out[[paste0(fam, "_cluster")]] <- fit(Y, fam, id = dat$g)
  out[[paste0(fam, "_weights")]] <- fit(Y, fam, obsWeights = dat$w)
  out[[paste0(fam, "_weights_cluster")]] <- fit(Y, fam, id = dat$g, obsWeights = dat$w)
}
out[["_meta"]] <- list(
  r_version = R.version.string,
  tmle = as.character(packageVersion("tmle")),
  SuperLearner = as.character(packageVersion("SuperLearner")),
  generated = format(Sys.Date())
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "_fixtures/tmle_design_R.json")
cat("wrote _fixtures/tmle_design_data.csv and _fixtures/tmle_design_R.json\n")
