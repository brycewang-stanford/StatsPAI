#!/usr/bin/env Rscript
# Joint inference for hdm::rlassoEffects on the existing fixture
# tests/reference_parity/_fixtures/rlasso_effect.csv (written by
# _generate_rlasso.R). Exports the covariance that
# confint(<rlassoEffects>, joint = TRUE) simulates from, computed exactly as
# hdm:::confint.rlassoEffects does, and one simulated joint interval.
#
#   Rscript tests/reference_parity/_generate_rlasso_effects_joint.R
#
# [@chernozhukov2016hdm]

suppressMessages({
  library(hdm)
  library(jsonlite)
})
FIX <- "tests/reference_parity/_fixtures"
D <- read.csv(file.path(FIX, "rlasso_effect.csv"))
y <- D$y
X <- as.matrix(D[, !(names(D) %in% c("y"))])
idx <- 1:6
out <- list(hdm_version = as.character(packageVersion("hdm")), index = idx - 1)
for (m in c("partialling out", "double selection")) {
  r <- rlassoEffects(X, y, index = idx, method = m)
  e <- r$residuals$e
  v <- r$residuals$v
  ev <- e * v
  Ev2 <- colMeans(v^2)
  k <- length(idx)
  Omega <- matrix(NA, k, k)
  for (j in 1:k) for (l in 1:k) Omega[j, l] <- 1 / (Ev2[j] * Ev2[l]) * mean(ev[, j] * ev[, l])
  set.seed(1)
  ci <- confint(r, joint = TRUE)
  hatc <- (ci[, 2] - ci[, 1]) / 2 / sqrt(diag(Omega))  # = sup-t critical value / sqrt(n)
  out[[gsub(" ", "_", m)]] <- list(
    names = names(r$coefficients), coef = as.numeric(r$coefficients), se = as.numeric(r$se),
    omega_over_n = Omega / nrow(X), joint_ci = unname(ci),
    critical_value_B500 = as.numeric(hatc[1] * sqrt(nrow(X))))
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(FIX, "rlasso_effects_joint_R.json"))
cat("wrote rlasso_effects_joint_R.json\n")
