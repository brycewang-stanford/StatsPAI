#!/usr/bin/env Rscript
# Base-R reference values for the g-computation (g-formula) parity test.
#
# Run ONCE from the repo root; the output (gformula_R.json) is committed
# as the golden fixture against which sp.g_computation is validated.
# Re-run only when the underlying DGP changes (_generate_gformula_data.py).
#
# DELIBERATELY base R only (stats::lm + hand-rolled JSON via sprintf):
# no CRAN dependencies, so the fixture is reproducible on any R install.
#
# Model form mirrors sp.g_computation's default Q-model exactly
# (single additive OLS of y on (d, x1, x2, x3) with intercept):
#
#   fit <- lm(y ~ d + x1 + x2 + x3)
#   psi <- mean(predict(fit, d=1)) - mean(predict(fit, d=0))
#
# Under this additive model psi collapses to coef(fit)["d"] algebraically
# (predictions at d=1 and d=0 differ only by the d coefficient); the
# stopifnot below asserts that identity to 1e-12 as a script self-check.
#
# Bootstrap SE is NOT pinned here: bootstrap resampling RNG streams differ
# across languages, so an R bootstrap SE would not be comparable draw-by-draw.
# Instead we export the classical OLS SE of coef d — on this homoskedastic
# Gaussian DGP the nonparametric-bootstrap SE of psi converges to the same
# quantity, and the Python test compares only loosely (recovery-grade band).

csv_path <- "tests/reference_parity/_fixtures/gformula_data.csv"
df <- read.csv(csv_path)

fit <- lm(y ~ d + x1 + x2 + x3, data = df)

d1 <- df; d1$d <- 1
d0 <- df; d0$d <- 0
psi <- mean(predict(fit, newdata = d1)) - mean(predict(fit, newdata = d0))

coef_d <- unname(coef(fit)["d"])
se_d <- unname(summary(fit)$coefficients["d", "Std. Error"])

# Self-check: additive-model collapse identity psi == beta_d.
stopifnot(abs(psi - coef_d) < 1e-12)

# Hand-rolled JSON (base R only). %.17g preserves full double precision.
num <- function(x) sprintf("%.17g", x)
json <- paste0(
  "{\n",
  "  \"meta\": {\n",
  "    \"R_version\": \"", R.version.string, "\",\n",
  "    \"model\": \"lm(y ~ d + x1 + x2 + x3)\",\n",
  "    \"psi_def\": \"mean(predict(fit, d=1)) - mean(predict(fit, d=0))\",\n",
  "    \"n\": ", nrow(df), ",\n",
  "    \"true_ate\": 1.2\n",
  "  },\n",
  "  \"psi\": ", num(psi), ",\n",
  "  \"coef_d\": ", num(coef_d), ",\n",
  "  \"se_d_classical\": ", num(se_d), "\n",
  "}\n"
)

out_path <- "tests/reference_parity/_fixtures/gformula_R.json"
writeLines(json, out_path)
cat("Wrote", out_path, "\n")
cat(sprintf("  psi=%.10f  coef_d=%.10f  se_d=%.10f\n", psi, coef_d, se_d))
