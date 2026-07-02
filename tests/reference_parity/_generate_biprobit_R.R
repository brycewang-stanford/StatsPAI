#!/usr/bin/env Rscript
# Frozen reference for sp.biprobit vs R VGAM::vglm(binom2.rho) bivariate probit.
# Both maximize the same joint likelihood, so slope/intercept coefficients and
# the error correlation rho match to machine precision. VGAM parameterizes rho
# through the rhobit link on the third linear predictor; rho = tanh(rhobit/2).
# Regenerate: Rscript tests/reference_parity/_generate_biprobit_R.R
suppressPackageStartupMessages({library(VGAM); library(jsonlite)})
df <- read.csv("tests/reference_parity/_fixtures/biprobit_data.csv")

fit <- vglm(cbind(y1, y2) ~ x, binom2.rho, data = df)
co <- coef(fit)
rho <- tanh(unname(co["(Intercept):3"]) / 2)

out <- list(
  eq1_intercept = unname(co["(Intercept):1"]),
  eq1_x = unname(co["x:1"]),
  eq2_intercept = unname(co["(Intercept):2"]),
  eq2_x = unname(co["x:2"]),
  rho = rho,
  loglik = as.numeric(logLik(fit)),
  provenance = list(
    r_version = R.version.string,
    VGAM_version = as.character(packageVersion("VGAM")),
    generated_by = "tests/reference_parity/_generate_biprobit_R.R"
  )
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/biprobit_R.json")
cat("wrote biprobit_R.json\n")
