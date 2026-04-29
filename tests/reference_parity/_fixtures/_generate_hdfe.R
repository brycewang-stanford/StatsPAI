#!/usr/bin/env Rscript
# R fixest reference values for sp.hdfe_ols parity.
suppressMessages({
  library(fixest)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/hdfe_data.csv")

# Two-way FE — entity (id) + time (year) absorption, clustered SEs at id
m <- feols(y ~ x1 + x2 | id + year, data = df, cluster = ~id)

co <- coef(m)
se <- sqrt(diag(vcov(m, cluster = ~id)))

out <- list(
  meta = list(
    R_version = R.version.string,
    fixest_version = as.character(packageVersion("fixest")),
    formula = "y ~ x1 + x2 | id + year",
    cluster = "id"
  ),
  twfe_clustered = list(
    x1 = list(coef = unname(co["x1"]), se = unname(se["x1"])),
    x2 = list(coef = unname(co["x2"]), se = unname(se["x2"]))
  ),
  n_obs = nobs(m),
  fes = list(id = nlevels(factor(df$id)), year = nlevels(factor(df$year)))
)
write_json(out, "tests/reference_parity/_fixtures/hdfe_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("x1: coef=%.6f se=%.6f\n", co["x1"], se["x1"]))
cat(sprintf("x2: coef=%.6f se=%.6f\n", co["x2"], se["x2"]))
