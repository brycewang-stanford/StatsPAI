#!/usr/bin/env Rscript
# R reference values for sp.sdid parity, using the *authors' own* reference
# implementation: the synthdid package (Arkhangelsky, Athey, Hirshberg,
# Imbens & Wager 2021, AER 111(12):4088-4118; doi:10.1257/aer.20190159).
#
# Run on StatsPAI's bundled California Prop 99 panel (exported to
# sdid_prop99_data.csv). The point estimate is deterministic; the placebo
# variance estimator is randomization-based, so only the estimate is pinned
# as an exact parity target (the SE is recorded for reference only).
#
# Install: remotes::install_github("synth-inference/synthdid")
suppressMessages({
  library(synthdid)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/sdid_prop99_data.csv")
df <- df[, c("state", "year", "packspercapita", "treated")]

setup <- panel.matrices(df, unit = "state", time = "year",
                        outcome = "packspercapita", treatment = "treated")
tau <- synthdid_estimate(setup$Y, setup$N0, setup$T0)

out <- list(
  meta = list(
    R_version = R.version.string,
    synthdid_version = as.character(packageVersion("synthdid")),
    dataset = "StatsPAI sp.california_prop99() (38 control states, T0=19)",
    note = paste("Point estimate is deterministic and is the parity target;",
                 "placebo SE is randomization-based, recorded for reference")
  ),
  sdid = list(
    estimate = as.numeric(tau),
    N0 = setup$N0,
    T0 = setup$T0,
    placebo_se_reference = sqrt(vcov(tau, method = "placebo"))
  )
)
write_json(out, "tests/reference_parity/_fixtures/sdid_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("synthdid estimate: %.10f  (N0=%d T0=%d)\n",
            as.numeric(tau), setup$N0, setup$T0))
