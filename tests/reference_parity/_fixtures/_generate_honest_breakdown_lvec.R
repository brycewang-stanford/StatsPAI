# R HonestDiD 0.2.8 (MIT): sensitivity sets at multiples of the native
# breakdown value for l_vec / window targets (step 2; inputs from
# _generate_honest_breakdown_lvec.py). Run in this directory.
suppressPackageStartupMessages({ library(HonestDiD); library(jsonlite) })
cases <- fromJSON("honest_breakdown_lvec_inputs.json", simplifyVector = TRUE)
out <- list()
for (nm in names(cases)) {
  cs <- cases[[nm]]
  l <- matrix(cs$l_weights, ncol = 1)
  S <- as.matrix(cs$S_win)
  # Conditional is deterministic: a tight bracket. C-LF and the FLCI use
  # simulated quantiles (R's draws against NumPy's): a 3% bracket.
  mult <- if (cs$hm == "Conditional") c(0.5, 0.997, 1.003, 1.5) else
    c(0.5, 0.97, 1.03, 1.5)
  if (cs$method == "smoothness") {
    r <- suppressWarnings(createSensitivityResults(
      betahat = cs$b_win, sigma = S, numPrePeriods = cs$npre,
      numPostPeriods = cs$npost, Mvec = cs$mstar * mult, method = "FLCI",
      l_vec = l, alpha = 0.05))
  } else {
    r <- suppressWarnings(createSensitivityResults_relativeMagnitudes(
      betahat = cs$b_win, sigma = S, numPrePeriods = cs$npre,
      numPostPeriods = cs$npost, Mbarvec = cs$mstar * mult, method = cs$hm,
      l_vec = l, alpha = 0.05))
  }
  out[[nm]] <- list(mstar = cs$mstar, mult = mult, lb = r$lb, ub = r$ub)
}
out$`_provenance` <- list(HonestDiD = as.character(packageVersion("HonestDiD")),
                          R = R.version.string)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "honest_breakdown_lvec_R.json")
