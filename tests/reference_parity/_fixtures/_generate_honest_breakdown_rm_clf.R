# R HonestDiD 0.2.8 (MIT): C-LF relative-magnitude sets at multiples of the
# native breakdown value (step 2; inputs from _generate_honest_breakdown_rm_clf.py).
suppressPackageStartupMessages({ library(HonestDiD); library(jsonlite) })
cases <- fromJSON("honest_breakdown_rm_clf_inputs.json", simplifyVector = TRUE)
MULT <- c(0.5, 0.9, 0.97, 1.03, 1.1, 1.5)
out <- list()
for (nm in names(cases)) {
  cs <- cases[[nm]]
  l <- rep(0, cs$npost); l[cs$e + 1] <- 1
  set.seed(0)
  r <- suppressWarnings(createSensitivityResults_relativeMagnitudes(
    betahat = cs$b, sigma = cs$S, numPrePeriods = cs$npre, numPostPeriods = cs$npost,
    Mbarvec = cs$mstar * MULT, method = "C-LF", l_vec = l, alpha = 0.05))
  out[[nm]] <- list(mstar = cs$mstar, mult = MULT, lb = r$lb, ub = r$ub)
}
out$`_provenance` <- list(HonestDiD = as.character(packageVersion("HonestDiD")),
                          R = R.version.string)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "honest_breakdown_rm_clf_R.json")
