#!/usr/bin/env Rscript
# Second reference for sp.krippendorff_alpha: R irr::kripp.alpha (irr is
# GPL; used as a black box, outputs only). irr wants a coders x units
# matrix, so the units x coders CSVs are transposed. irr orders the
# distinct values as strings (levels(as.factor(x))), which is numeric
# order here because every value is a single digit.
#
# Regenerate (from the repo root) with:
#   Rscript tests/reference_parity/_fixtures/_generate_krippendorff_R.R

suppressPackageStartupMessages({
  library(irr)
  library(jsonlite)
})

here <- function(p) file.path("tests", "reference_parity", "_fixtures", p)
out <- list()
for (ds in c("small", "data")) {
  m <- t(as.matrix(read.csv(here(paste0("krippendorff_", ds, ".csv")))))
  r <- list()
  for (lev in c("nominal", "ordinal", "interval", "ratio")) {
    ka <- kripp.alpha(m, method = lev)
    r[[lev]] <- list(alpha = ka$value, nmatchval = ka$nmatchval,
                     subjects = ka$subjects, raters = ka$raters)
  }
  out[[ds]] <- r
}
out$versions <- list(R = R.version.string,
                     irr = as.character(packageVersion("irr")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           here("krippendorff_R.json"))
