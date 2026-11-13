# R reference for sp.srm_test: Pearson chi-square goodness of fit.
#
#   Rscript tests/reference_parity/_fixtures/_generate_gaps_srm_R.R
#
# Writes gaps_srm_R.json next to this file. Each case stores the arm counts,
# the designed shares, and chisq.test(x, p = p)'s statistic, degrees of
# freedom, p-value, expected counts, Pearson residuals and standardised
# residuals. The counts are written out here, so there is no data file.
# R 4.5.2, stats (base).
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", args[grep("--file=", args)]))

cases <- list(
  two_equal      = list(x = c(5000, 5300), p = c(1, 1) / 2),
  two_balanced   = list(x = c(5012, 4988), p = c(1, 1) / 2),
  two_unequal    = list(x = c(8123, 1877), p = c(0.8, 0.2)),
  three_equal    = list(x = c(3400, 3300, 3600), p = c(1, 1, 1) / 3),
  three_ratio    = list(x = c(5210, 2480, 2310), p = c(2, 1, 1) / 4),
  four_design    = list(x = c(412, 395, 801, 392), p = c(0.2, 0.2, 0.4, 0.2)),
  empty_arm      = list(x = c(60, 0, 40), p = c(0.5, 0.1, 0.4)),
  five_small     = list(x = c(21, 19, 25, 17, 18), p = rep(0.2, 5))
)
out <- list()
for (nm in names(cases)) {
  cs <- cases[[nm]]
  r <- suppressWarnings(chisq.test(cs$x, p = cs$p))
  out[[nm]] <- list(
    x = cs$x, p = cs$p,
    statistic = unname(r$statistic), df = unname(r$parameter),
    p_value = r$p.value, expected = as.numeric(r$expected),
    residuals = as.numeric(r$residuals), stdres = as.numeric(r$stdres)
  )
}
out[["_meta"]] <- list(R = R.version.string)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           file.path(here, "gaps_srm_R.json"))
cat("wrote gaps_srm_R.json\n")
