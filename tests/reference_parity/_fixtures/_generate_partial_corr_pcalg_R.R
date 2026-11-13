# Reference p-values for sp.partial_corr_pvalue: pcalg::gaussCItest (the
# Fisher z test of a zero partial correlation) on ness_gaussian.csv, every
# pair of the eight columns with conditioning sets of size 0 to 3 drawn
# with a fixed seed, on the full sample and on its first 60 rows.
# ppcor::pcor.test is recorded next to it when available: it tests the same
# partial correlation with a t statistic, so only the correlation itself
# is comparable. pcalg is GPL; black box.
#
#   Rscript _generate_partial_corr_pcalg_R.R
suppressMessages({
  library(pcalg)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0) here <- "."
d_all <- read.csv(file.path(here, "ness_gaussian.csv"))
set.seed(20261113)
rows <- list()
for (n in c(nrow(d_all), 60)) {
  d <- d_all[seq_len(n), ]
  suff <- list(C = cor(d), n = n)
  p <- ncol(d)
  for (i in 1:(p - 1)) for (j in (i + 1):p) for (k in 0:3) {
    rest <- setdiff(seq_len(p), c(i, j))
    S <- if (k == 0) integer(0) else sort(sample(rest, k))
    rows[[length(rows) + 1]] <- list(
      n = n, x = names(d)[i], y = names(d)[j], z = I(names(d)[S]),
      pvalue = gaussCItest(i, j, S, suff)
    )
  }
}
res <- list(
  versions = list(R = R.version.string,
                  pcalg = as.character(packageVersion("pcalg"))),
  rows = rows
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA),
           file.path(here, "partial_corr_pcalg_R.json"))
cat("rows:", length(rows), "\n")
