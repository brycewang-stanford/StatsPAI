#!/usr/bin/env Rscript
# rdrobust 4.0.0 with user-supplied asymmetric bandwidths h = c(left, right),
# b = c(left, right), on rdsenate_params.csv; with and without clusters.
#   Rscript tests/reference_parity/_fixtures/_generate_rdrobust_asym_R.R
suppressMessages({library(rdrobust); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
d <- read.csv(file.path(here, "rdsenate_params.csv"))
d <- d[!is.na(d$vote) & !is.na(d$margin), ]
out <- list()
for (cl in c(FALSE, TRUE)) {
  r <- rdrobust(y = d$vote, x = d$margin, c = 0, h = c(14, 19), b = c(22, 27),
                cluster = if (cl) d$clust else NULL)
  out[[if (cl) "cluster" else "plain"]] <- list(
    conv = r$coef[1], se_conv = r$se[1], bc = r$coef[2], se_rob = r$se[3])
}
write_json(list(meta = list(R_version = R.version.string,
                            rdrobust = as.character(packageVersion("rdrobust"))),
                cases = out),
           file.path(here, "rdrobust_asym_R.json"), digits = NA, auto_unbox = TRUE)
cat("wrote rdrobust_asym_R.json\n")
