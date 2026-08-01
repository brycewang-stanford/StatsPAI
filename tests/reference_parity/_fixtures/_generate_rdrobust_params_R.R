#!/usr/bin/env Rscript
# =====================================================================
#  rdrobust parameter-surface reference: fuzzy / covs / cluster / vce /
#  deriv.  Companion to _generate_rdrobust_R.R (which covers the
#  bwselect x p x kernel grid on the sharp, vce="nn" default path).
#
#  Writes rdrobust_params_R.json and rdsenate_params.csv.
#
#  The sharp+nn path is already exact (see test_rdrobust_parity.py).
#  This fixture exists to measure and then close the REMAINING surface,
#  where sp.rdrobust still falls back to its legacy estimator.
#
#  Environment: R 4.5.2 / rdrobust 4.0.0
# =====================================================================
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

set.seed(20260801)
data(rdrobust_RDsenate)
d <- rdrobust_RDsenate
d <- d[!is.na(d$margin) & !is.na(d$vote), ]
n <- nrow(d)
# Deterministic auxiliaries so the Python side sees identical bytes.
d$cov1 <- as.numeric(scale(seq_len(n) %% 17))
d$cov2 <- as.numeric(scale((seq_len(n) * 7) %% 23))
d$clust <- (seq_len(n) %% 50) + 1
# One-sided-noncompliance treatment for the fuzzy specs.
d$treat <- as.numeric((d$margin >= 0) & (seq_len(n) %% 10 != 0))
write.csv(d, file.path(OUT, "rdsenate_params.csv"), row.names = FALSE)

grab <- function(r) list(
  coef_conventional = r$coef[1], coef_robust = r$coef[3],
  se_conventional = r$se[1], se_robust = r$se[3],
  h_left = r$bws[1, 1], h_right = r$bws[1, 2],
  b_left = r$bws[2, 1], b_right = r$bws[2, 2]
)
out <- list()
add <- function(key, expr) {
  r <- try(expr, silent = TRUE)
  if (inherits(r, "try-error")) { cat("FAILED", key, "\n"); return(invisible()) }
  out[[key]] <<- grab(r); cat("ok", key, "\n")
}

for (p in c(1, 2)) {
  add(paste0("covs_p", p),
      rdrobust(y = d$vote, x = d$margin, c = 0, p = p,
               covs = cbind(d$cov1, d$cov2)))
  add(paste0("fuzzy_p", p),
      rdrobust(y = d$vote, x = d$margin, c = 0, p = p, fuzzy = d$treat))
  add(paste0("cluster_p", p),
      rdrobust(y = d$vote, x = d$margin, c = 0, p = p, cluster = d$clust))
}
for (v in c("hc0", "hc1", "hc2", "hc3")) {
  add(paste0("vce_", v), rdrobust(y = d$vote, x = d$margin, c = 0, vce = v))
}
add("deriv1", rdrobust(y = d$vote, x = d$margin, c = 0, p = 2, deriv = 1))
add("covs_fuzzy",
    rdrobust(y = d$vote, x = d$margin, c = 0, fuzzy = d$treat,
             covs = cbind(d$cov1, d$cov2)))

out[["_meta"]] <- list(
  generated_by = "_generate_rdrobust_params_R.R",
  rdrobust_version = as.character(packageVersion("rdrobust")),
  r_version = R.version.string, n = n, n_specs = length(out) - 1
)
write(toJSON(out, digits = 14, auto_unbox = TRUE), file.path(OUT, "rdrobust_params_R.json"))
cat("[params-fixture] wrote", length(out) - 1, "specs\n")
