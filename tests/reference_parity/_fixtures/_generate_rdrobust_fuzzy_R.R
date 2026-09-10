#!/usr/bin/env Rscript
# =====================================================================
#  rdrobust fuzzy-RD reference with TWO-SIDED noncompliance.
#
#  Why a second fixture rather than another spec in
#  _generate_rdrobust_params_R.R: that file's `treat` is one-sided
#  (nobody below the cutoff is treated), which trips rdbwselect's
#  `perf_comp` branch -- var(T_l) == 0 makes it drop T and fall back to
#  the SHARP bandwidth. A one-sided design therefore cannot exercise the
#  fuzzy bandwidth path at all; the h it reports is the sharp h, and a
#  test built on it would pass with the fuzzy V/B machinery entirely
#  absent. Two-sided noncompliance is what puts the delta-method
#  weights into rdrobust_bw.
#
#  Writes rdrobust_fuzzy_R.json and rdsenate_fuzzy.csv.
#  Environment: R 4.5.2 / rdrobust 4.0.0
# =====================================================================
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

set.seed(20260911)
data(rdrobust_RDsenate)
d <- rdrobust_RDsenate
d <- d[!is.na(d$margin) & !is.na(d$vote), ]
n <- nrow(d)
d$cov1 <- as.numeric(scale(seq_len(n) %% 17))
d$cov2 <- as.numeric(scale((seq_len(n) * 7) %% 23))
d$clust <- (seq_len(n) %% 50) + 1
# Two-sided noncompliance, deterministic so the Python side sees the
# same bytes: 15% of the eligible do not take up, 10% of the ineligible
# do. Both sides carry variation, so rdbwselect keeps the fuzzy branch.
above <- d$margin >= 0
d$treat2 <- as.numeric(ifelse(above, seq_len(n) %% 20 >= 3, seq_len(n) %% 10 == 0))
stopifnot(var(d$treat2[!above]) > 0, var(d$treat2[above]) > 0)
write.csv(d, file.path(OUT, "rdsenate_fuzzy.csv"), row.names = FALSE)

grab <- function(r) list(
  coef_conventional = r$coef[1], coef_robust = r$coef[3],
  se_conventional = r$se[1], se_robust = r$se[3],
  h_left = r$bws[1, 1], h_right = r$bws[1, 2],
  b_left = r$bws[2, 1], b_right = r$bws[2, 2]
)
out <- list()
add <- function(key, expr) {
  r <- try(expr, silent = TRUE)
  if (inherits(r, "try-error")) { cat("FAILED", key, ":", conditionMessage(attr(r, "condition")), "\n"); return(invisible()) }
  out[[key]] <<- grab(r); cat("ok", key, "\n")
}

for (p in c(1, 2)) {
  add(paste0("fuzzy2_p", p),
      rdrobust(y = d$vote, x = d$margin, c = 0, p = p, fuzzy = d$treat2))
  add(paste0("fuzzy2_covs_p", p),
      rdrobust(y = d$vote, x = d$margin, c = 0, p = p, fuzzy = d$treat2,
               covs = cbind(d$cov1, d$cov2)))
}
add("fuzzy2_cluster",
    rdrobust(y = d$vote, x = d$margin, c = 0, fuzzy = d$treat2,
             cluster = d$clust))
for (v in c("hc0", "hc1", "hc2", "hc3")) {
  add(paste0("fuzzy2_vce_", v),
      rdrobust(y = d$vote, x = d$margin, c = 0, fuzzy = d$treat2, vce = v))
}
add("fuzzy2_uniform",
    rdrobust(y = d$vote, x = d$margin, c = 0, fuzzy = d$treat2,
             kernel = "uniform"))
add("fuzzy2_cerrd",
    rdrobust(y = d$vote, x = d$margin, c = 0, fuzzy = d$treat2,
             bwselect = "cerrd"))
add("fuzzy2_msetwo",
    rdrobust(y = d$vote, x = d$margin, c = 0, fuzzy = d$treat2,
             bwselect = "msetwo"))
# Bandwidth-only reference: this is the quantity a one-sided design
# cannot reach, so it is pinned separately from the estimate.
bwr <- rdbwselect(y = d$vote, x = d$margin, c = 0, fuzzy = d$treat2)
out[["fuzzy2_bwselect"]] <- list(h_left = bwr$bws[1, 1], h_right = bwr$bws[1, 2],
                                 b_left = bwr$bws[1, 3], b_right = bwr$bws[1, 4])
cat("ok fuzzy2_bwselect\n")

out[["_meta"]] <- list(
  r_version = R.version.string,
  rdrobust_version = as.character(packageVersion("rdrobust")),
  generated = format(Sys.time(), "%Y-%m-%d"),
  design = "two-sided noncompliance; both sides carry first-stage variation"
)
write_json(out, file.path(OUT, "rdrobust_fuzzy_R.json"),
           auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rdrobust_fuzzy_R.json\n")
