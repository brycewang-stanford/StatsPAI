#!/usr/bin/env Rscript
# ShiftShareSE 1.1.0 (GPL-3, run as a black box; no source read) on the
# designs of _generate_akm_ci_shapes_data.py: every inference row of
# ivreg_ss.fit and reg_ss.fit, at beta0 = 0.5 and alpha = 0.10.
#   python tests/reference_parity/_fixtures/_generate_akm_ci_shapes_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_akm_ci_shapes_R.R
suppressMessages({library(ShiftShareSE); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
loc <- read.csv(file.path(here, "akm_ci_shapes_loc.csv"))
Wall <- as.matrix(read.csv(file.path(here, "akm_ci_shapes_shares.csv"), header = FALSE))
pack <- function(r) list(beta = unname(r$beta), se = as.list(r$se), p = as.list(r$p),
                         ci_l = as.list(r$ci.l), ci_r = as.list(r$ci.r))
out <- list(meta = list(R = R.version.string,
                        ShiftShareSE = as.character(packageVersion("ShiftShareSE"))))
for (d in sort(unique(loc$design))) {
  i <- loc$design == d
  L <- loc[i, ]; W <- Wall[i, ]
  Z <- cbind(1, L$c1)
  w <- if (d >= 4) L$w else NULL
  m <- c("homosk", "ehw", "region_cluster", "akm", "akm0")
  iv <- suppressWarnings(ivreg_ss.fit(y1 = L$y, y2 = L$x, X = L$X, W = W, Z = Z, w = w,
                                      method = m, beta0 = 0.5, alpha = 0.10,
                                      region_cvar = L$region))
  ols <- suppressWarnings(reg_ss.fit(y = L$y, X = L$X, W = W, Z = Z, w = w,
                                     method = m, beta0 = 0.5, alpha = 0.10,
                                     region_cvar = L$region))
  out[[paste0("d", d)]] <- list(iv = pack(iv), ols = pack(ols))
}
write_json(out, file.path(here, "akm_ci_shapes_R.json"), digits = I(17),
           auto_unbox = TRUE, na = "string")
cat("wrote akm_ci_shapes_R.json\n")
