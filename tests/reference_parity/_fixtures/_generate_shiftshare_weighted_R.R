#!/usr/bin/env Rscript
# ShiftShareSE 1.1.0 ivreg_ss.fit / reg_ss.fit with location weights w.
#   python tests/reference_parity/_fixtures/_generate_shiftshare_weighted_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_shiftshare_weighted_R.R
suppressMessages({library(ShiftShareSE); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
loc <- read.csv(file.path(here, "shiftshare_weighted_loc.csv"))
W <- as.matrix(read.csv(file.path(here, "shiftshare_weighted_shares.csv"), header = FALSE))
Z <- cbind(1, loc$c1, loc$c2)
iv <- ivreg_ss.fit(y1 = loc$y, y2 = loc$x, X = loc$z, W = W, Z = Z, w = loc$w,
                   method = c("homosk", "ehw", "akm", "akm0"))
ols <- reg_ss.fit(y = loc$y, X = loc$z, W = W, Z = Z, w = loc$w,
                  method = c("homosk", "ehw", "akm", "akm0"))
pack <- function(r) list(beta = unname(r$beta), se = as.list(r$se),
                         ci_l = as.list(r$ci.l), ci_r = as.list(r$ci.r))
write_json(list(meta = list(R = R.version.string,
                            ShiftShareSE = as.character(packageVersion("ShiftShareSE"))),
                iv = pack(iv), ols = pack(ols)),
           file.path(here, "shiftshare_weighted_R.json"), digits = NA, auto_unbox = TRUE)
cat("wrote shiftshare_weighted_R.json\n")
