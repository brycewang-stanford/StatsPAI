#!/usr/bin/env Rscript
# ShiftShareSE 1.1.0 AKM inference on a share matrix with collinear columns:
# the columns qr() keeps (dqrdc2, tol 1e-7) and ivreg_ss.fit's AKM SE.
#   python tests/reference_parity/_fixtures/_generate_akm_collinear_shares_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_akm_collinear_shares_R.R
suppressMessages({library(ShiftShareSE); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
loc <- read.csv(file.path(here, "akm_collinear_loc.csv"))
W <- as.matrix(read.csv(file.path(here, "akm_collinear_shares.csv"), header = FALSE))
q <- qr(W)
Z <- cbind(1, loc$c1, loc$c2)
r <- suppressWarnings(ivreg_ss.fit(y1 = loc$y, y2 = loc$x, X = loc$z, W = W, Z = Z,
                                    method = c("ehw", "akm", "akm0")))
write_json(list(
  meta = list(R_version = R.version.string,
              ShiftShareSE = as.character(packageVersion("ShiftShareSE"))),
  rank = q$rank, keep = sort(q$pivot[seq_len(q$rank)] - 1),
  beta = unname(r$beta), se_ehw = unname(r$se["EHW"]), se_akm = unname(r$se["AKM"]),
  akm0_lo = unname(r$ci.l["AKM0"]), akm0_hi = unname(r$ci.r["AKM0"])),
  file.path(here, "akm_collinear_shares_R.json"), digits = NA, auto_unbox = TRUE)
cat("wrote akm_collinear_shares_R.json\n")
