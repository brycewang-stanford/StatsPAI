#!/usr/bin/env Rscript
# R nprobust 1.0.0 (GPL-3), run as a BLACK BOX for the clean-room rewrite of
# sp.lprobust_at_point / sp.lpbwselect_* (docs/dev/cleanroom/lprobust.md).
# Only exported functions are called; their outputs are recorded.
#   python tests/reference_parity/_fixtures/_generate_lprobust_selectors_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_lprobust_selectors_R.R
suppressMessages({library(nprobust); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
d <- read.csv(file.path(here, "lprobust_selectors_data.csv"))
kern <- c(epanechnikov = "epa", triangular = "tri", uniform = "uni", gaussian = "gau")
evals <- list(smooth = c(-0.5, 0.4, 1.3), ties = c(-1, 0, 0.7))
orders <- list(c(1, 0), c(2, 1), c(3, 0))
bws <- list(); fits <- list(); glob <- list()
for (des in names(evals)) {
  s <- d[d$design == des, ]
  for (k in names(kern)) {
    for (e in evals[[des]]) {
      for (pd in orders) {
        for (opt in list(c(21, 1), c(80, 0))) {
          for (sel in c("mse-dpi", "ce-rot")) {
            r <- lpbwselect(s$y, s$x, eval = e, p = pd[1], deriv = pd[2], kernel = kern[[k]],
                            bwselect = sel, bwcheck = opt[1], bwregul = opt[2])$bws
            bws[[length(bws) + 1]] <- list(design = des, selector = sel, kernel = k, eval = e,
              p = pd[1], deriv = pd[2], bwcheck = opt[1], bwregul = opt[2],
              h = r[1, 2], b = r[1, 3])
          }
        }
        if (k != "gaussian") {
          r <- lpbwselect(s$y, s$x, eval = e, p = pd[1], deriv = pd[2], kernel = kern[[k]],
                          bwselect = "mse-rot")$bws
          bws[[length(bws) + 1]] <- list(design = des, selector = "mse-rot", kernel = k, eval = e,
            p = pd[1], deriv = pd[2], bwcheck = 21, bwregul = 1, h = r[1, 2], b = NA)
        }
      }
      for (pd in list(c(0, 0), c(1, 0), c(1, 1), c(2, 0))) {
        for (nn in c(1, 3)) {
          h <- 0.8; b <- 1.1
          est <- lprobust(s$y, s$x, eval = e, h = h, b = b, p = pd[1], deriv = pd[2],
                          kernel = kern[[k]], nnmatch = nn, bwcheck = 1)$Estimate
          fits[[length(fits) + 1]] <- list(design = des, kernel = k, eval = e, p = pd[1],
            deriv = pd[2], nnmatch = nn, h = h, b = b, N = est[1, 4], tau_us = est[1, 5],
            tau_bc = est[1, 6], se_us = est[1, 7], se_rb = est[1, 8])
        }
      }
    }
    for (pd in orders) {
      for (g in c(7, 30)) {
        r <- lpbwselect(s$y, s$x, eval = 0, p = pd[1], deriv = pd[2], kernel = kern[[k]],
                        bwselect = "imse-dpi", imsegrid = g)$bws
        glob[[length(glob) + 1]] <- list(design = des, selector = "imse-dpi", kernel = k,
          p = pd[1], deriv = pd[2], imsegrid = g, h = r[1, 2], b = r[1, 3])
      }
      if (k != "gaussian") {
        r <- lpbwselect(s$y, s$x, eval = 0, p = pd[1], deriv = pd[2], kernel = kern[[k]],
                        bwselect = "imse-rot")$bws
        glob[[length(glob) + 1]] <- list(design = des, selector = "imse-rot", kernel = k,
          p = pd[1], deriv = pd[2], imsegrid = 30, h = r[1, 2], b = NA)
      }
    }
  }
}
write_json(list(
  meta = list(R_version = R.version.string,
              nprobust = as.character(packageVersion("nprobust")),
              note = "black-box outputs only; lprobust run with bwcheck = 1 so h and b are used as given"),
  pointwise = bws, at_point = fits, global = glob),
  file.path(here, "lprobust_selectors_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
cat("wrote lprobust_selectors_R.json\n")
