# Reference values for sp.rd_honest vs R RDHonest 1.0.1.9000
# (Armstrong & Kolesar 2018, 2020).
#
# Structured so a mismatch localises. Cells with BOTH M and h fixed isolate
# the CI formula from the selectors; cells with M fixed and h chosen isolate
# the bandwidth rule; the free cell exercises the whole chain.
library(RDHonest); library(jsonlite)
set.seed(20260802)

data(lee08)
write.csv(data.frame(y = lee08$voteshare, x = lee08$margin),
          "rdhonest_lee08.csv", row.names = FALSE)

# A synthetic design with a KNOWN curvature so M is not a free parameter:
# f(x) = 0.5x + 1.5x^2 on each side => |f''| = 3.
n <- 4000
xs <- runif(n, -1, 1)
ys <- 0.5 * xs + 1.5 * xs^2 + 2.0 * (xs >= 0) + rnorm(n, 0, 0.4)
write.csv(data.frame(y = ys, x = xs), "rdhonest_curved.csv", row.names = FALSE)

grab <- function(r) {
  co <- r$coefficients
  list(
    estimate  = unname(co$estimate),
    se        = unname(co$std.error),
    bias      = unname(co$maximum.bias),
    ci_lower  = unname(co$conf.low),
    ci_upper  = unname(co$conf.high),
    h         = unname(co$bandwidth),
    eff_obs   = unname(co$eff.obs),
    M         = unname(co$M),
    leverage  = unname(co$leverage)
  )
}

out <- list()
datasets <- list(
  lee08  = data.frame(y = lee08$voteshare, x = lee08$margin),
  curved = data.frame(y = ys, x = xs)
)

for (nm in names(datasets)) {
  d <- datasets[[nm]]
  # (a) M and h BOTH fixed: pure CI-formula comparison.
  for (M in c(0.5, 2, 6)) {
    for (h in c(5, 10)) {
      hh <- if (nm == "curved") h / 20 else h   # curved x lives on [-1, 1]
      MM <- if (nm == "curved") M else M / 100  # lee08 y is a vote share
      out[[sprintf("%s_fixed_M%g_h%g", nm, M, h)]] <- grab(
        RDHonest(y ~ x, data = d, M = MM, h = hh, kern = "triangular")
      )
    }
  }
  # (b) M fixed, h chosen by each criterion.
  for (crit in c("MSE", "FLCI")) {
    MM <- if (nm == "curved") 3 else 0.02
    out[[sprintf("%s_bwsel_%s", nm, crit)]] <- grab(
      RDHonest(y ~ x, data = d, M = MM, opt.criterion = crit,
               kern = "triangular")
    )
  }
  # (c) everything data-driven, including M.
  out[[sprintf("%s_free", nm)]] <- grab(
    RDHonest(y ~ x, data = d, kern = "triangular")
  )
  # (d) kernel variation, M and h fixed.
  for (k in c("uniform", "epanechnikov")) {
    MM <- if (nm == "curved") 3 else 0.02
    hh <- if (nm == "curved") 0.4 else 8
    out[[sprintf("%s_kern_%s", nm, k)]] <- grab(
      RDHonest(y ~ x, data = d, M = MM, h = hh, kern = k)
    )
  }
}

# The critical-value function itself: honest CIs are estimate +/-
# cv_{1-alpha}(bias/se) * se, NOT estimate +/- (bias + z*se).
out[["cvb"]] <- list(
  t     = c(0, 0.5, 1, 2, 4, 8),
  cv95  = unname(as.numeric(CVb(c(0, 0.5, 1, 2, 4, 8), alpha = 0.05))),
  cv90  = unname(as.numeric(CVb(c(0, 0.5, 1, 2, 4, 8), alpha = 0.10)))
)

out[["_meta"]] <- list(
  n_lee08 = nrow(lee08), n_curved = n,
  true_M_curved = 3, true_jump_curved = 2.0,
  RDHonest_version = as.character(packageVersion("RDHonest"))
)
write_json(out, "rdhonest_R.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rdhonest_R.json with", length(out) - 1, "cells\n")
