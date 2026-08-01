# Reference values for sp.rdrandinf / sp.rdwinselect vs R rdlocrand 2.0
# (Cattaneo, Titiunik & Vazquez-Bare) and sp.rdpower / sp.rdsampsi vs
# rdpower 3.0.
library(rdlocrand); library(rdpower); library(jsonlite)
set.seed(20260802)

d <- read.csv("rdsenate.csv")
X <- d$margin; Y <- d$vote

out <- list()

# ---- rdrandinf on fixed windows (no window selection, so the numbers are
# ---- a pure function of the data and the statistic) -----------------------
for (w in c(2.5, 5, 10)) {
  for (st in c("diffmeans", "ttest", "ranksum")) {
    # reps must be > 0, but the randomization p-value it produces is a
    # draw from R's RNG that Python cannot reproduce. Only the DETERMINISTIC
    # quantities are pinned: the observed statistic and the asymptotic
    # p-value. The randomization p-value is checked separately, by its
    # sampling behaviour, not by equality.
    r <- rdrandinf(Y, X, wl = -w, wr = w, statistic = st, reps = 1000,
                   quietly = TRUE)
    out[[sprintf("randinf_w%g_%s", w, st)]] <- list(
      obs_stat = unname(r$obs.stat), asy_pvalue = unname(r$asy.pvalue),
      Nl = unname(r$sumstats[2, 1]), Nr = unname(r$sumstats[2, 2]),
      wl = -w, wr = w
    )
  }
}

# ---- rdwinselect: the window sequence and balance p-values ----------------
covs <- cbind(d$class, d$termshouse)
r <- rdwinselect(X, covs, wmin = 0.5, wstep = 0.5, nwindows = 6,
                 reps = 1000, quietly = TRUE)
# columns are: p-value, Variable, Bi.test, Obs<c, Obs>=c, w_left, w_right
out[["winselect"]] <- list(
  p_value = unname(r$results[, "p-value"]),
  Nl      = unname(r$results[, "Obs<c"]),
  Nr      = unname(r$results[, "Obs>=c"]),
  w_left  = unname(r$results[, "w_left"]),
  w_right = unname(r$results[, "w_right"])
)

# ---- rdpower / rdsampsi: closed-form, so exact parity is the bar ---------
for (tau in c(1, 3, 5)) {
  # rdpower/rdsampsi derive the variances and effective Ns from the data
  # via rdrobust, so the fixture pins the whole chain, not just the closed
  # form on top of it.
  p <- rdpower(data = cbind(Y, X), tau = tau, plot = FALSE)
  out[[sprintf("power_tau%g", tau)]] <- list(
    power_rbc = unname(p$power.rbc), se_rbc = unname(p$se.rbc),
    power_conv = unname(p$power.conv), se_conv = unname(p$se.conv),
    Nh_l = unname(p$Nh.l), Nh_r = unname(p$Nh.r),
    h_l = unname(p$samph.l), h_r = unname(p$samph.r)
  )
  ss <- rdsampsi(data = cbind(Y, X), tau = tau, plot = FALSE)
  out[[sprintf("sampsi_tau%g", tau)]] <- list(
    n_total = unname(ss$sampsi.h.tot),
    n_left = unname(ss$sampsi.h.l), n_right = unname(ss$sampsi.h.r),
    Nh_l = unname(ss$Nh.l), Nh_r = unname(ss$Nh.r)
  )
}

out[["_meta"]] <- list(
  n = nrow(d),
  rdlocrand_version = as.character(packageVersion("rdlocrand")),
  rdpower_version = as.character(packageVersion("rdpower"))
)
write_json(out, "rdlocrand_R.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rdlocrand_R.json\n")
