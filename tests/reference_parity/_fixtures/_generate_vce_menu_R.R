# Reference values for the variance-estimator menu of sp.regress / sp.iv.
#
# Reads vce_menu_data.csv (written by _generate_vce_menu_data.py) and fits
#   OLS   y ~ d + x1 + x2 + x3               (lm, unweighted and with weights = w)
#   2SLS  y ~ d + x1 + x2 + x3 | z1 + z2 + x1 + x2 + x3   (over-identified)
#         y ~ d + x1 + x2 + x3 | z1 + x1 + x2 + x3        (just-identified)
# under the variance estimators of the canonical R implementations:
#   sandwich      vcovHC HC0-HC4, vcovCL (CR1), NeweyWest
#   clubSandwich  vcovCR CR2 / CR3 (Pustejovsky-Tipton)
#   fixest        multiway clustering at its default ssc()
#   fwildclusterboot  WCR wild cluster bootstrap after lm, with B >= 2^G so
#                 that every Rademacher sign vector is enumerated and the
#                 p-value is deterministic
#
# Run from this directory:  Rscript _generate_vce_menu_R.R
# Writes vce_menu_R.json. R 4.5.2; sandwich 3.1.1, clubSandwich 0.6.2,
# ivreg 0.6.7, fixest 0.14.0, fwildclusterboot 0.14.3.
suppressPackageStartupMessages({
  library(sandwich)
  library(clubSandwich)
  library(ivreg)
  library(fixest)
  library(fwildclusterboot)
  library(jsonlite)
})

d <- read.csv("vce_menu_data.csv")
stopifnot(nrow(d) == 360)

grab <- function(b, V) {
  list(b = as.list(b), se = as.list(sqrt(diag(as.matrix(V)))))
}

out <- list()

# ---------------------------------------------------------------- OLS ----
for (wt in c("none", "w")) {
  fit <- if (wt == "none") lm(y ~ d + x1 + x2 + x3, data = d) else
    lm(y ~ d + x1 + x2 + x3, data = d, weights = w)
  sfx <- if (wt == "none") "" else "_w"
  b <- coef(fit)
  out[[paste0("ols_classical", sfx)]] <- grab(b, vcov(fit))
  for (ty in c("HC0", "HC1", "HC2", "HC3", "HC4")) {
    out[[paste0("ols_", tolower(ty), sfx)]] <- grab(b, vcovHC(fit, type = ty))
  }
  out[[paste0("ols_cr1", sfx)]] <- grab(b, vcovCL(fit, cluster = ~g, type = "HC1"))
  for (ty in c("CR2", "CR3")) {
    out[[paste0("ols_", tolower(ty), sfx)]] <- grab(
      b, vcovCR(fit, cluster = d$g, type = ty)
    )
  }
  if (wt == "w") {
    # clubSandwich reads lm weights as not inverse-variance by default;
    # inverse_var = TRUE is the reading of analytic (precision) weights.
    out[["ols_cr2_w_inverse_var"]] <- grab(
      b, vcovCR(fit, cluster = d$g, type = "CR2", inverse_var = TRUE)
    )
  }
  # Newey-West, 4 lags, Bartlett kernel, no prewhitening, without and with
  # the N / (N - K) factor (sp.regress hac_small=False / True).
  out[[paste0("ols_hac", sfx)]] <- grab(
    b, NeweyWest(fit, lag = 4, prewhite = FALSE, adjust = FALSE)
  )
  out[[paste0("ols_hac_small", sfx)]] <- grab(
    b, NeweyWest(fit, lag = 4, prewhite = FALSE, adjust = TRUE)
  )
  # multiway clustering, fixest default ssc(): one (N-1)/(N-K) factor and
  # one G_min / (G_min - 1) factor for every term (cluster.df = "min")
  fe <- if (wt == "none") feols(y ~ d + x1 + x2 + x3, data = d) else
    feols(y ~ d + x1 + x2 + x3, data = d, weights = ~w)
  out[[paste0("ols_twoway", sfx)]] <- grab(coef(fe), vcov(fe, cluster = ~ g + yr))
  out[[paste0("ols_threeway", sfx)]] <- grab(coef(fe), vcov(fe, cluster = ~ g + yr + h))
}

# Wild cluster (restricted) bootstrap, Rademacher, 12 clusters: B = 2^12
# enumerates every sign vector. fwildclusterboot reports the symmetric
# two-sided p-value of the cluster-robust t statistic.
fit <- lm(y ~ d + x1 + x2 + x3, data = d)
for (p in c("d", "x1", "x2", "x3")) {
  bt <- boottest(fit, clustid = "g", param = p, B = 4096, type = "rademacher",
                 impose_null = TRUE, p_val_type = "two-tailed",
                 ssc = boot_ssc(adj = TRUE, cluster.adj = TRUE))
  out[[paste0("ols_wild_", p)]] <- list(p_val = bt$p_val, t_stat = bt$t_stat,
                                         B = bt$boot_iter, G = bt$N_G)
}

# --------------------------------------------------------------- 2SLS ----
forms <- list(
  over = y ~ d + x1 + x2 + x3 | z1 + z2 + x1 + x2 + x3,
  just = y ~ d + x1 + x2 + x3 | z1 + x1 + x2 + x3
)
fforms <- list(
  over = y ~ x1 + x2 + x3 | d ~ z1 + z2,
  just = y ~ x1 + x2 + x3 | d ~ z1
)
for (nm in names(forms)) {
  fit <- ivreg(forms[[nm]], data = d)
  b <- coef(fit)
  out[[paste0("iv_", nm, "_classical")]] <- grab(b, vcov(fit))
  for (ty in c("HC0", "HC1", "HC2", "HC3")) {
    out[[paste0("iv_", nm, "_", tolower(ty))]] <- grab(b, vcovHC(fit, type = ty))
  }
  out[[paste0("iv_", nm, "_cr1")]] <- grab(b, vcovCL(fit, cluster = ~g, type = "HC1"))
  for (ty in c("CR2", "CR3")) {
    out[[paste0("iv_", nm, "_", tolower(ty))]] <- grab(
      b, vcovCR(fit, cluster = d$g, type = ty)
    )
  }
  fe <- feols(fforms[[nm]], data = d)
  out[[paste0("iv_", nm, "_twoway")]] <- grab(coef(fe), vcov(fe, cluster = ~ g + yr))
  out[[paste0("iv_", nm, "_threeway")]] <- grab(coef(fe), vcov(fe, cluster = ~ g + yr + h))
}

out[["_meta"]] <- list(
  R = R.version.string,
  sandwich = as.character(packageVersion("sandwich")),
  clubSandwich = as.character(packageVersion("clubSandwich")),
  ivreg = as.character(packageVersion("ivreg")),
  fixest = as.character(packageVersion("fixest")),
  fwildclusterboot = as.character(packageVersion("fwildclusterboot")),
  data = "tests/reference_parity/_fixtures/vce_menu_data.csv"
)
write_json(out, "vce_menu_R.json", auto_unbox = TRUE, digits = I(17), pretty = TRUE)
cat("wrote vce_menu_R.json\n")
