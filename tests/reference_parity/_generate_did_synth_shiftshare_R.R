#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_did_synth_shiftshare_parity.py
#
# Requires: R 4.5 + ShiftShareSE (CRAN) + AER + sandwich + data.table +
# fixest + jsonlite + ssaggregate (GitHub kylebutts/ssaggregate) +
# bartik.weight (GitHub paulgp/bartik-weight, subdir R-code/pkg; its
# src/Makevars had to drop `CXX_STD = CXX11` and `$(FLIBS)` to build on
# R 4.5 / arm64 -- build flags only, the C++ is untouched).
# Run _fixtures/_generate_did_synth_shiftshare_data.py first, then this file
# from any directory. Writes _fixtures/did_synth_shiftshare_R.json.
#
# Conventions this fixture pins, each of which decides a number below
# -------------------------------------------------------------------
# * ShiftShareSE::ivreg_ss / reg_ss: X is the shift-share variable itself
#   (column B = shares %*% g), W the N x K share matrix, controls Z always
#   include the intercept. AKM: cR_k = hX_k * sum_i W_ik e_i with hX the
#   coefficients of the control-residualised X on W; se = sqrt(sum cR^2)/RX.
#   AKM0 inverts the null-imposed test: CI = mid +/- sqrt(dis), reported
#   se = sqrt(dis)/qnorm(1 - alpha/2); its p-value uses se0 (null-imposed).
#   All p-values / CIs are normal.
# * ivreg_ss's "Homoscedastic" row uses rss / N (no df correction) and its
#   "EHW" row has no small-sample factor; its "Reg. cluster" row has none
#   either. reg_ss's (OLS) EHW multiplies by n/(n-p), its homoscedastic row
#   divides by df.residual, and its cluster row by
#   G/(G-1) * (n-1)/(n-p). These asymmetries are the reference's own.
# * The shares here do NOT sum to one (incomplete shares) and the intercept
#   is not in the span of W.
# * ssaggregate (BHJ): residualise y, x on the controls (fixest OLS), then
#   s_n = sum_l s_ln / sum_{l,n} s_ln and ybar_n = sum_l s_ln y_l / sum_l s_ln.
#   The shock-level IV is AER::ivreg(y ~ x | g, weights = s_n) with an
#   intercept and HC0 (sandwich) SEs, the analogue of BHJ's recommended
#   `ivreg2 ..., robust`.
# * bartik.weight::bw: alpha_k = g_k Z_k' M_W x / sum_k g_k Z_k' M_W x and
#   beta_k = Z_k' M_W y / Z_k' M_W x, M_W annihilating [controls, 1].
# * 2SLS: AER::ivreg; "classical" vcov uses sigma^2 = RSS/(n-k); HC0/HC1 via
#   sandwich::vcovHC.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(ShiftShareSE); library(AER); library(sandwich)
  library(data.table); library(jsonlite); library(ssaggregate)
  library(bartik.weight)
})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
HERE <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
FIX <- file.path(HERE, "_fixtures")

loc <- read.csv(file.path(FIX, "did_synth_shiftshare_loc.csv"))
shk <- read.csv(file.path(FIX, "did_synth_shiftshare_shocks.csv"))
K <- nrow(shk)
shcols <- paste0("sh", seq_len(K))
W <- as.matrix(loc[, shcols])

ss_out <- function(r) {
  list(beta = r$beta, se = as.list(r$se), p = as.list(r$p),
       ci_l = as.list(r$ci.l), ci_r = as.list(r$ci.r))
}

out <- list()

# ---- AKM (ShiftShareSE) ----------------------------------------------------
out$akm_iv_ctrl <- ss_out(ivreg_ss(y ~ c1 + ssum | x, X = B, data = loc, W = W,
                                   method = "all", region_cvar = state))
out$akm_iv_noctrl <- ss_out(ivreg_ss(y ~ 1 | x, X = B, data = loc, W = W,
                                     method = "all", region_cvar = state))
out$akm_ols_ctrl <- ss_out(reg_ss(y ~ c1 + ssum, X = B, data = loc, W = W,
                                  method = "all", region_cvar = state))
out$akm_ols_noctrl <- ss_out(reg_ss(y ~ 1, X = B, data = loc, W = W,
                                    method = "all", region_cvar = state))
# alpha = 0.10 and beta0 != 0 exercise the AKM0 inversion's inputs.
out$akm_iv_ctrl_a10_b0 <- ss_out(ivreg_ss(y ~ c1 + ssum | x, X = B, data = loc,
                                          W = W, method = c("akm", "akm0"),
                                          alpha = 0.10, beta0 = 1.5))

# ---- 2SLS (sp.bartik) ------------------------------------------------------
iv2 <- function(f) {
  m <- AER::ivreg(f, data = loc)
  list(coef = as.list(coef(m)),
       se_classical = as.list(sqrt(diag(vcov(m)))),
       se_hc0 = as.list(sqrt(diag(sandwich::vcovHC(m, type = "HC0")))),
       se_hc1 = as.list(sqrt(diag(sandwich::vcovHC(m, type = "HC1")))))
}
out$tsls_ctrl <- iv2(y ~ c1 + ssum + x | c1 + ssum + B)
out$tsls_noctrl <- iv2(y ~ x | B)

# ---- Rotemberg weights (bartik.weight) -------------------------------------
glob <- data.frame(n = shk$n, g = shk$g)
rw <- function(ctrl) {
  r <- bartik.weight::bw(loc, "y", "x", ctrl, NULL, loc, shcols, glob, "g")
  list(n = r$n, alpha = r$alpha, beta = r$beta)
}
out$rotemberg_ctrl <- rw(c("c1", "ssum"))
out$rotemberg_noctrl <- rw(NULL)

# ---- BHJ shock-level aggregation (ssaggregate) -----------------------------
bhj <- function(ctrl) {
  d <- data.table::copy(as.data.table(loc))  # ssaggregate mutates by reference
  agg <- suppressWarnings(ssaggregate(d, vars = ~ y + x, n = "n", s = "sh",
                                      controls = ctrl))
  agg <- as.data.frame(agg)
  agg$n <- as.integer(agg$n)
  agg <- merge(agg, glob, by = "n")
  agg <- agg[order(agg$n), ]
  m <- AER::ivreg(y ~ x | g, weights = s_n, data = agg)
  list(n = agg$n, s_n = agg$s_n, y = agg$y, x = agg$x,
       beta = unname(coef(m)["x"]),
       se_hc0 = unname(sqrt(diag(sandwich::vcovHC(m, type = "HC0")))["x"]))
}
out$bhj_ctrl <- bhj(~ c1 + ssum)
out$bhj_noctrl <- bhj(~ 1)
# The incomplete-shares warning is part of the reference's contract.
out$bhj_noctrl_warns <- {
  w <- FALSE
  withCallingHandlers(
    ssaggregate(data.table::copy(as.data.table(loc)), vars = ~ y + x,
                n = "n", s = "sh", controls = ~ 1),
    warning = function(cond) { w <<- TRUE; invokeRestart("muffleWarning") })
  w
}

pk <- function(p) as.character(utils::packageVersion(p))
out$versions <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  ShiftShareSE = pk("ShiftShareSE"), AER = pk("AER"), sandwich = pk("sandwich"),
  fixest = pk("fixest"), data.table = pk("data.table"),
  ssaggregate = paste0(pk("ssaggregate"), " (GitHub kylebutts/ssaggregate@",
                       utils::packageDescription("ssaggregate")$RemoteSha, ")"),
  bartik.weight = paste0(pk("bartik.weight"),
                         " (GitHub paulgp/bartik-weight@722ceb85484d6a2bf77985edf2403515eacd1770, R-code/pkg)")
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), na = "null", pretty = TRUE),
           file.path(FIX, "did_synth_shiftshare_R.json"))
cat("wrote", file.path(FIX, "did_synth_shiftshare_R.json"), "\n")
