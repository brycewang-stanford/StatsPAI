#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_mr_R_parity.py
#
# Requires: R 4.5 + MendelianRandomization + TwoSampleMR + RadialMR +
# MRPRESSO + mr.raps (archived from CRAN; GitHub qingyuanzhao/mr.raps).
# Run from any directory; writes mr_ldl.csv and mr_R.json next to itself.
#
# Data: the 28-variant LDL-cholesterol -> coronary heart disease summary
# statistics bundled with MendelianRandomization (ldlc, chdlodds, hdlc,
# trig and their standard errors), exported byte-for-byte so both sides
# read the same numbers.
#
# Conventions this fixture pins, each of which decides a number below
# -------------------------------------------------------------------
# * mr_ivw(model = "default") is the random-effects model for more than
#   three variants: SE = fixed-effect SE * max(1, residual standard error).
# * mr_egger orients every variant so the exposure association is positive
#   before regressing; the intercept's sign depends on it.
# * mr_cML needs the GWAS sample size for its BIC; n = 17723 (the smaller
#   of the two samples) is passed explicitly, and per-K fits use K_vec = K.
# * mr.raps.overdispersed.robust is run with the package's own tuning
#   constants (Huber 1.345, Tukey 4.685). Its tau^2 root (uniroot, default
#   tol) and Gaussian moments (integrate, rel.tol 1.2e-4) are loose, so the
#   moments are emitted too: the test checks the sandwich formula at R's
#   own (beta, tau^2, moments) and grades the fitted values separately.
# * ivw_radial is run with alpha = 0.05 and no Bonferroni adjustment -- the
#   RadialMR convention; StatsPAI defaults to the Bonferroni threshold.
# * TwoSampleMR::mr_steiger takes r (not beta / se); r is built with
#   get_r_from_bsen from the same summary statistics and sample sizes
#   17723 (exposure) and 60801 (outcome).
# * MR-PRESSO's p-values are Monte Carlo; only its deterministic output
#   (raw estimate, RSS, outlier-corrected fit given the outlier set) and the
#   outlier set itself -- two variants with p = 0 at B = 2000 -- are pinned.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(MendelianRandomization); library(RadialMR)
  library(MRPRESSO); library(mr.raps); library(jsonlite)
})
# TwoSampleMR is used through :: only -- attaching it masks
# MendelianRandomization's mr_ivw / mr_egger / mr_median.
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

d <- data.frame(bx = ldlc, bxse = ldlcse, by = chdlodds, byse = chdloddsse,
                hdl = hdlc, hdlse = hdlcse, tg = trig, tgse = trigse)
write.csv(d, file.path(OUT, "mr_ldl.csv"), row.names = FALSE)
o <- mr_input(bx = ldlc, bxse = ldlcse, by = chdlodds, byse = chdloddsse)
out <- list()

# ---- IVW / Egger / median / mode (MendelianRandomization) -----------------
iv_d <- mr_ivw(o); iv_f <- mr_ivw(o, model = "fixed"); iv_r <- mr_ivw(o, model = "random")
out$ivw_default <- list(est = iv_d@Estimate, se = iv_d@StdError, model = iv_d@Model,
                        rse = iv_d@RSE, Q = unname(iv_d@Heter.Stat[1]),
                        Fstat = iv_d@Fstat)
out$ivw_fixed  <- list(est = iv_f@Estimate, se = iv_f@StdError)
out$ivw_random <- list(est = iv_r@Estimate, se = iv_r@StdError)
eg <- mr_egger(o)
out$egger <- list(est = eg@Estimate, se = eg@StdError.Est, int = eg@Intercept,
                  int_se = eg@StdError.Int, Q = unname(eg@Heter.Stat[1]),
                  rse = eg@RSE)
out$median_weighted  <- mr_median(o, weighting = "weighted")@Estimate
out$median_simple    <- mr_median(o, weighting = "simple")@Estimate
out$median_penalized <- mr_median(o, weighting = "penalized")@Estimate
out$mode_weighted   <- mr_mbe(o, weighting = "weighted", stderror = "simple")@Estimate
out$mode_unweighted <- mr_mbe(o, weighting = "unweighted", stderror = "simple")@Estimate

# ---- leave-one-out: mr_ivw(default) on each 27-variant subset -------------
loo <- sapply(seq_along(ldlc), function(i) {
  r <- mr_ivw(mr_input(bx = ldlc[-i], bxse = ldlcse[-i], by = chdlodds[-i], byse = chdloddsse[-i]))
  c(r@Estimate, r@StdError)
})
out$loo <- list(est = loo[1, ], se = loo[2, ])

# ---- multivariable IVW ----------------------------------------------------
mvi <- mr_mvinput(bx = cbind(ldlc, hdlc, trig), bxse = cbind(ldlcse, hdlcse, trigse),
                  by = chdlodds, byse = chdloddsse)
mv <- mr_mvivw(mvi); mvf <- mr_mvivw(mvi, model = "fixed")
out$mvivw <- list(est = as.numeric(mv@Estimate), se = as.numeric(mv@StdError),
                  model = mv@Model, se_fixed = as.numeric(mvf@StdError))

# ---- constrained maximum likelihood ---------------------------------------
out$cml_per_k <- lapply(0:6, function(k) {
  r <- mr_cML(o, MA = FALSE, DP = FALSE, K_vec = k, n = 17723)
  list(K = k, est = r@Estimate, se = r@StdError, invalid = as.integer(r@BIC_invalid))
})
sel <- mr_cML(o, MA = FALSE, DP = FALSE, n = 17723)
ma  <- mr_cML(o, MA = TRUE, DP = FALSE, n = 17723)
out$cml_bic <- list(est = sel@Estimate, se = sel@StdError, invalid = as.integer(sel@BIC_invalid))
out$cml_ma  <- list(est = ma@Estimate, se = ma@StdError)

# ---- robust adjusted profile score ----------------------------------------
g <- function(r) list(est = r$beta.hat, se = r$beta.se,
                      tau2 = if (is.null(r$tau2.hat)) NA else r$tau2.hat,
                      tau2_se = if (is.null(r$tau2.se)) NA else r$tau2.se)
out$raps_simple <- g(mr.raps.simple(ldlc, chdlodds, ldlcse, chdloddsse))
out$raps_l2 <- g(mr.raps.overdispersed(ldlc, chdlodds, ldlcse, chdloddsse))
out$raps_huber <- g(suppressWarnings(mr.raps.overdispersed.robust(
  ldlc, chdlodds, ldlcse, chdloddsse, loss.function = "huber", k = 1.345)))
out$raps_tukey <- g(suppressWarnings(mr.raps.overdispersed.robust(
  ldlc, chdlodds, ldlcse, chdloddsse, loss.function = "tukey", k = 4.685)))
# The loss's Gaussian moments exactly as mr.raps.overdispersed.robust
# computes them (integrate(), default rel.tol), so the sandwich formula can be
# checked at R's own estimates independently of solver and quadrature error.
moments <- function(rho) {
  I <- function(f) integrate(function(x) f(x) * dnorm(x), -Inf, Inf)$value
  delta <- I(function(x) x * rho(x, deriv = 1))
  c(delta = delta, c1 = I(function(x) rho(x, deriv = 1)^2),
    c2 = I(function(x) x^2 * rho(x, deriv = 1)^2) - delta^2,
    c3 = I(function(x) x^2 * rho(x, deriv = 2)))
}
out$raps_huber$consts <- as.list(moments(function(r, ...) mr.raps:::rho.huber(r, 1.345, ...)))
out$raps_tukey$consts <- as.list(moments(function(r, ...) mr.raps:::rho.tukey(r, 4.685, ...)))

# ---- radial IVW -----------------------------------------------------------
rd <- format_radial(ldlc, chdlodds, ldlcse, chdloddsse, seq_along(ldlc))
ri <- ivw_radial(rd, alpha = 0.05, weights = 1, tol = 0.0001, summary = FALSE)
out$radial <- list(Wj = ri$data$Wj, Qj = ri$data$Qj,
                   outliers = as.integer(if (is.data.frame(ri$outliers)) ri$outliers$SNP else ri$outliers),
                   Q = as.numeric(ri$qstatistic))

# ---- Egger intercept test / heterogeneity (TwoSampleMR) --------------------
te <- TwoSampleMR::mr_egger_regression(ldlc, chdlodds, ldlcse, chdloddsse)
ti <- TwoSampleMR::mr_ivw(ldlc, chdlodds, ldlcse, chdloddsse)
out$tsmr_egger <- list(b = te$b, se = te$se, b_i = te$b_i, se_i = te$se_i,
                       pval_i = te$pval_i, Q = te$Q, Q_df = te$Q_df, Q_pval = te$Q_pval)
out$tsmr_ivw <- list(b = ti$b, se = ti$se, Q = ti$Q, Q_df = ti$Q_df, Q_pval = ti$Q_pval)

# ---- Steiger directionality (TwoSampleMR) ---------------------------------
rx <- TwoSampleMR::get_r_from_bsen(ldlc, ldlcse, 17723)
ry <- TwoSampleMR::get_r_from_bsen(chdlodds, chdloddsse, 60801)
st <- function(re, ro) {
  s <- TwoSampleMR::mr_steiger(p_exp = NA, p_out = NA, n_exp = rep(17723, 28),
                               n_out = rep(60801, 28), r_exp = re, r_out = ro)
  list(r2_exp = s$r2_exp, r2_out = s$r2_out, dir = s$correct_causal_direction, p = s$steiger_test)
}
out$steiger_fwd <- st(rx, ry)

# ---- MR-PRESSO ------------------------------------------------------------
set.seed(1)
pr <- mr_presso(BetaOutcome = "by", BetaExposure = "bx", SdOutcome = "byse", SdExposure = "bxse",
                OUTLIERtest = TRUE, DISTORTIONtest = TRUE, data = d,
                NbDistribution = 2000, SignifThreshold = 0.05)
m <- pr$`Main MR results`
gt <- pr$`MR-PRESSO results`$`Global Test`
ot <- pr$`MR-PRESSO results`$`Outlier Test`
out$presso <- list(raw_est = m$`Causal Estimate`[1], raw_se = m$Sd[1],
                   cor_est = m$`Causal Estimate`[2], cor_se = m$Sd[2],
                   RSSobs = as.numeric(gt$RSSobs), outliers = which(ot$Pvalue < 0.05))

out$provenance <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  MendelianRandomization = as.character(packageVersion("MendelianRandomization")),
  TwoSampleMR = as.character(packageVersion("TwoSampleMR")),
  RadialMR = as.character(packageVersion("RadialMR")),
  MRPRESSO = as.character(packageVersion("MRPRESSO")),
  mr.raps = as.character(packageVersion("mr.raps")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null"),
           file.path(OUT, "mr_R.json"))
cat("wrote mr_R.json\n")
