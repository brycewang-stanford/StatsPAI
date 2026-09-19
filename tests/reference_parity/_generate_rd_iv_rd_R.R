#!/usr/bin/env Rscript
# Ground truth for tests/reference_parity/test_rd_iv_rd_R_parity.py (RD block
# of the rd_iv parity family).
#
# Writes, under tests/reference_parity/_fixtures/:
#   rd_iv_kink.csv     seeded synthetic kink design: y, x, t (fuzzy
#                      treatment with a slope change at 0), g (50 clusters)
#   rd_iv_density.csv  seeded running variable with bunching just above 0,
#                      rounded to 3 decimals so it has mass points
#   rd_iv_rd_R.json    reference values at 17 significant digits
# and reads the existing rdsenate.csv (rdrobust's Senate data).
#
# Re-run only when the contract changes:
#   Rscript tests/reference_parity/_generate_rd_iv_rd_R.R
#
# References exercised
#   rdrobust  rdrobust(deriv = 1)  -- the kink estimator behind sp.rkd and
#                                     sp.rdrobust(deriv=1)
#             rdplot()             -- bins, bin means, global polynomial
#   rddensity rddensity() + rdplotdensity()  -- lpdensity on each side
#   rdd       DCdensity()          -- McCrary's density test (CRAN archive
#                                     0.57; rdd is no longer on CRAN)
#   rdhte     rdhte(), rdbwhte(), rdhte_lincom()  -- heterogeneous RD effects
#                                     (also writes rd_iv_hte.csv)
#   RDHonest  RDHonest(y | d ~ x)  -- honest fuzzy RD (writes rd_iv_frd.csv)
#   rdlocrand rdsensitivity(), rdrbounds(), rdrandinf()  -- randomization
#                                     p-values (writes rd_iv_locrand.csv);
#                                     Monte-Carlo, compared at MC error

suppressMessages({
  library(rdrobust)
  library(rddensity)
  library(lpdensity)
  library(rdd)
  library(rdhte)
  library(RDHonest)
  library(rdlocrand)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("--file=", "", grep("--file=", args, value = TRUE)))
FIX <- file.path(here, "_fixtures")
if (!dir.exists(FIX)) FIX <- "tests/reference_parity/_fixtures"

# ---- data ------------------------------------------------------------------
set.seed(7)
n <- 2000
x <- runif(n, -1, 1)
g <- sample(1:50, n, replace = TRUE)
t <- 0.5 + x + 1.5 * pmax(x, 0) + 0.3 * x^2 + rnorm(n, 0, 0.3)
y <- 1 + 0.4 * t - 0.2 * x^2 + rnorm(n, 0, 0.5) + 0.2 * rnorm(50)[g]
write.csv(data.frame(y, x, t, g), file.path(FIX, "rd_iv_kink.csv"),
          row.names = FALSE)
kink <- read.csv(file.path(FIX, "rd_iv_kink.csv"))

set.seed(3)
xd <- round(c(rnorm(700, 0.1, 1), runif(150, 0, 0.3)), 3)
write.csv(data.frame(x = xd), file.path(FIX, "rd_iv_density.csv"),
          row.names = FALSE)
dens <- read.csv(file.path(FIX, "rd_iv_density.csv"))$x

senate <- read.csv(file.path(FIX, "rdsenate.csv"))

# ---- rdrobust(deriv = 1) ---------------------------------------------------
rd_out <- function(r) list(
  coef = unname(as.numeric(r$coef)),        # conventional, bias-corr., robust
  se = unname(as.numeric(r$se)),
  h = unname(as.numeric(r$bws[1, ])),
  b = unname(as.numeric(r$bws[2, ])),
  N_h = unname(as.numeric(r$N_h))
)
kink_specs <- list(
  sharp_p1_nn   = list(fuzzy = FALSE, p = 1, vce = "nn"),
  sharp_pdef_nn = list(fuzzy = FALSE, p = NULL, vce = "nn"),
  sharp_p1_hc1  = list(fuzzy = FALSE, p = 1, vce = "hc1"),
  sharp_p1_hc1_h04 = list(fuzzy = FALSE, p = 1, vce = "hc1", h = 0.4),
  sharp_p1_uni_nn  = list(fuzzy = FALSE, p = 1, vce = "nn", kernel = "uni"),
  sharp_p1_hc1_cl  = list(fuzzy = FALSE, p = 1, vce = "hc1", cluster = TRUE),
  fuzzy_p1_nn   = list(fuzzy = TRUE, p = 1, vce = "nn"),
  fuzzy_pdef_nn = list(fuzzy = TRUE, p = NULL, vce = "nn"),
  fuzzy_p1_hc1  = list(fuzzy = TRUE, p = 1, vce = "hc1"),
  fuzzy_p1_hc1_h04 = list(fuzzy = TRUE, p = 1, vce = "hc1", h = 0.4)
)
kink_out <- list()
for (nm in names(kink_specs)) {
  s <- kink_specs[[nm]]
  a <- list(y = kink$y, x = kink$x, deriv = 1, vce = s$vce)
  if (!is.null(s$p)) a$p <- s$p
  if (isTRUE(s$fuzzy)) a$fuzzy <- kink$t
  if (!is.null(s$h)) a$h <- s$h
  if (!is.null(s$kernel)) a$kernel <- s$kernel
  if (isTRUE(s$cluster)) a$cluster <- kink$g
  kink_out[[nm]] <- rd_out(do.call(rdrobust, a))
}

# ---- rdplot ------------------------------------------------------------------
plot_out <- function(r) list(
  J = unname(r$J), J_IMSE = unname(r$J_IMSE), J_MV = unname(r$J_MV),
  coef = unname(as.numeric(r$coef)),
  mean_bin = r$vars_bins$rdplot_mean_bin,
  mean_x = r$vars_bins$rdplot_mean_x,
  mean_y = r$vars_bins$rdplot_mean_y,
  se_y = r$vars_bins$rdplot_se_y,
  N = r$vars_bins$rdplot_N,
  ci_l = r$vars_bins$rdplot_ci_l,
  ci_r = r$vars_bins$rdplot_ci_r,
  # every 37th point of the 2 x 500 polynomial grid (R indices 1, 38, ...)
  poly_idx = seq(1, 1000, by = 37),
  poly_y = r$vars_poly$rdplot_y[seq(1, 1000, by = 37)]
)
rdplot_out <- list()
for (b in c("es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr")) {
  rdplot_out[[paste0("kink_", b)]] <-
    plot_out(rdplot(kink$y, kink$x, binselect = b, hide = TRUE))
}
rdplot_out$kink_tri_p2_h05_nbins <- plot_out(
  rdplot(kink$y, kink$x, kernel = "tri", p = 2, h = 0.5, nbins = c(7, 9),
         hide = TRUE))
rdplot_out$kink_covs <- plot_out(suppressMessages(
  rdplot(kink$y, kink$x, covs = kink$t, hide = TRUE)))
rdplot_out$senate_default <- plot_out(
  rdplot(senate$vote, senate$margin, hide = TRUE))

# ---- rdplotdensity ---------------------------------------------------------------
dens_out <- function(X, c = 0) {
  r <- rddensity(X, c = c)
  pd <- rdplotdensity(r, X, noPlot = TRUE)
  cols <- c("grid", "bw", "nh", "f_p", "f_q", "se_p", "se_q")
  list(h = c(r$h$left, r$h$right),
       left = lapply(cols, function(k) unname(pd$Estl$Estimate[, k])),
       right = lapply(cols, function(k) unname(pd$Estr$Estimate[, k])),
       cols = cols)
}
rdplotdensity_out <- list(
  density = dens_out(dens),
  senate = dens_out(senate$margin)
)

# ---- DCdensity -----------------------------------------------------------------
mc <- function(...) {
  r <- DCdensity(..., plot = FALSE, ext.out = TRUE)
  list(theta = r$theta, se = r$se, z = r$z, p = r$p, bin = r$binsize, bw = r$bw)
}
mccrary_out <- list(
  density_c0 = mc(dens, 0),
  density_c05_fixed = mc(dens, 0.5, bw = 0.4, bin = 0.05),
  senate_c0 = mc(senate$margin, 0)
)

# ---- rdhte ------------------------------------------------------------------
set.seed(11)
nh <- 2000
xh <- runif(nh, -1, 1)
zh <- rnorm(nh)
yh <- 0.5 * xh + (2 + 1.5 * zh) * (xh >= 0) + 0.3 * zh + rnorm(nh, 0, 0.5)
write.csv(data.frame(y = yh, x = xh, z = zh, zb = as.numeric(zh > 0.3),
                     cl = rep(1:100, 20)),
          file.path(FIX, "rd_iv_hte.csv"), row.names = FALSE)
hd <- read.csv(file.path(FIX, "rd_iv_hte.csv"))
y <- hd$y; x <- hd$x; z <- hd$z; zb <- hd$zb; cl <- hd$cl
hte_out <- function(r) list(
  names = names(r$coef),
  coef = unname(r$coef), coef_bc = unname(r$coef.bc),
  se_rb = unname(r$se.rb), vcov = unname(as.numeric(r$vcov)),
  h = unname(as.numeric(t(r$h))), Nh = unname(as.numeric(t(r$Nh))))
lincom_out <- function(l) list(
  estimate = l$individual$estimate, z = l$individual$z_stat,
  p = l$individual$p_value, ci = c(l$individual$conf.low, l$individual$conf.high),
  joint = l$joint$statistic)
r_sub <- rdhte(y = y, x = x, covs.hte = zb)
r_cont <- rdhte(y = y, x = x, covs.hte = z)
rdhte_out <- list(
  cont_h04_hc1 = hte_out(rdhte(y = y, x = x, covs.hte = z, h = 0.4, vce = "hc1")),
  cont_h04_hc3 = hte_out(rdhte(y = y, x = x, covs.hte = z, h = 0.4)),
  cont_h04_hc0 = hte_out(rdhte(y = y, x = x, covs.hte = z, h = 0.4, vce = "hc0")),
  cont_h04_hc2 = hte_out(rdhte(y = y, x = x, covs.hte = z, h = 0.4, vce = "hc2")),
  cont_default = hte_out(r_cont),
  cont_cluster = hte_out(rdhte(y = y, x = x, covs.hte = z, cluster = cl)),
  cont_p2_h05 = hte_out(rdhte(y = y, x = x, covs.hte = z, p = 2, h = 0.5)),
  cont_epa_h04 = hte_out(rdhte(y = y, x = x, covs.hte = z, kernel = "epa", h = 0.4)),
  cont_uni_h04 = hte_out(rdhte(y = y, x = x, covs.hte = z, kernel = "uni", h = 0.4)),
  sub_default = hte_out(r_sub),
  sub_h04 = hte_out(rdhte(y = y, x = x, covs.hte = zb, h = 0.4)),
  rdbwhte_cont = unname(as.numeric(t(rdbwhte(y = y, x = x, covs.hte = z)$h))),
  rdbwhte_sub = unname(as.numeric(t(rdbwhte(y = y, x = x, covs.hte = zb)$h))),
  lincom_sub = lincom_out(suppressWarnings(
    rdhte_lincom(r_sub, linfct = c("`zb1` - `zb0` = 0"), digits = 15))),
  lincom_cont = lincom_out(suppressWarnings(
    rdhte_lincom(r_cont, linfct = c("T + `T:z` = 0"), digits = 15)))
)

# ---- RDHonest, fuzzy ----------------------------------------------------------
set.seed(21)
nf <- 1500
xf <- runif(nf, -1, 1)
df_ <- as.numeric(runif(nf) < 0.2 + 0.5 * (xf >= 0) + 0.2 * xf)
yf <- 1 + 2 * df_ + 0.6 * xf - 0.8 * xf^2 + rnorm(nf, 0, 0.7)
write.csv(data.frame(y = yf, d = df_, x = xf, g = rep(1:75, 20)),
          file.path(FIX, "rd_iv_frd.csv"), row.names = FALSE)
frd <- read.csv(file.path(FIX, "rd_iv_frd.csv"))
cols_h <- c("estimate", "std.error", "maximum.bias", "conf.low", "conf.high",
            "bandwidth", "M.rf", "M.fs", "first.stage", "p.value")
hon <- function(...) {
  co <- RDHonest(y | d ~ x, data = frd, ...)$coefficients
  as.list(setNames(unname(unlist(co[1, cols_h])), cols_h))
}
rdhonest_out <- list(
  fixed = hon(M = c(2, 0.5), h = 0.4),
  fixed_uniform = hon(M = c(2, 0.5), h = 0.4, kern = "uniform"),
  default = suppressMessages(hon()),
  cluster = {
    # clusterid is evaluated in `data` by RDHonest's NSE; call it directly.
    co <- RDHonest(y | d ~ x, data = frd, M = c(2, 0.5), h = 0.4,
                   clusterid = g, se.method = "EHW")$coefficients
    as.list(setNames(unname(unlist(co[1, cols_h])), cols_h))
  }
)

# ---- rdlocrand: sensitivity and Rosenbaum bounds ------------------------------
set.seed(9)
nl <- 400
xl <- runif(nl, -1, 1)
yl <- 0.35 * (xl >= 0) + 0.3 * xl + rnorm(nl, 0, 0.5)
write.csv(data.frame(y = yl, x = xl), file.path(FIX, "rd_iv_locrand.csv"),
          row.names = FALSE)
lr <- read.csv(file.path(FIX, "rd_iv_locrand.csv"))
wl_ <- c(0.1, 0.15, 0.25, 0.4)
sens <- rdsensitivity(lr$y, lr$x, wlist = wl_, tlist = 0, reps = 4000,
                      nodraw = TRUE, quietly = TRUE)
rb <- rdrbounds(lr$y, lr$x, wlist = 0.15, expgamma = c(1.2, 1.5, 2),
                reps = 4000, bound = "both")
locrand_out <- list(
  reps = 4000,
  sens_windows = wl_,
  sens_pvalues = as.numeric(sens$results),
  obs_stat = sapply(wl_, function(w)
    rdrandinf(lr$y, lr$x, wl = -w, wr = w, reps = 10, quietly = TRUE)$obs.stat),
  rbounds_gamma = c(1.2, 1.5, 2),
  rbounds_upper = as.numeric(rb$upper.bound),
  rbounds_lower = as.numeric(rb$lower.bound)
)

ref <- list(
  meta = list(
    R_version = R.version.string,
    rdrobust_version = as.character(packageVersion("rdrobust")),
    rddensity_version = as.character(packageVersion("rddensity")),
    lpdensity_version = as.character(packageVersion("lpdensity")),
    rdd_version = as.character(packageVersion("rdd")),
    rdhte_version = as.character(packageVersion("rdhte")),
    sandwich_version = as.character(packageVersion("sandwich")),
    RDHonest_version = as.character(packageVersion("RDHonest")),
    rdlocrand_version = as.character(packageVersion("rdlocrand")),
    generated = "deterministic; re-run only on contract change"
  ),
  kink = kink_out,
  rdplot = rdplot_out,
  rdplotdensity = rdplotdensity_out,
  mccrary = mccrary_out,
  rdhte = rdhte_out,
  rdhonest_fuzzy = rdhonest_out,
  locrand = locrand_out
)
writeLines(toJSON(ref, digits = I(17), auto_unbox = TRUE, pretty = FALSE,
                  null = "null", na = "null"),
           file.path(FIX, "rd_iv_rd_R.json"))
cat("wrote rd_iv_kink.csv, rd_iv_density.csv, rd_iv_hte.csv, rd_iv_frd.csv,\n  rd_iv_locrand.csv, rd_iv_rd_R.json\n")
