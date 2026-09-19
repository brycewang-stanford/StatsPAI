# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_timeseries_R_parity.py.
#
# Reads the CSVs written by _fixtures/_generate_timeseries_data.py and writes
# _fixtures/timeseries_R.json (digits = 17). Run from the repository root:
#
#   Rscript tests/reference_parity/_generate_timeseries_R.R
#
# Packages: urca (ca.jo, ur.df), aTSA (coint.test), vars (VAR, irf,
# causality), strucchange (efp, sctest, boundary, Fstats, breakpoints),
# sandwich (NeweyWest), rugarch (ugarchfit), plm (purtest).
# ---------------------------------------------------------------------------
suppressMessages({
  library(jsonlite)
  library(urca)
  library(aTSA)
  library(vars)
  library(strucchange)
  library(sandwich)
  library(rugarch)
  library(plm)
})

fx <- "tests/reference_parity/_fixtures"
out <- list()
out$versions <- list(
  R = R.version.string,
  urca = as.character(packageVersion("urca")),
  aTSA = as.character(packageVersion("aTSA")),
  vars = as.character(packageVersion("vars")),
  strucchange = as.character(packageVersion("strucchange")),
  sandwich = as.character(packageVersion("sandwich")),
  rugarch = as.character(packageVersion("rugarch")),
  plm = as.character(packageVersion("plm"))
)

# ---- Johansen: urca::ca.jo ------------------------------------------------
co <- read.csv(file.path(fx, "ts_coint.csv"))
jo <- list()
for (ec in c("none", "const", "trend")) for (K in c(2, 3)) {
  tr <- ca.jo(co[, c("y1", "y2", "y3")], ecdet = ec, K = K, type = "trace")
  mx <- ca.jo(co[, c("y1", "y2", "y3")], ecdet = ec, K = K, type = "eigen")
  jo[[paste0(ec, "_K", K)]] <- list(
    lambda = unname(tr@lambda),
    trace = rev(unname(tr@teststat)),   # ordered H0: r <= 0, 1, 2
    maxeig = rev(unname(mx@teststat)),
    V = unname(tr@V),                   # columns normalised to V[1, ] = 1
    cval_trace_5pct = rev(unname(tr@cval[, "5pct"])),
    N = nrow(tr@Z0)
  )
}
out$johansen <- jo

# ---- Engle-Granger: step-1 lm residuals + urca::ur.df(type = "none") and
# ---- aTSA::coint.test (type 1 = no trend in the residual ADF) -------------
eg_case <- function(dep, regs, lags, trend) {
  n <- nrow(co)
  X <- as.matrix(co[, regs, drop = FALSE])
  tt <- 0:(n - 1)
  if (trend == "c") fit <- lm(co[[dep]] ~ X)
  if (trend == "ct") fit <- lm(co[[dep]] ~ X + tt)
  if (trend == "ctt") fit <- lm(co[[dep]] ~ X + tt + I(tt^2))
  e <- unname(residuals(fit))
  adf <- ur.df(e, type = "none", lags = lags)
  list(coef = unname(coef(fit)), stat = unname(adf@teststat[1]), n = n)
}
eg <- list(
  y1_y2_L0 = eg_case("y1", "y2", 0, "c"),
  y1_y2_L2 = eg_case("y1", "y2", 2, "c"),
  y1_y23_L1 = eg_case("y1", c("y2", "y3"), 1, "c"),
  w_y1_L1 = eg_case("w", "y1", 1, "c"),
  y1_y2_L1_ct = eg_case("y1", "y2", 1, "ct"),
  y1_y2_L1_ctt = eg_case("y1", "y2", 1, "ctt")
)
# aTSA: nlag = L + 1 (its embed() keeps nlag - 1 lagged differences)
at <- coint.test(co$y1, co$y2, nlag = 3, output = FALSE)
eg$aTSA_y1_y2_L2_stat <- unname(at[1, "EG"])
out$engle_granger <- eg

# ---- VAR: vars::VAR, irf, causality ---------------------------------------
vd <- read.csv(file.path(fx, "ts_var.csv"))
v3 <- VAR(vd, p = 2, type = "const")
ir_o <- irf(v3, impulse = "infl", response = c("gdp", "infl", "rate"),
            n.ahead = 8, ortho = TRUE, boot = FALSE)
ir_n <- irf(v3, impulse = "infl", response = c("gdp", "infl", "rate"),
            n.ahead = 8, ortho = FALSE, boot = FALSE)
ir_c <- irf(v3, impulse = "infl", response = c("gdp", "infl", "rate"),
            n.ahead = 8, ortho = TRUE, cumulative = TRUE, boot = FALSE)
v2 <- VAR(vd[, c("gdp", "infl")], p = 2, type = "const")
cz <- causality(v2, cause = "infl")$Granger
out$var <- list(
  irf_ortho_infl = lapply(as.data.frame(ir_o$irf$infl), unname),
  irf_simple_infl = lapply(as.data.frame(ir_n$irf$infl), unname),
  irf_ortho_cum_infl = lapply(as.data.frame(ir_c$irf$infl), unname),
  granger_biv_F = unname(cz$statistic),
  granger_biv_df = unname(cz$parameter),
  granger_biv_p = unname(cz$p.value)
)

# ---- strucchange ------------------------------------------------------------
br <- read.csv(file.path(fx, "ts_break.csv"))
cu <- efp(y ~ x, data = br, type = "Rec-CUSUM")
cs <- sctest(cu)
cu1 <- efp(ys ~ 1, data = br, type = "Rec-CUSUM")
cs1 <- sctest(cu1)
fs <- Fstats(y ~ x, data = br, from = 0.15)
fs1 <- Fstats(y ~ 1, data = br, from = 0.15)
bp <- breakpoints(ym ~ 1, data = br, h = 0.15, breaks = 5)
sbp <- summary(bp)
bpx <- breakpoints(y ~ x, data = br, h = 0.15, breaks = 5)
sbx <- summary(bpx)
bp_by_m <- function(b, M) lapply(1:M, function(m) breakpoints(b, breaks = m)$breakpoints)
out$strucchange <- list(
  cusum_process = unname(as.numeric(cu$process)),
  cusum_stat = unname(cs$statistic),
  cusum_p = unname(cs$p.value),
  cusum_bound = c(
    `0.01` = as.numeric(boundary(cu, alpha = 0.01))[1],
    `0.05` = as.numeric(boundary(cu, alpha = 0.05))[1],
    `0.10` = as.numeric(boundary(cu, alpha = 0.10))[1]
  ),
  cusum_stable_stat = unname(cs1$statistic),
  cusum_stable_p = unname(cs1$p.value),
  Fstats_x = unname(as.numeric(fs$Fstats)),
  Fstats_x_from = floor(0.15 * nrow(br)),
  Fstats_x_breakpoint = fs$breakpoint,
  Fstats_1 = unname(as.numeric(fs1$Fstats)),
  Fstats_1_breakpoint = fs1$breakpoint,
  bp_ym_RSS = unname(sbp$RSS["RSS", ]),
  bp_ym_BIC = unname(sbp$RSS["BIC", ]),
  bp_ym_breaks = bp$breakpoints,
  bp_ym_by_m = bp_by_m(bp, 5),
  bp_yx_RSS = unname(sbx$RSS["RSS", ]),
  bp_yx_BIC = unname(sbx$RSS["BIC", ]),
  bp_yx_breaks = bpx$breakpoints,
  bp_yx_by_m = bp_by_m(bpx, 5)
)

# ---- ITS: lm + sandwich::NeweyWest ----------------------------------------
it <- read.csv(file.path(fx, "ts_its.csv"))
n <- nrow(it)
idx <- 0:(n - 1)
it$D <- as.numeric(idx >= 40)
it$tpost <- ifelse(idx >= 40, idx - 40, 0)
m <- lm(y ~ month + D + tpost, data = it)
V0 <- NeweyWest(m, lag = 4, prewhite = FALSE, adjust = FALSE)
V1 <- NeweyWest(m, lag = 4, prewhite = FALSE, adjust = TRUE)
out$its <- list(
  coef = unname(coef(m)),
  se_nw = unname(sqrt(diag(V0))),
  se_nw_adj = unname(sqrt(diag(V1)))
)

# ---- GARCH(1,1): rugarch --------------------------------------------------
gr <- read.csv(file.path(fx, "ts_garch.csv"))$r
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)
gf <- ugarchfit(spec, gr, solver = "hybrid")
out$garch <- list(
  coef = unname(coef(gf)),            # mu, omega, alpha1, beta1
  loglik = likelihood(gf),
  se = unname(sqrt(diag(vcov(gf)))),
  se_robust = unname(sqrt(diag(vcov(gf, robust = TRUE)))),
  sigma2_first = unname(as.numeric(sigma(gf))[1:3]^2)
)

# ---- plm::purtest -----------------------------------------------------------
pn <- read.csv(file.path(fx, "ts_panel.csv"))
pdf <- pdata.frame(pn, index = c("id", "time"))
pu <- list()
for (ex in c("intercept", "trend")) for (dc in c(FALSE, TRUE)) {
  key <- paste0(ex, "_dfcor", as.integer(dc))
  res <- list()
  for (tt in c("levinlin", "ips", "madwu", "invnormal", "logit", "Pm")) {
    r <- purtest(pdf$y, test = tt, exo = ex, lags = 1, dfcor = dc,
                 p.approx = "MacKinnon1994")
    res[[tt]] <- c(stat = unname(r$statistic$statistic),
                   p = unname(r$statistic$p.value))
  }
  for (hc in c(TRUE, FALSE)) {
    r <- purtest(pdf$y, test = "hadri", exo = ex, dfcor = dc, Hcons = hc)
    res[[paste0("hadri_H", as.integer(hc))]] <-
      c(stat = unname(r$statistic$statistic), p = unname(r$statistic$p.value))
  }
  r <- purtest(pdf$y, test = "ips", exo = ex, lags = 1, dfcor = dc)
  res$ips_trho <- unname(sapply(r$idres, function(z) z$trho))
  pu[[key]] <- res
}
r <- purtest(pdf$y, test = "levinlin", exo = "none", lags = 1, dfcor = FALSE)
pu$none_llc <- c(stat = unname(r$statistic$statistic), p = unname(r$statistic$p.value))
r <- purtest(pdf$yrw, test = "ips", exo = "intercept", lags = 1, dfcor = FALSE)
pu$rw_ips <- c(stat = unname(r$statistic$statistic), p = unname(r$statistic$p.value))
ns <- asNamespace("plm")
llc_tab <- get("adj.levinlin", ns)
ips_tab <- get("adj.ips.wtbar", ns)
pu$tables <- list(
  llc_T = as.numeric(dimnames(llc_tab)[[1]]),
  llc_none = unname(llc_tab[, , "none"]),
  llc_intercept = unname(llc_tab[, , "intercept"]),
  llc_trend = unname(llc_tab[, , "trend"]),
  ips_T = as.numeric(dimnames(ips_tab)[[2]]),
  ips_mean_intercept = unname(ips_tab[, , "mean", "intercept"]),
  ips_var_intercept = unname(ips_tab[, , "var", "intercept"]),
  ips_mean_trend = unname(ips_tab[, , "mean", "trend"]),
  ips_var_trend = unname(ips_tab[, , "var", "trend"])
)
tgrid <- c(-6, -4.2, -3.1, -2.6, -1.9, -1.2, -0.5, 0.4, 1.5)
pu$padf1994 <- list(
  t = tgrid,
  none = sapply(tgrid, function(x) as.numeric(get("padf", ns)(x, exo = "none", p.approx = "MacKinnon1994"))),
  intercept = sapply(tgrid, function(x) as.numeric(get("padf", ns)(x, exo = "intercept", p.approx = "MacKinnon1994"))),
  trend = sapply(tgrid, function(x) as.numeric(get("padf", ns)(x, exo = "trend", p.approx = "MacKinnon1994")))
)
out$purtest <- pu

writeLines(
  toJSON(out, digits = 17, auto_unbox = TRUE, pretty = TRUE, na = "null"),
  file.path(fx, "timeseries_R.json")
)
cat("wrote", file.path(fx, "timeseries_R.json"), "\n")
