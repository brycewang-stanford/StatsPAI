# Reference values for tests/reference_parity/test_ts_causal_parity.py
#
# lpirfs (GPL) and plm (GPL) are used as black boxes: only their documented
# interfaces and their outputs on the committed synthetic files. svars (MIT)
# gives the historical decomposition. Requires lpirfs, plm, svars, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_ts_causal_R.R
suppressMessages({library(lpirfs); library(plm); library(svars); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "ts_causal_series.csv"))
p = read.csv(file.path(here, "ts_causal_panel.csv"))
pu = read.csv(file.path(here, "ts_causal_panel_unbal.csv"))
v = read.csv(file.path(here, "ts_causal_var.csv"))
H = 6
lin = function(f) list(irf = f$irf_lin_mean[1, ], se = f$irf_lin_up[1, ] - f$irf_lin_mean[1, ])
en = d[, c("y", "g")]
iv = list(
  ols = lin(lp_lin_iv(endog_data = en, shock = d[, "z", drop = FALSE], lags_endog_lin = 2,
                      trend = 0, confint = 1, hor = H)),
  iv = lin(lp_lin_iv(endog_data = en, shock = d[, "g", drop = FALSE], use_twosls = TRUE,
                     instrum = d[, "z", drop = FALSE], lags_endog_lin = 2, trend = 1,
                     confint = 1, hor = H)),
  iv_contemp = lin(lp_lin_iv(endog_data = en, shock = d[, "g", drop = FALSE], use_twosls = TRUE,
                             instrum = d[, "z", drop = FALSE], lags_endog_lin = 1, trend = 0,
                             confint = 1, hor = H, contemp_data = d[, "w", drop = FALSE], nw_lag = 3)),
  iv_olsse = lin(lp_lin_iv(endog_data = en, shock = d[, "g", drop = FALSE], use_twosls = TRUE,
                           instrum = d[, "z", drop = FALSE], lags_endog_lin = 2, trend = 0,
                           confint = 1, hor = H, use_nw = FALSE)),
  cumul = lin(lp_lin_iv(endog_data = en, shock = d[, "z", drop = FALSE], lags_endog_lin = 2,
                        trend = 0, confint = 1, hor = H, cumul_mult = TRUE)))
nl1 = function(f) list(s1 = f$irf_s1_mean[1, ], s1se = f$irf_s1_up[1, ] - f$irf_s1_mean[1, ],
                       s2 = f$irf_s2_mean[1, ], s2se = f$irf_s2_up[1, ] - f$irf_s2_mean[1, ])
ch = lp_nl(endog_data = en, lags_endog_lin = 2, lags_endog_nl = 2, trend = 0, shock_type = 1,
           confint = 1, hor = H, switching = d$state, use_hp = FALSE, gamma = 3)
state = list(
  logistic = nl1(lp_nl_iv(endog_data = en, lags_endog_nl = 2, shock = d[, "z", drop = FALSE],
                          trend = 0, confint = 1, hor = H, switching = d$state,
                          use_hp = FALSE, gamma = 3)),
  hp = nl1(lp_nl_iv(endog_data = en, lags_endog_nl = 1, shock = d[, "z", drop = FALSE],
                    trend = 1, confint = 1, hor = H, switching = d$state, use_hp = TRUE,
                    lambda = 1600, gamma = 2)),
  dummy = nl1(lp_nl_iv(endog_data = en, lags_endog_nl = 2, shock = d[, "z", drop = FALSE],
                       trend = 0, confint = 1, hor = H, switching = d$d, use_logistic = FALSE,
                       use_hp = FALSE, nw_lag = 2)),
  cholesky = list(s1 = ch$irf_s1_mean[1, , 2], s1se = ch$irf_s1_up[1, , 2] - ch$irf_s1_mean[1, , 2],
                  s2 = ch$irf_s2_mean[1, , 2], s2se = ch$irf_s2_up[1, , 2] - ch$irf_s2_mean[1, , 2]))
pl = function(f) list(irf = as.numeric(f$irf_panel_mean),
                      se = as.numeric(f$irf_panel_up - f$irf_panel_mean))
panel = list(
  default = pl(lp_lin_panel(data_set = p, endog_data = "y", shock = "x", cumul_mult = TRUE,
                            diff_shock = TRUE, confint = 1, hor = H)),
  cluster = pl(lp_lin_panel(data_set = p, endog_data = "y", shock = "x", cumul_mult = FALSE,
                            diff_shock = FALSE, c_exog_data = "c", l_exog_data = c("y", "x"),
                            lags_exog_data = 2, confint = 1, hor = H, robust_cov = "vcovHC",
                            robust_method = "arellano", robust_type = "HC0",
                            robust_cluster = "group")),
  scc_twoways = pl(lp_lin_panel(data_set = p, endog_data = "y", shock = "x", cumul_mult = FALSE,
                                diff_shock = FALSE, l_exog_data = "y", lags_exog_data = 1,
                                confint = 1, hor = H, panel_effect = "twoways",
                                robust_cov = "vcovSCC", robust_type = "HC0")),
  iv = pl(lp_lin_panel(data_set = p, endog_data = "y", shock = "x", cumul_mult = FALSE,
                       diff_shock = FALSE, iv_reg = TRUE, instrum = "z", c_exog_data = "c",
                       confint = 1, hor = H)),
  pooling = pl(lp_lin_panel(data_set = p, endog_data = "y", shock = "x", cumul_mult = FALSE,
                            diff_shock = FALSE, l_exog_data = "y", lags_exog_data = 1,
                            confint = 1, hor = H, panel_model = "pooling", robust_cov = "vcovHC",
                            robust_type = "HC0", robust_cluster = "group",
                            robust_method = "arellano")))
dh = list()
for (K in 1:2) for (tst in c("Ztilde", "Zbar", "Wbar")) {
  r = pgrangertest(y ~ x, data = p, index = c("id", "time"), order = K, test = tst)
  dh[[paste0(tst, K)]] = c(unname(r$statistic), if (is.null(r$p.value)) NA else unname(r$p.value))
}
r = pgrangertest(y ~ x, data = pu, index = c("id", "time"), order = 2, test = "Ztilde")
dh[["unbalanced"]] = c(unname(r$statistic), unname(r$p.value))
dh[["wald_K1"]] = pgrangertest(y ~ x, data = p, index = c("id", "time"), order = 1)$indgranger$Chisq
s = id.chol(vars::VAR(v, p = 2, type = "const"))
hd_out = lapply(1:3, function(k) unname(as.matrix(hd(s, series = k)$hidec[, 4:6])))
out = list(
  versions = list(R = R.version.string, lpirfs = as.character(packageVersion("lpirfs")),
                  plm = as.character(packageVersion("plm")),
                  svars = as.character(packageVersion("svars"))),
  lp_lin_iv = iv, lp_nl = state, lp_lin_panel = panel, pgrangertest = dh, svars_hd = hd_out)
write_json(out, file.path(here, "ts_causal_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
