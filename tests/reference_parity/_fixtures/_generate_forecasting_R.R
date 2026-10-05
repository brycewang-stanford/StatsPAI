# Reference values for the forecasting functions, from the R packages
# written by the authors of the methods: forecast (Hyndman et al.) for
# ets / Arima / auto.arima / naive / snaive / rwf / meanf / accuracy /
# tsCV / ndiffs / nsdiffs / BoxCox.lambda / mstl / stlf / fourier, stats
# for stl / decompose / Box.test, hts for MinT / combinef.
#
#   Rscript tests/reference_parity/_fixtures/_generate_forecasting_R.R
#
# Reads forecasting_series.csv, forecasting_hts_*.csv and
# forecasting_sp_ets.json (written by _generate_forecasting_data.py) and
# writes forecasting_R.json next to them.
suppressMessages({library(forecast); library(jsonlite)})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(here) == 0 || here == "") here <- "tests/reference_parity/_fixtures"
dat <- read.csv(file.path(here, "forecasting_series.csv"))
rd <- function(n) { d <- dat[dat$series == n, ]; ts(d$y, frequency = d$period[1]) }
fcl <- function(fc) list(mean = as.numeric(fc$mean), lower = unname(as.matrix(fc$lower)),
                         upper = unname(as.matrix(fc$upper)))
out <- list(versions = list(R = R.version.string, forecast = as.character(packageVersion("forecast"))))

# ---- ETS ---------------------------------------------------------------
spfits <- fromJSON(file.path(here, "forecasting_sp_ets.json"), simplifyVector = FALSE)
ets_out <- list()
for (tag in names(spfits)) {
  f <- spfits[[tag]]
  y <- rd(f$series); m <- frequency(y)
  et <- substr(f$model, 1, 1); tt <- substr(f$model, 2, 2); st <- substr(f$model, 3, 3)
  damped <- isTRUE(f$damped)
  fit <- ets(y, model = f$model, damped = if (tt == "N") NULL else damped)
  h <- if (m > 1) 2 * m + 2 else 10
  fc <- forecast(fit, h = h, level = c(80, 95))
  # forecast's own likelihood code at the parameters sp.ets estimated
  p <- f$params; ini <- unlist(f$initial_state)
  r <- forecast:::pegelsresid.C(y, m, ini, et, tt, st, damped, p$alpha,
                                if (is.null(p$beta)) NULL else p$beta,
                                if (is.null(p$gamma)) NULL else p$gamma,
                                if (is.null(p$phi)) NULL else p$phi, 3)
  ets_out[[tag]] <- c(list(series = f$series, model = f$model, damped = damped, period = m,
    method = fit$method, par = as.list(fit$par), loglik = fit$loglik, aic = fit$aic,
    aicc = fit$aicc, bic = fit$bic, sigma2 = fit$sigma2, fitted = as.numeric(fit$fitted),
    residuals = as.numeric(fit$residuals), init_state = as.numeric(fit$states[1, ]),
    last_state = as.numeric(fit$states[nrow(fit$states), ]),
    loglik_at_sp_params = -0.5 * r$lik), fcl(fc))
}
out$ets <- ets_out
out$ets_auto <- list()
for (n in c("level", "trend", "quarterly", "monthly", "arma", "walk")) {
  fit <- ets(rd(n)); out$ets_auto[[n]] <- list(method = fit$method, aicc = fit$aicc, loglik = fit$loglik)
}

# ---- benchmark methods, accuracy, cross-validation ---------------------
walk <- rd("walk"); q <- rd("quarterly")
out$simple <- list(
  naive = c(fcl(naive(walk, h = 10, level = c(80, 95))), list(residuals = as.numeric(residuals(naive(walk))))),
  drift = c(fcl(rwf(walk, h = 10, drift = TRUE, level = c(80, 95))), list(residuals = as.numeric(residuals(rwf(walk, drift = TRUE))))),
  snaive = c(fcl(snaive(q, h = 10, level = c(80, 95))), list(residuals = as.numeric(residuals(snaive(q))))),
  mean = fcl(meanf(q, h = 10, level = c(80, 95))))
tr <- window(q, end = c(18, 4)); te <- window(q, start = c(19, 1))
a1 <- accuracy(snaive(tr, h = length(te)), te); a2 <- accuracy(rwf(tr, h = length(te), drift = TRUE), te)
out$accuracy <- list(n_train = length(tr), names = colnames(a1), snaive = as.numeric(a1[2, ]), drift = as.numeric(a2[2, ]))
out$tscv <- list(drift_h3 = unname(as.matrix(tsCV(walk, rwf, drift = TRUE, h = 3))),
                 snaive_h4 = unname(as.matrix(tsCV(q, snaive, h = 4))))

# ---- tests and tools ----------------------------------------------------
dw <- diff(walk); ar <- rd("arma")
bt <- function(x, ...) { r <- Box.test(x, ...); c(as.numeric(r$statistic), as.numeric(r$parameter), r$p.value) }
out$box <- list(lb10 = bt(dw, lag = 10, type = "Ljung-Box"), bp10 = bt(dw, lag = 10, type = "Box-Pierce"),
                lb12_fitdf3 = bt(ar, lag = 12, type = "Ljung-Box", fitdf = 3))
out$ndiffs <- list(level = ndiffs(rd("level")), trend = ndiffs(rd("trend")), arma = ndiffs(ar), walk = ndiffs(walk),
                   monthly_sdiff = ndiffs(diff(rd("monthly"), 12)))
ns <- function(x) c(nsdiffs(x), as.numeric(forecast:::seas.heuristic(x))[1])
out$nsdiffs <- list(quarterly = ns(q), monthly = ns(rd("monthly")), walk4 = ns(ts(as.numeric(walk), frequency = 4)))
out$lambda <- list(quarterly = BoxCox.lambda(q), monthly = BoxCox.lambda(rd("monthly")), level = BoxCox.lambda(rd("level")))
mo <- rd("monthly")
st <- function(...) unname(as.matrix(stl(mo, ...)$time.series))
out$stl <- list(default11 = st(s.window = 11), robust = st(s.window = 13, t.window = 21, robust = TRUE),
                robust_outer2 = st(s.window = 13, t.window = 21, robust = TRUE, outer = 2),
                periodic = st(s.window = "periodic"),
                exact = st(s.window = 7, s.degree = 1, inner = 5, s.jump = 1, t.jump = 1, l.jump = 1))
mm <- msts(as.numeric(rd("multi")), seasonal.periods = c(8, 40)); ms <- mstl(mm)
out$mstl <- list(names = colnames(ms), values = unname(as.matrix(ms)))
out$stlf_naive <- fcl(stlf(mo, h = 24, method = "naive", level = c(80, 95)))
dc <- decompose(mo); dm <- decompose(q, type = "multiplicative")
out$classical <- list(add_trend = as.numeric(dc$trend), add_seasonal = as.numeric(dc$seasonal),
                      mult_trend = as.numeric(dm$trend), mult_seasonal = as.numeric(dm$seasonal))
out$fourier <- list(q_K2 = unname(as.matrix(fourier(q, K = 2))), m_K3 = unname(as.matrix(fourier(mo, K = 3))),
                    m_K3_future = unname(as.matrix(fourier(mo, K = 3, h = 6))))

# ---- ARIMA --------------------------------------------------------------
desc <- function(m) list(order = as.numeric(arimaorder(m)), coef = as.list(coef(m)), se = as.numeric(sqrt(diag(m$var.coef))),
                         loglik = m$loglik, aic = m$aic, aicc = m$aicc, bic = m$bic, sigma2 = m$sigma2, label = as.character(m))
af <- function(m, h, ...) c(desc(m), fcl(forecast(m, h = h, level = c(80, 95), ...)))
out$arima <- list(
  arma_201 = af(Arima(ar, order = c(2, 0, 1), method = "ML"), 8),
  walk_011_drift = af(Arima(walk, order = c(0, 1, 1), include.drift = TRUE, method = "ML"), 8),
  monthly_011_011 = af(Arima(mo, order = c(0, 1, 1), seasonal = c(0, 1, 1), method = "ML"), 12),
  quarterly_log_100_011_drift = af(Arima(log(q), order = c(1, 0, 0), seasonal = c(0, 1, 1), include.drift = TRUE, method = "ML"), 8))
dy <- rd("dyn_y"); dx <- as.numeric(rd("dyn_x"))
xf <- c(0.5, -0.5, 1, 0, 0.25, -1)
fdyn <- Arima(dy, order = c(1, 0, 0), xreg = cbind(x = dx), method = "ML")
out$arima$dyn_100_x <- c(af(fdyn, 6, xreg = cbind(x = xf)), list(x_future = xf))
auto <- list()
for (n in c("level", "trend", "arma", "walk", "quarterly", "monthly")) {
  y <- rd(n)
  a <- auto.arima(y, approximation = FALSE)
  b <- if (frequency(y) > 1) NULL else auto.arima(y, approximation = FALSE, stepwise = FALSE)
  auto[[n]] <- list(stepwise = desc(a), full = if (is.null(b)) NULL else desc(b))
}
out$auto_arima <- auto
out$auto_arima_dyn <- desc(auto.arima(dy, xreg = cbind(x = dx), approximation = FALSE))

# ---- reconciliation (hts, optional) -------------------------------------
if (requireNamespace("hts", quietly = TRUE)) {
  res <- as.matrix(read.csv(file.path(here, "forecasting_hts_res.csv"), header = FALSE))
  base <- as.matrix(read.csv(file.path(here, "forecasting_hts_base.csv"), header = FALSE))
  nodes <- list(3, c(2, 3, 2)); H <- function(x) unname(as.matrix(x))
  out$hts <- list(version = as.character(packageVersion("hts")),
    mint_shrink = H(hts::MinT(base, nodes = nodes, residual = res, covariance = "shr", keep = "all", algorithms = "lu")),
    mint_cov = H(hts::MinT(base, nodes = nodes, residual = res, covariance = "sam", keep = "all", algorithms = "lu")),
    ols = H(hts::combinef(base, nodes = nodes, weights = NULL, keep = "all", algorithms = "lu")),
    wls_var = H(hts::combinef(base, nodes = nodes, weights = 1 / colMeans(res^2), keep = "all", algorithms = "lu")),
    wls_struct = H(hts::combinef(base, nodes = nodes, weights = 1 / c(7, 2, 3, 2, rep(1, 7)), keep = "all", algorithms = "lu")))
} else stop("package 'hts' is needed for the reconciliation references")
write_json(out, file.path(here, "forecasting_R.json"), digits = NA, auto_unbox = TRUE, null = "null", na = "null")
cat("wrote forecasting_R.json\n")
for (n in names(auto)) cat(n, ":", auto[[n]]$stepwise$label, "|", if (is.null(auto[[n]]$full)) "-" else auto[[n]]$full$label, "\n")
for (n in names(out$ets_auto)) cat(n, ":", out$ets_auto[[n]]$method, "\n")
