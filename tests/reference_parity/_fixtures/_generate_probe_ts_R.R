# Data and R reference values for tests/reference_parity/test_probe_ts_parity.py
# (2026-11 macro / finance time-series probe). Run from the repository root:
#
#   Rscript tests/reference_parity/_fixtures/_generate_probe_ts_R.R
#
# Needs vars, urca, forecast, strucchange, sandwich, lmtest and jsonlite.
# Writes the CSV files the Python and Stata sides read, taken from datasets
# shipped with R, vars and urca (probe_ts_canada.csv, probe_ts_denmark.csv,
# probe_ts_nile.csv, probe_ts_lynx.csv, probe_ts_airpassengers.csv), the
# rolling-origin forecast errors probe_ts_dm_errors.csv, and probe_ts_R.json.
# GPL packages are run as black boxes; nothing here is derived from their
# source.
args <- commandArgs(trailingOnly = FALSE)
fx <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
suppressMessages({
  library(vars); library(urca); library(forecast); library(strucchange)
  library(sandwich); library(lmtest); library(jsonlite)
})
w <- function(d, n) write.csv(d, file.path(fx, paste0("probe_ts_", n, ".csv")),
                              row.names = FALSE)
data(Canada); data(denmark)
canada <- as.data.frame(Canada)
w(canada, "canada")
w(denmark[, c("LRM", "LRY", "IBO", "IDE")], "denmark")
w(data.frame(y = as.numeric(Nile)), "nile")
w(data.frame(y = as.numeric(lynx)), "lynx")
w(data.frame(y = as.numeric(AirPassengers)), "airpassengers")

out <- list(versions = list(
  R = R.version.string,
  vars = as.character(packageVersion("vars")),
  urca = as.character(packageVersion("urca")),
  forecast = as.character(packageVersion("forecast")),
  strucchange = as.character(packageVersion("strucchange")),
  sandwich = as.character(packageVersion("sandwich"))
))

## ---- ARIMA: observed-information standard errors (stats::arima) -----------
nile <- as.numeric(Nile); ly <- log10(as.numeric(lynx))
la <- ts(log(as.numeric(AirPassengers)), frequency = 12)
se <- function(m) as.list(sqrt(diag(vcov(m))))
out$arima <- list(
  nile_011 = se(arima(nile, order = c(0, 1, 1), method = "ML")),
  lynx_200 = se(arima(ly, order = c(2, 0, 0), method = "ML")),
  air = se(arima(la, order = c(0, 1, 1), seasonal = c(0, 1, 1), method = "ML"))
)

## ---- VAR(2) on Canada: lm coefficients, t p-values, predict ----------------
V <- c("e", "prod", "rw", "U")
m <- VAR(canada[, V], p = 2, type = "const")
co <- lapply(V, function(eq) {
  s <- coef(m)[[eq]]
  rn <- sub("^(.*)\\.l([12])$", "L\\2.\\1", rownames(s)); rn <- sub("const", "_cons", rn)
  list(term = rn, coef = unname(s[, 1]), se = unname(s[, 2]), p = unname(s[, 4]))
})
names(co) <- V
p <- predict(m, n.ahead = 8, ci = 0.95)
fc <- lapply(V, function(v) list(fcst = unname(p$fcst[[v]][, 1]),
                                 lower = unname(p$fcst[[v]][, 2]),
                                 upper = unname(p$fcst[[v]][, 3])))
names(fc) <- V
nt <- normality.test(m)$jb.mul
out$var <- list(coef = co, predict = fc, normality = list(
  jb = unname(nt$JB$statistic), jb_p = unname(nt$JB$p.value),
  skew = unname(nt$Skewness$statistic), skew_p = unname(nt$Skewness$p.value),
  kurt = unname(nt$Kurtosis$statistic), kurt_p = unname(nt$Kurtosis$p.value)))

## ---- Structural breaks: BIC of the Nile partitions (strucchange) ----------
nd <- data.frame(y = nile)
bp <- breakpoints(y ~ 1, data = nd, h = 0.15)
sm <- summary(bp)
out$breaks <- list(bic = unname(sm$RSS["BIC", ]), rss = unname(sm$RSS["RSS", ]),
                   date_m1 = unname(sm$breakpoints[1, ][!is.na(sm$breakpoints[1, ])]))

## ---- Interrupted time series: segmented regression + Newey-West -----------
a <- data.frame(ly = as.numeric(la), i = 0:143)
a$D <- as.numeric(a$i >= 96); a$tp <- pmax(a$i - 96, 0)
its <- lm(ly ~ i + D + tp + sin(2 * pi * i / 12) + cos(2 * pi * i / 12) +
            sin(4 * pi * i / 12) + cos(4 * pi * i / 12), a)
ct <- coeftest(its, NeweyWest(its, lag = 12, prewhite = FALSE, adjust = FALSE), df = Inf)
out$its <- list(coef = unname(ct[, 1]), se = unname(ct[, 2]), p = unname(ct[, 4]))

## ---- Diebold-Mariano on rolling-origin errors -----------------------------
y <- ts(as.numeric(AirPassengers), frequency = 12)
e1 <- tsCV(y, snaive, h = 4, initial = 59)
e2 <- tsCV(y, rwf, drift = TRUE, h = 4, initial = 59)
err <- data.frame(e1 = e1[, 1], e2 = e2[, 1], e1h4 = e1[, 4], e2h4 = e2[, 4])
err <- err[60:143, ]
w(err, "dm_errors")
dm <- list()
for (alt in c("two.sided", "less", "greater")) for (pw in 1:2) {
  t <- dm.test(err$e1, err$e2, alternative = alt, h = 1, power = pw)
  dm[[paste0("h1_", alt, "_p", pw)]] <- list(stat = unname(t$statistic), p = unname(t$p.value))
}
ok <- complete.cases(err$e1h4, err$e2h4)
for (vr in c("acf", "bartlett")) {
  t <- dm.test(err$e1h4[ok], err$e2h4[ok], h = 4, power = 2, varestimator = vr)
  dm[[paste0("h4_", vr)]] <- list(stat = unname(t$statistic), p = unname(t$p.value))
}
out$dm <- dm

writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(fx, "probe_ts_R.json"))
