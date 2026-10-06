# Reference values for tests/reference_parity/test_forecasting_r_parity.py.
#   FIXCSV=tests/reference_parity/_fixtures/forecasting_series.csv \
#   OUTJSON=tests/reference_parity/_fixtures/forecasting_boxcox_R.json Rscript tests/reference_parity/_fixtures/_generate_forecasting_boxcox_R.R
suppressMessages({library(forecast); library(jsonlite)})
d <- read.csv(Sys.getenv("FIXCSV")); rd <- function(s) { x <- d[d$series==s,]; ts(x$y, frequency=x$period[1]) }
fcl <- function(fc) list(mean=as.numeric(fc$mean), lower=unname(as.matrix(fc$lower)), upper=unname(as.matrix(fc$upper)))
q <- rd("quarterly"); w <- rd("walk"); out <- list(version=as.character(packageVersion("forecast")))
for (ba in c(FALSE, TRUE)) { tag <- if (ba) "adj" else "med"
  out[[paste0("drift_log_",tag)]] <- fcl(rwf(w, drift=TRUE, lambda=0, h=10, level=c(80,95), biasadj=ba))
  out[[paste0("snaive_l3_",tag)]] <- fcl(snaive(q, lambda=0.3, h=8, level=c(80,95), biasadj=ba))
  f <- Arima(q, order=c(1,0,0), seasonal=c(0,1,1), lambda=0.5, biasadj=ba, method="ML")
  out[[paste0("arima_l5_",tag)]] <- c(fcl(forecast(f, h=8, level=c(80,95), biasadj=ba)), list(loglik=f$loglik))
}
write_json(out, Sys.getenv("OUTJSON"), digits=NA, auto_unbox=TRUE); cat("ok\n")
