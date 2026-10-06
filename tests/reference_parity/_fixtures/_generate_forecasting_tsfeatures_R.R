# Reference values for tests/reference_parity/test_forecasting_r_parity.py.
#   FIXCSV=tests/reference_parity/_fixtures/forecasting_series.csv \
#   OUTJSON=tests/reference_parity/_fixtures/forecasting_tsfeatures_R.json Rscript tests/reference_parity/_fixtures/_generate_forecasting_tsfeatures_R.R
suppressMessages({library(tsfeatures); library(jsonlite)})
d <- read.csv(Sys.getenv("FIXCSV")); out <- list()
for (s in c("quarterly","monthly","walk","arma")) {
  x <- d[d$series==s,]; y <- ts(x$y, frequency=x$period[1])
  f <- suppressWarnings(tsfeatures(y, features=c("acf_features","pacf_features","stl_features","lumpiness","stability","crossing_points","flat_spots","arch_stat","max_level_shift","max_var_shift")))
  out[[s]] <- as.list(as.data.frame(f))
}
write_json(list(version=as.character(packageVersion("tsfeatures")), features=out), Sys.getenv("OUTJSON"), digits=NA, auto_unbox=TRUE); cat("ok\n")
