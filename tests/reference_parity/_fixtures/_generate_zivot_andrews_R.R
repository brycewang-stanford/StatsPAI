# urca 1.3-4 reference for sp.zivot_andrews.
# Run from tests/reference_parity/_fixtures; writes zivot_andrews_R.json.
suppressMessages({library(urca); library(jsonlite)})
z <- read.csv("zivot_andrews.csv")
out <- list(urca = as.character(packageVersion("urca")), cases = list())
for (v in c("rw", "brk")) for (m in c("intercept", "trend", "both")) for (k in c(0, 3)) {
  u <- ur.za(z[[v]], model = m, lag = k)
  out$cases[[length(out$cases) + 1]] <- list(
    series = v, model = m, lag = k, stat = u@teststat, bpoint = u@bpoint,
    cval = u@cval, tstats = as.numeric(u@tstats),
    coef = as.numeric(coef(u@testreg)), se = as.numeric(coef(summary(u@testreg))[, 2]))
}
write_json(out, "zivot_andrews_R.json", digits = NA, auto_unbox = TRUE)
