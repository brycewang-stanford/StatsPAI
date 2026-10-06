# Reference values for tests/reference_parity/test_lrvar_parity.py
#
# Runs on the committed synthetic file lrvar.csv and writes lrvar_R.json.
# The long-run variance of a series is n times the HAC variance of its mean,
# i.e. of the coefficient of an intercept-only regression.
# Requires: sandwich, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_lrvar_R.R
suppressMessages({library(sandwich); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "lrvar.csv"))
n = nrow(d)
Y = as.matrix(d[, c("y1", "y2")])
out = list(versions = list(R = R.version.string,
                           sandwich = as.character(packageVersion("sandwich"))), n = n)
m1 = lm(x ~ 1, data = d)
m2 = lm(Y ~ 1)
kernels = c("Bartlett", "Parzen", "Quadratic Spectral", "Tukey-Hanning", "Truncated")
cases = list()
add = function(model, tag, kernel, bwrule, prewhite, adjust) {
  if (bwrule == "andrews") {
    bw = bwAndrews(model, kernel = kernel, prewhite = prewhite)
  } else if (bwrule == "newey-west") {
    bw = bwNeweyWest(model, kernel = kernel, prewhite = prewhite)
  } else bw = as.numeric(bwrule)
  v = kernHAC(model, kernel = kernel, bw = bw, prewhite = prewhite, adjust = adjust)
  list(series = tag, kernel = kernel, rule = bwrule, prewhite = prewhite,
       adjust = adjust, bw = bw, lrvar = as.numeric(n * v))
}
for (tag in c("x", "Y")) {
  model = if (tag == "x") m1 else m2
  for (k in kernels) for (pw in c(0, 1)) for (adj in c(FALSE, TRUE)) {
    rules = c("andrews", "7.5")
    if (k %in% c("Bartlett", "Parzen", "Quadratic Spectral")) rules = c(rules, "newey-west")
    for (r in rules) cases[[length(cases) + 1]] = add(model, tag, k, r, pw, adj)
  }
}
out$kernhac = cases
## NeweyWest(): Bartlett weights 1 - j / (lag + 1), lag = floor(bwNeweyWest)
nw = list()
for (tag in c("x", "Y")) {
  model = if (tag == "x") m1 else m2
  for (pw in c(0, 1)) for (adj in c(FALSE, TRUE)) {
    nw[[length(nw) + 1]] = list(series = tag, prewhite = pw, adjust = adj, lag = "auto",
      bw = bwNeweyWest(model, prewhite = pw),
      lrvar = as.numeric(n * NeweyWest(model, prewhite = pw, adjust = adj)))
    nw[[length(nw) + 1]] = list(series = tag, prewhite = pw, adjust = adj, lag = 4,
      lrvar = as.numeric(n * NeweyWest(model, lag = 4, prewhite = pw, adjust = adj)))
  }
}
out$neweywest = nw
## sandwich::lrvar returns the variance of the mean (long-run variance / n)
out$lrvar = list(
  andrews = lrvar(d$x, type = "Andrews"),
  andrews_raw = lrvar(d$x, type = "Andrews", prewhite = FALSE, adjust = FALSE),
  neweywest = lrvar(d$x, type = "Newey-West"),
  neweywest_raw = lrvar(d$x, type = "Newey-West", prewhite = FALSE, adjust = FALSE),
  andrews_Y = as.numeric(lrvar(Y, type = "Andrews")),
  neweywest_Y = as.numeric(lrvar(Y, type = "Newey-West")))
write_json(out, file.path(here, "lrvar_R.json"), digits = NA, auto_unbox = TRUE)
