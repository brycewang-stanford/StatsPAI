# Reference values for tests/reference_parity/test_scest_multi_parity.py
#
# R scpi (GPL-3) is used as a black box: scdataMulti + scest through their
# documented interface, outputs only. Install it into a private library
# (it needs CVXR >= 1.9), e.g.
#   R_LIBS=<private lib> Rscript tests/reference_parity/_fixtures/_generate_scest_multi_R.R
suppressMessages({
  .libPaths(c(Sys.getenv("R_LIBS"), .libPaths()))
  library(scpi); library(jsonlite)
})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "scest_multi_panel.csv"))
grab = function(sd, e) list(
  w = as.numeric(e$est.results$w), w_names = rownames(e$est.results$w),
  Y_post_fit = as.numeric(e$est.results$Y.post.fit),
  Y_post_fit_names = rownames(e$est.results$Y.post.fit),
  Y_pre_fit = as.numeric(e$est.results$Y.pre.fit),
  Q = unname(sapply(e$est.results$w.constr, function(z) if (is.null(z$Q)) NA else z$Q)),
  donors = colnames(sd$B))
out = list()
for (eff in c("unit-time", "unit", "time")) {
  sd = scdataMulti(d, id.var = "unit", time.var = "year", outcome.var = "y",
                   treatment.var = "treat", effect = eff, verbose = FALSE)
  for (wc in c("simplex", "lasso", "ridge", "ols")) {
    out[[paste(eff, wc, sep = "|")]] = grab(sd, scest(sd, w.constr = list(name = wc)))
  }
}
for (eff in c("unit-time", "time")) {
  sd = scdataMulti(d, id.var = "unit", time.var = "year", outcome.var = "y",
                   treatment.var = "treat", effect = eff, anticipation = 1, verbose = FALSE)
  out[[paste(eff, "simplex", "anticipation1", sep = "|")]] = grab(sd, scest(sd, w.constr = list(name = "simplex")))
}
sd = scdataMulti(d, id.var = "unit", time.var = "year", outcome.var = "y",
                 treatment.var = "treat", units.est = c("u01", "u03"), verbose = FALSE)
out[["unit-time|simplex|units_est"]] = grab(sd, scest(sd, w.constr = list(name = "simplex")))
out$versions = list(R = R.version.string, scpi = as.character(packageVersion("scpi")),
                    CVXR = as.character(packageVersion("CVXR")))
write_json(out, file.path(here, "scest_multi_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
