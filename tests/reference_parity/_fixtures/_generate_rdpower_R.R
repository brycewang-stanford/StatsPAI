# Black-box fixture for the clean-room rewrite of sp.rdpower / sp.rdsampsi
# (statspai.rd.rdpower). Runs R rdpower 3.0 (Cattaneo, Titiunik and
# Vazquez-Bare; GPL-3) only through its documented interface on the Senate
# data and records the returned values; nothing here reads or reproduces its
# source.
#
#   Rscript tests/reference_parity/_fixtures/_generate_rdpower_R.R
#
# Reads rdsenate.csv next to this file; writes rdpower_R.json:
#   power_<cell>   rdpower(data = cbind(vote, margin), ...): power.rbc, se.rbc,
#                  Vl.rb, Vr.rb (scaled variances), N.l, N.r, Nh.l, Nh.r,
#                  samph.l, samph.r, tau, alpha
#   sampsi_<cell>  rdsampsi(data = ...): sampsi.h.tot / .l / .r (window sizes
#                  M, M_left, M_right), sampsi.tot (implied total n), var.l,
#                  var.r, N.*, Nh.*, samph.*, tau, beta, alpha
#
# Each call runs under a 10-second elapsed-time limit: rdsampsi's iterative
# solver does not return for some inputs (e.g. tau = 0.5, or tau = 1 with
# beta = 0.9 on these data). Such calls are recorded with the error message.

suppressMessages({
  library(rdpower)
  library(jsonlite)
})
stopifnot(as.character(packageVersion("rdpower")) == "3.0")
stopifnot(as.character(packageVersion("rdrobust")) == "4.0.0")

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
if (is.null(here)) {
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- dirname(normalizePath(f))
}

d <- read.csv(file.path(here, "rdsenate.csv"))
YX <- cbind(d$vote, d$margin)

out <- list()
run <- function(tag, fn, args) {
  setTimeLimit(elapsed = 10, transient = TRUE)
  res <- tryCatch(
    suppressWarnings({
      invisible(capture.output(v <- do.call(fn, c(list(data = YX), args))))
      v
    }),
    error = function(e) e
  )
  setTimeLimit(elapsed = Inf)
  if (inherits(res, "error")) {
    out[[tag]] <<- list(args = args, error = conditionMessage(res))
    return(invisible())
  }
  keep <- res[vapply(res, function(z) is.numeric(z) || is.character(z), TRUE)]
  out[[tag]] <<- c(list(args = args), lapply(keep, unname))
}

# rdpower: power of the robust bias-corrected test --------------------------
for (tau in c(1, 3, 5, 10)) run(sprintf("power_tau%g", tau), rdpower, list(tau = tau))
run("power_p2", rdpower, list(tau = 3, p = 2))
run("power_uniform_hc1", rdpower, list(tau = 3, kernel = "uniform", vce = "hc1"))
run("power_cutoff5", rdpower, list(tau = 3, cutoff = 5))
run("power_alpha01", rdpower, list(tau = 3, alpha = 0.01))
run("power_msetwo", rdpower, list(tau = 3, bwselect = "msetwo"))
run("power_h", rdpower, list(tau = 3, h = c(12, 20)))

# rdsampsi: smallest window sample reaching power beta -----------------------
for (tau in c(2, 3, 5, 7.5, 10, 20)) {
  run(sprintf("sampsi_tau%g", tau), rdsampsi, list(tau = tau))
}
for (tau in c(3, 5)) {
  run(sprintf("sampsi_beta09_tau%g", tau), rdsampsi, list(tau = tau, beta = 0.9))
  run(sprintf("sampsi_alpha01_tau%g", tau), rdsampsi, list(tau = tau, alpha = 0.1))
}
run("sampsi_p2", rdsampsi, list(tau = 3, p = 2))
run("sampsi_uniform_hc1", rdsampsi, list(tau = 3, kernel = "uniform", vce = "hc1"))
run("sampsi_msetwo", rdsampsi, list(tau = 3, bwselect = "msetwo"))
run("sampsi_cutoff5", rdsampsi, list(tau = 5, cutoff = 5))
# the solver that does not return (recorded, not compared)
run("sampsi_tau05", rdsampsi, list(tau = 0.5))

out[["_meta"]] <- list(
  n = nrow(d),
  rdpower_version = as.character(packageVersion("rdpower")),
  rdrobust_version = as.character(packageVersion("rdrobust")),
  R_version = R.version.string
)
write_json(out, file.path(here, "rdpower_R.json"), auto_unbox = TRUE,
           digits = NA, pretty = TRUE)
cat("wrote rdpower_R.json\n")
