# Reference values for tests/reference_parity/test_dlm_parity.py
#
# Runs on the committed synthetic file dlm.csv and writes dlm_R.json.
# Requires: dlm, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_dlm_R.R
suppressMessages({library(dlm); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "dlm.csv"))
out = list(versions = list(R = R.version.string, dlm = as.character(packageVersion("dlm"))))
pack = function(y, mod) {
  f = dlmFilter(y, mod); s = dlmSmooth(f)
  vf = dlmSvd2var(f$U.C, f$D.C); vs = dlmSvd2var(s$U.S, s$D.S)
  k = ncol(as.matrix(f$m))
  list(m = as.matrix(f$m)[-1, , drop = FALSE], s = as.matrix(s$s)[-1, , drop = FALSE],
       C = matrix(t(sapply(vf[-1], function(v) diag(as.matrix(v)))), ncol = k, byrow = (k == 1)),
       S = matrix(t(sapply(vs[-1], function(v) diag(as.matrix(v)))), ncol = k, byrow = (k == 1)),
       f = as.numeric(f$f), negll = dlmLL(y, mod))
}
## time-varying regression at fixed variances
out$tvp = pack(d$y, dlmModReg(d$x, dV = 0.3, dW = c(0.05, 0.02), m0 = c(0, 0), C0 = diag(1e7, 2)))
## a proper prior and one constant coefficient
out$tvp_prior = pack(d$y, dlmModReg(d$x, dV = 0.2, dW = c(0.08, 0), m0 = c(1, 0.5), C0 = diag(c(4, 1))))
## local level at fixed variances
out$level = pack(d$level, dlmModPoly(1, dV = 1.2, dW = 0.1, m0 = 0, C0 = 1e7))
## maximum likelihood
fit = dlmMLE(d$y, parm = c(0, 0, 0), function(p) dlmModReg(d$x, dV = exp(p[1]), dW = exp(p[2:3])),
             control = list(maxit = 2000, factr = 10))
out$mle_tvp = list(par = exp(fit$par), negll = fit$value, convergence = fit$convergence)
fit = dlmMLE(d$level, parm = c(0, 0), function(p) dlmModPoly(1, dV = exp(p[1]), dW = exp(p[2])),
             control = list(maxit = 2000, factr = 10))
out$mle_level = list(par = exp(fit$par), negll = fit$value, convergence = fit$convergence)
write_json(out, file.path(here, "dlm_R.json"), digits = 17, auto_unbox = TRUE)
