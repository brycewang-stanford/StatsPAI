# Reference values for tests/reference_parity/test_tvp_var_parity.py
#
# Each equation of a VAR(2) with random-walk coefficients is a Gaussian
# state space model. KFAS filters and smooths it at fixed variances; the
# numbers are compared with sp.tvp_var(method = "kalman") at the same
# variances. KFAS dates its prior at the first observation and sp.dlm one
# period earlier, so P1 = C0 + W. Runs on the committed synthetic file tvp_var.csv and writes
# tvp_var_R.json. Requires: KFAS, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_tvp_var_R.R
suppressMessages({library(KFAS); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = as.matrix(read.csv(file.path(here, "tvp_var.csv")))
n = nrow(d); K = ncol(d); p = 2; k = K * p + 1
Y = d[(p + 1):n, ]
X = cbind(d[p:(n - 1), ], d[1:(n - 2), ], 1)          # lag 1, lag 2, constant
V = c(1.0, 0.9, 0.7)                                   # error variances
W = outer(1:K, 1:k, function(i, j) 0.002 * ((i + j) %% 3))  # zeros included
rows = seq(1, nrow(Y), by = 3)                         # every third date, to keep the file small
run = function(C0) {
  eqs = lapply(1:K, function(i) {
    y = Y[, i]
    mod = SSModel(y ~ -1 + SSMregression(~ -1 + X, Q = diag(W[i, ]), a1 = rep(0, k),
                                         P1 = diag(C0 + W[i, ], k), P1inf = diag(0, k)), H = V[i])
    out = KFS(mod, filtering = "state", smoothing = "state")
    list(filtered = out$att[rows, ], smoothed = out$alphahat[rows, ],
         filtered_var = t(apply(out$Ptt, 3, diag))[rows, ],
         smoothed_var = t(apply(out$V, 3, diag))[rows, ],
         loglik = as.numeric(logLik(mod)))
  })
  eqs
}
out = list(versions = list(R = R.version.string, KFAS = as.character(packageVersion("KFAS"))),
           lags = p, rows = rows, obs_var = V, state_var = W,
           diffuse = run(1e7), proper = run(4))
write_json(out, file.path(here, "tvp_var_R.json"), digits = NA, auto_unbox = TRUE)
