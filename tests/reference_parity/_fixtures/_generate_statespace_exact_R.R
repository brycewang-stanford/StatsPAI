# Reference values for the exact diffuse tests in
# tests/reference_parity/test_statespace_parity.py
#
# Reads statespace_exact.csv and statespace_exact_spec.json, runs KFAS with an
# exact diffuse initial state (P1inf) on each model, writes
# statespace_exact_R.json. Requires: KFAS, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_statespace_exact_R.R
#
# Mapping. The spec is X_t = F_t X_{t-1} + V_t, Y_t = A_t + G_t X_t + W_t with
# X_0 ~ (x0, P0* + kappa P0inf), P0inf = diag(diffuse), P0* the stationary
# covariance of the other states. KFAS starts from alpha_1, so T_t = F_{t+1},
# Q^KFAS_t = Q_{t+1}, a1 = F_1 x0, P1 = F_1 P0* F_1' + Q_1 and
# P1inf = F_1 P0inf F_1'. KFAS wants P1inf diagonal with ones and P1 zero in
# the rows and columns of diffuse states; in every model here F_1 P0inf F_1'
# has the same column space as diag(diffuse), which is all the limit depends
# on, so P1inf = diag(diffuse) is passed and P1 is zeroed there. The finite
# part of the predicted covariance at the first dates therefore differs from
# the one StatsPAI reports by a matrix inside the diffuse space; P is exported
# and compared from the end of the diffuse period on (d + 1 in KFAS's count).
# The diffuse log-likelihood shifts by -0.5 log pdet(F_1 P0inf F_1'), which is
# zero except in model `tv` (-log|det F_1|).
suppressMessages({library(KFAS); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "statespace_exact.csv"))
spec = fromJSON(file.path(here, "statespace_exact_spec.json"))
stack = function(M, n) {
  M = as.array(M)
  if (length(dim(M)) == 2) array(M, c(dim(M), n)) else aperm(M, c(2, 3, 1))
}
sl = function(a, i) matrix(a[, , i], dim(a)[1], dim(a)[2])
run = function(s, y) {
  y = as.matrix(y); n = nrow(y); p = ncol(y)
  Fa = stack(s$F, n); Qa = stack(s$Q, n); Ra = stack(s$R, n); Ga = stack(s$G, n)
  m = dim(Fa)[1]
  A = if (is.matrix(s$A)) s$A else matrix(s$A, n, p, byrow = TRUE)
  F1 = sl(Fa, 1); Q1 = sl(Qa, 1)
  dif = as.logical(s$diffuse); keep = which(!dif)
  P0 = matrix(0, m, m)
  if (length(keep) > 0) {
    Fk = F1[keep, keep, drop = FALSE]; Qk = Q1[keep, keep, drop = FALSE]
    k = length(keep)
    P0[keep, keep] = matrix(solve(diag(k * k) - kronecker(Fk, Fk), as.vector(Qk)), k, k)
  }
  a1 = F1 %*% s$x0
  P1 = F1 %*% P0 %*% t(F1) + Q1
  P1[dif, ] = 0; P1[, dif] = 0
  lead = c(2:n, n)
  yy = y - A
  mod = SSModel(yy ~ -1 + SSMcustom(Z = Ga, T = Fa[, , lead, drop = FALSE], R = diag(m),
                                    Q = Qa[, , lead, drop = FALSE], a1 = a1, P1 = P1,
                                    P1inf = diag(as.numeric(dif), m)), H = Ra)
  k = KFS(mod, filtering = "state", smoothing = "state")
  list(d = k$d, j = k$j,
       predicted = matrix(k$a[1:n, ], n, m),
       P_pred = aperm(k$P[, , 1:n, drop = FALSE], c(3, 1, 2)),
       Pinf_pred = aperm(k$Pinf, c(3, 1, 2)),
       filtered = matrix(k$att, n, m), P_filt = aperm(k$Ptt, c(3, 1, 2)),
       smoothed = matrix(k$alphahat, n, m), P_smooth = aperm(k$V, c(3, 1, 2)),
       loglik = as.numeric(logLik(mod)),
       loglik_marginal = as.numeric(logLik(mod, marginal = TRUE)),
       kfs_loglik = k$logLik)
}
out = list(versions = list(R = R.version.string, KFAS = as.character(packageVersion("KFAS"))))
out$level = run(spec$level, d$level)
out$trend = run(spec$trend, d$trend)
out$tvp = run(spec$tvp, d$tvp_y)
out$mixed = run(spec$mixed, d$mixed)
out$biv = run(spec$biv, d[, c("biv1", "biv2")])
out$tv = run(spec$tv, d$tv)
write_json(out, file.path(here, "statespace_exact_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
