# Reference values for tests/reference_parity/test_statespace_parity.py
#
# Reads the committed synthetic file statespace.csv and the system matrices
# in statespace_spec.json, runs KFAS on each model, writes statespace_R.json.
# Requires: KFAS, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_statespace_R.R
#
# Mapping. The spec is X_t = F_t X_{t-1} + V_t, Y_t = A_t + G_t X_t + W_t with
# X_0 ~ (x0, P0). KFAS writes alpha_{t+1} = T_t alpha_t + eta_t and starts
# from alpha_1 ~ (a1, P1), so T_t = F_{t+1}, Q^KFAS_t = Q_{t+1},
# a1 = F_1 x0, P1 = F_1 P0 F_1' + Q_1. KFAS has no observation intercept:
# A_t is subtracted from y. For several observables KFAS processes the
# elements of y one at a time, so its v and F are not the joint prediction
# errors; they are exported for single-observable models only.
suppressMessages({library(KFAS); library(jsonlite)})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "statespace.csv"))
spec = fromJSON(file.path(here, "statespace_spec.json"))
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
  P0 = if (is.null(s$P0)) matrix(solve(diag(m * m) - kronecker(F1, F1), as.vector(Q1)), m, m)
       else matrix(s$P0, m, m)
  a1 = F1 %*% s$x0
  P1 = F1 %*% P0 %*% t(F1) + Q1
  lead = c(2:n, n)
  yy = y - A
  mod = SSModel(yy ~ -1 + SSMcustom(Z = Ga, T = Fa[, , lead, drop = FALSE], R = diag(m),
                                    Q = Qa[, , lead, drop = FALSE], a1 = a1, P1 = P1,
                                    P1inf = matrix(0, m, m)), H = Ra)
  k = KFS(mod, filtering = "state", smoothing = "state")
  out = list(predicted = matrix(k$a[1:n, ], n, m),
             P_pred = aperm(k$P[, , 1:n, drop = FALSE], c(3, 1, 2)),
             filtered = matrix(k$att, n, m), P_filt = aperm(k$Ptt, c(3, 1, 2)),
             smoothed = matrix(k$alphahat, n, m), P_smooth = aperm(k$V, c(3, 1, 2)),
             loglik = as.numeric(logLik(mod)))
  if (p == 1) { out$v = as.numeric(k$v); out$S = as.numeric(k$F) }
  out
}
out = list(versions = list(R = R.version.string, KFAS = as.character(packageVersion("KFAS"))))
out$level = run(spec$level, d$level)
out$ar2 = run(spec$ar2, d$ar2)
out$biv = run(spec$biv, d[, c("biv1", "biv2")])
out$tvp = run(spec$tvp, d$tvp_y)
out$mixed = run(spec$mixed, d[, c("mixed1", "mixed2", "mixed3")])
out$tv = run(spec$tv, d$tv)
write_json(out, file.path(here, "statespace_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
