# ---------------------------------------------------------------------------
# R reference for the model averaging tests of
# tests/reference_parity/test_hansen_methods_stata_parity.py
#
# Reads textbook_cs.csv and writes hansen_model_average_R.csv: one
# "key,value" row per number. Five candidate regressions of y; selection
# criteria, smoothed AIC / BIC weights, and Mallows and jackknife weights
# from quadprog::solve.QP, with the definitions of the programs that come
# with Hansen's Econometrics (figure28_5.R). The residual variance of the
# Mallows penalty is that of the largest model, n - k in the denominator.
#
# Run (from this directory):  Rscript _generate_hansen_model_average.R
# ---------------------------------------------------------------------------
library(quadprog)
d <- read.csv("textbook_cs.csv")
y <- d$y
n <- length(y)
one <- rep(1, n)
X <- list(
  cbind(one, d$x1),
  cbind(one, d$x1, d$x2),
  cbind(one, d$x1, d$x2, d$d),
  cbind(one, d$x1, d$x2, d$d, d$z1, d$z2),
  cbind(one, d$x1, d$x1^2, d$x2, d$x2^2, d$d, d$z1, d$z2)
)
M <- length(X)
e <- matrix(0, n, M); r <- matrix(0, n, M); kk <- numeric(M)
for (m in 1:M) {
  x <- X[[m]]
  invx <- solve(t(x) %*% x)
  b <- invx %*% (t(x) %*% y)
  e[, m] <- y - x %*% b
  r[, m] <- e[, m] / (1 - rowSums(x * (x %*% invx)))
  kk[m] <- ncol(x)
}
sig <- colMeans(e^2)
bic <- n * log(2 * pi * sig) + kk * log(n)
aic <- n * log(2 * pi * sig) + kk * 2
cv <- colSums(r^2)
wbic <- exp(-(bic - min(bic)) / 2); wbic <- wbic / sum(wbic)
waic <- exp(-(aic - min(aic)) / 2); waic <- waic / sum(waic)
Amat <- t(rbind(matrix(1, 1, M), diag(M), -diag(M)))
bvec <- c(1, rep(0, M), rep(-1, M))
s2 <- sum(e[, M]^2) / (n - kk[M])
wmma <- solve.QP(t(e) %*% e, -kk * s2, Amat, bvec, 1)$solution
wjma <- solve.QP(t(r) %*% r, rep(0, M), Amat, bvec, 1)$solution
out <- file("hansen_model_average_R.csv", "w")
cat("key,value\n", file = out)
emit <- function(key, value) cat(sprintf("%s,%.16e\n", key, value), file = out)
for (m in 1:M) {
  emit(sprintf("aic%d", m), aic[m]); emit(sprintf("bic%d", m), bic[m])
  emit(sprintf("cv%d", m), cv[m]); emit(sprintf("waic%d", m), waic[m])
  emit(sprintf("wbic%d", m), wbic[m]); emit(sprintf("wmma%d", m), wmma[m])
  emit(sprintf("wjma%d", m), wjma[m])
}
close(out)
