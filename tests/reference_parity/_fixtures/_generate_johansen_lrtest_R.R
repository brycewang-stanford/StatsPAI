# urca 1.3-4 reference for sp.johansen_lrtest (and sp.johansen statistics).
# Run from tests/reference_parity/_fixtures; writes johansen_lrtest_R.json.
suppressMessages({library(urca); library(jsonlite)})
z <- read.csv("johansen_lrtest.csv")
out <- list(urca = as.character(packageVersion("urca")))
for (ec in c("none", "const", "trend")) {
  j <- ca.jo(z, type = "trace", ecdet = ec, K = 3, spec = "transitory")
  n <- nrow(j@V)
  pad <- function(M) if (n == 5) rbind(M, 0) else M
  free <- function(M) if (n == 5) rbind(cbind(M, 0), c(rep(0, ncol(M)), 1)) else M
  H <- matrix(c(1, 0, -1, 0, 0, 1, -1, 0, 0, 0, 0, 1), 4, 3)
  b <- blrtest(j, H = free(H), r = 2)
  k1 <- bh5lrtest(j, H = pad(matrix(c(1, 0, -1, 0), 4, 1)), r = 2)
  k2 <- blrtest(j, H = pad(matrix(c(1, 0, -1, 0, 0, 1, -1, 0), 4, 2)), r = 2)
  a <- alrtest(j, A = diag(4)[, 1:3], r = 2)
  a2 <- alrtest(j, A = matrix(c(1, 0, 0, 0, 0, 1, 0, 0), 4, 2), r = 2)
  out[[ec]] <- list(
    trace = rev(j@teststat), lambda = j@lambda,
    beta = list(stat = b@teststat, df = b@pval[2], p = b@pval[1],
                V = b@V[, 1:2] / rep(b@V[1, 1:2], each = n)),
    known1 = list(stat = k1@teststat, df = k1@pval[2], p = k1@pval[1]),
    known2 = list(stat = k2@teststat, df = k2@pval[2], p = k2@pval[1]),
    loading = list(stat = a@teststat, df = a@pval[2], p = a@pval[1]),
    loading2 = list(stat = a2@teststat, df = a2@pval[2], p = a2@pval[1])
  )
}
write_json(out, "johansen_lrtest_R.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
