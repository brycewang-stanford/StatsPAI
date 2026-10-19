# pcalg 2.7.12 (GPL >= 2, used as a black box: outputs only) reference for
# sp.ida: 40 random linear-Gaussian DAGs (randomDAG, rmvDAG, n = 300), their
# CPDAGs (dag2cpdag), sample covariances, and ida(method = "local" / "global").
#   cd tests/reference_parity/_fixtures && Rscript _generate_ida_pcalg.R
suppressMessages(library(pcalg)); library(jsonlite)
set.seed(7)
out <- list()
for (k in 1:40) {
  p <- sample(5:8, 1)
  g <- randomDAG(p, prob = 0.4, lB = 0.3, uB = 1)
  X <- rmvDAG(300, g)
  cp <- dag2cpdag(g)
  A <- as(cp, "matrix")  # A[i, j] = 1: edge i -> j (both entries: undirected)
  A[A != 0] <- 1
  S <- cov(X)
  x <- sample(1:p, 1); y <- sample(setdiff(1:p, x), 1)
  gN <- as(A, "graphNEL")
  out[[k]] <- list(A = unname(A), S = unname(S), x = x - 1, y = y - 1,
                   local = as.numeric(ida(x, y, S, gN, method = "local", type = "cpdag")),
                   global = as.numeric(ida(x, y, S, gN, method = "global", type = "cpdag")))
}
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE), "ida_pcalg.json")
