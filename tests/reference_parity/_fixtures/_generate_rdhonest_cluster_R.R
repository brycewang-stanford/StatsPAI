# Reference values for sp.rd_honest(cluster=) vs R RDHonest(clusterid=).
# Run from this directory:  Rscript _generate_rdhonest_cluster_R.R
# Cells: M and h fixed (clustered SE formula), M fixed with h chosen
# (Moulton-corrected bandwidth search), and the free cell (rule-of-thumb M).
library(RDHonest); library(jsonlite)
set.seed(20260930)
G <- 150; per <- 20
cl <- rep(seq_len(G), each = per)
xc <- runif(G, -1, 1)
x <- pmin(pmax(xc[cl] + rnorm(G * per, 0, 0.15), -1), 1)
u <- rnorm(G, 0, 0.5)[cl]
y <- 0.5 * x + 1.2 * x^2 + 1.0 * (x >= 0) + u + rnorm(G * per, 0, 0.5)
d <- data.frame(y = y, x = x, cl = cl)
write.csv(d, "rdhonest_cluster.csv", row.names = FALSE)
d <- read.csv("rdhonest_cluster.csv")

grab <- function(r) {
  co <- r$coefficients
  list(estimate = unname(co$estimate), se = unname(co$std.error),
       bias = unname(co$maximum.bias), ci_lower = unname(co$conf.low),
       ci_upper = unname(co$conf.high), h = unname(co$bandwidth),
       M = unname(co$M))
}
out <- list()
for (M in c(1, 3)) for (h in c(0.3, 0.6)) {
  out[[sprintf("fixed_M%g_h%g", M, h)]] <- grab(
    RDHonest(y ~ x, data = d, M = M, h = h, kern = "triangular", clusterid = cl, se.method = "EHW"))
}
for (crit in c("MSE", "FLCI")) {
  out[[sprintf("bwsel_%s", crit)]] <- grab(
    RDHonest(y ~ x, data = d, M = 2.4, opt.criterion = crit,
             kern = "triangular", clusterid = cl, se.method = "EHW"))
}
out[["free_MSE"]] <- grab(RDHonest(y ~ x, data = d, kern = "triangular",
                                   clusterid = cl, se.method = "EHW"))
out[["unclustered_fixed_M1_h0.3"]] <- grab(
  RDHonest(y ~ x, data = d, M = 1, h = 0.3, kern = "triangular"))
out[["_version"]] <- as.character(packageVersion("RDHonest"))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           "rdhonest_cluster_R.json")
