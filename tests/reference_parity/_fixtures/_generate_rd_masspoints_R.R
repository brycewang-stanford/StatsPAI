# Reference values for sp.rdrobust / sp.rdbwselect (masspoints=) vs R
# rdrobust 4.0.0 on a running variable with heavy ties (x on a 0.02 grid).
# Run from this directory:  Rscript _generate_rd_masspoints_R.R
library(rdrobust); library(jsonlite)
set.seed(20260930)
n <- 1500
x <- round(runif(n, -1, 1) / 0.02) * 0.02
y <- 0.6 * x - 0.4 * x^2 + 0.7 * (x >= 0) + rnorm(n, 0, 0.5)
write.csv(data.frame(y = y, x = x), "rd_masspoints.csv", row.names = FALSE)
d <- read.csv("rd_masspoints.csv")
out <- list()
for (mp in c("adjust", "check", "off")) {
  for (bs in c("mserd", "msetwo", "cerrd")) {
    b <- suppressWarnings(rdbwselect(d$y, d$x, masspoints = mp, bwselect = bs))
    out[[sprintf("bw_%s_%s", mp, bs)]] <- list(
      h_left = unname(b$bws[1, 1]), h_right = unname(b$bws[1, 2]),
      b_left = unname(b$bws[1, 3]), b_right = unname(b$bws[1, 4]))
  }
  r <- suppressWarnings(rdrobust(d$y, d$x, masspoints = mp))
  out[[sprintf("rd_%s", mp)]] <- list(
    h = unname(r$bws[1, 1]), b = unname(r$bws[2, 1]),
    coef = unname(r$coef[1, 1]), se_conv = unname(r$se[1, 1]),
    coef_bc = unname(r$coef[2, 1]), se_rb = unname(r$se[3, 1]))
}
out[["_version"]] <- as.character(packageVersion("rdrobust"))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           "rd_masspoints_R.json")
