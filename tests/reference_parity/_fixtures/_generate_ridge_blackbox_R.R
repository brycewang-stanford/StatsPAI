# Black-box outputs of R MASS::lm.ridge (GPL-2 | GPL-3; run, not read) for
# the clean-room rewrite of sp.ridge (docs/dev/cleanroom/ridge.md):
# longley with an intercept, and a synthetic file with and without one.
# Run from this directory: Rscript _generate_ridge_blackbox_R.R
suppressMessages(library(MASS))
set.seed(11); n <- 40
d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = runif(n))
d$x4 <- d$x1 + 0.3 * rnorm(n)
d$y <- 1 + d$x1 - 0.5 * d$x2 + 0.2 * d$x4 + rnorm(n)
write.csv(d, "ridge_blackbox_data.csv", row.names = FALSE)
lo <- longley; names(lo)[1] <- "y"; names(lo) <- make.names(names(lo))
write.csv(lo, "ridge_blackbox_longley.csv", row.names = FALSE)
lam <- c(0, 0.01, 0.1, 1, 5, 25)
fits <- list(longley = list(f = y ~ ., d = lo),
             syn = list(f = y ~ x1 + x2 + x3 + x4, d = d),
             syn_noint = list(f = y ~ 0 + x1 + x2 + x3 + x4, d = d))
out <- lapply(fits, function(z) {
  r <- lm.ridge(z$f, z$d, lambda = lam)
  list(lambda = lam, names = colnames(coef(r)), coef = unname(coef(r)),
       gcv = unname(r$GCV), kHKB = r$kHKB, kLW = r$kLW, scales = unname(r$scales))
})
jsonlite::write_json(out, "ridge_blackbox_R.json", digits = NA, auto_unbox = TRUE)
cat("MASS", as.character(packageVersion("MASS")), "\n")
