# Black-box fixture for statspai.core._binned_kde (clean-room, 2026-10-20).
# Runs R's bw.nrd0 / density() (R base, GPL) and dineq::rif (GPL-3) as black
# boxes; no R source was read. Output: binned_kde_blackbox_R.json.
#   Rscript _generate_binned_kde_blackbox_R.R
suppressPackageStartupMessages({ library(jsonlite); library(dineq) })
set.seed(20261020)
ds <- list(
  normal200 = rnorm(200),
  exp57 = rexp(57) * 3 + 1,
  iqr0 = c(-20, rep(0, 98), 20),
  const10 = rep(3, 10),
  zero5 = rep(0, 5),
  two = c(1, 2),
  t300 = rt(300, 3),
  pois300 = rpois(300, 2),
  gap = c(rnorm(50), rnorm(50, 40))
)
thin <- function(v) v[seq(1, length(v), by = 7)]
out <- list(R_version = R.version.string,
            dineq_version = as.character(packageVersion("dineq")),
            density = list(), rif = list())
for (nm in names(ds)) {
  x <- ds[[nm]]
  rec <- list(x = x, bw = bw.nrd0(x))
  d <- density(x); rec$x512 <- thin(d$x); rec$y512 <- thin(d$y)
  for (nn in c(100, 1000)) {
    dd <- density(x, n = nn)
    rec[[paste0("x", nn)]] <- thin(dd$x); rec[[paste0("y", nn)]] <- thin(dd$y)
  }
  rec$y_cut1 <- thin(density(x, cut = 1)$y)
  rec$y_bw037 <- thin(density(x, bw = 0.37)$y)
  w <- runif(length(x)); w <- w / sum(w)
  rec$w <- w; rec$y_w <- thin(suppressWarnings(density(x, weights = w))$y)
  out$density[[nm]] <- rec
}
mk <- function(x, w, tau) {
  r <- suppressWarnings(rif(x, weights = w, method = "quantile", quantile = tau))
  list(x = x, w = if (is.null(w)) NULL else w, tau = tau, rif = r)
}
set.seed(7)
out$rif$pos <- mk(rexp(150) * 10 + 1, NULL, 0.5)
out$rif$neg <- mk(rnorm(150), NULL, 0.3)
out$rif$negw <- mk(rnorm(120), runif(120, 0.5, 2), 0.7)
out$rif$posw <- mk(rlnorm(200), rpois(200, 3) + 1, 0.2)
out$rif$mixed0 <- mk(c(rep(0, 20), rexp(80)), NULL, 0.5)
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, null = "null"),
           "binned_kde_blackbox_R.json")
