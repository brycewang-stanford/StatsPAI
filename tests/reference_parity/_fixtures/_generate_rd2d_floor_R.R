# Black-box fixture for the clean-room rewrite of sp.rd2d / sp.rd2d_bw
# (statspai.rd.rd2d and its _rd2d_* modules): a small sample (n = 600) on
# which the minimum-sample floors (bwcheck = 50 + p + 1) bind, so the floor
# rules of the selectors and of the estimation windows are exercised. Runs R
# rd2d 1.0.0 (Cattaneo, Titiunik and Yu; GPL-3) only through its documented
# interface; nothing here reads or reproduces its source.
#
#   Rscript tests/reference_parity/_fixtures/_generate_rd2d_floor_R.R
#
# Writes, next to this file:
#   rd2d_floor.csv     y, x1, x2, t (treated above x2 = 0.3 x1), g (30
#                      clusters), takeup, yf (fuzzy outcome); seeded
#   rd2d_floor_R.json  per cell: args, R's $main table and, where stored,
#                      the covariance of the bias-corrected estimates

suppressMessages({
  library(rd2d)
  library(jsonlite)
})
stopifnot(as.character(packageVersion("rd2d")) == "1.0.0")

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)
if (is.null(here)) {
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  here <- dirname(normalizePath(f))
}

set.seed(20261021)
n <- 600
x1 <- runif(n, -1, 1)
x2 <- runif(n, -1, 1)
t <- as.integer(x2 >= 0.3 * x1)
g <- sample.int(30, n, replace = TRUE)
mu <- 0.5 + 0.8 * x1 - 0.6 * x2 + 0.7 * x1 * x2 + 0.5 * x1^2
y <- mu + t * (1 + 0.6 * x1) + (0.4 + 0.2 * abs(x1)) * rnorm(n)
takeup <- as.integer(runif(n) < ifelse(t == 1, 0.85, 0.15))
yf <- mu + 1.5 * takeup + 0.4 * rnorm(n)
write.csv(data.frame(y = y, x1 = x1, x2 = x2, t = t, g = g, takeup = takeup, yf = yf),
          file.path(here, "rd2d_floor.csv"), row.names = FALSE)
d <- read.csv(file.path(here, "rd2d_floor.csv"))
X <- cbind(d$x1, d$x2)
b <- rbind(c(-0.4, -0.12), c(0, 0), c(0.4, 0.12))
D <- sapply(1:3, function(j) sqrt((d$x1 - b[j, 1])^2 + (d$x2 - b[j, 2])^2) * (2 * d$t - 1))

out <- list()
tab <- function(x) lapply(as.data.frame(x), as.numeric)
loc <- function(tag, ...) {
  a <- list(...)
  r <- suppressWarnings(do.call(rd2d, c(list(Y = if (isTRUE(a$fz)) d$yf else d$y, X = X,
    assignment = d$t, b = b, params.cov = "main"), a[setdiff(names(a), "fz")])))
  out[[tag]] <<- list(approach = "location", args = a[setdiff(names(a), c("fuzzy", "cluster"))],
                      main = tab(r$main), cov = unname(as.matrix(r$params.cov$main)))
}
dist <- function(tag, ...) {
  a <- list(...)
  r <- suppressWarnings(do.call(rd2d.distance, c(list(Y = if (isTRUE(a$fz)) d$yf else d$y,
    distance = D, b = b, cbands = FALSE, params.cov = "main"), a[setdiff(names(a), "fz")])))
  out[[tag]] <<- list(approach = "distance", args = a[setdiff(names(a), c("fuzzy", "cluster"))],
                      main = tab(r$main), cov = unname(as.matrix(r$params.cov$main)))
}

loc("L_default")
loc("L_msetwo", bwselect = "msetwo")
loc("L_imserd", bwselect = "imserd")
loc("L_certwo", bwselect = "certwo")
loc("L_user_h01", h = 0.1)
loc("L_rad", kernel_type = "rad")
loc("L_rad_user_h01", kernel_type = "rad", h = 0.1)
loc("L_fuzzy", fz = TRUE, fuzzy = d$takeup)
loc("L_cluster_sep", cluster = d$g, fitmethod = "separate")
loc("L_nostd_p2", stdvars = FALSE, p = 2)
dist("D_default")
dist("D_fuzzy", fz = TRUE, fuzzy = d$takeup)
dist("D_kink_unknown", kink.unknown = TRUE)
dist("D_kink_position", kink.position = 2)
dist("D_imserd", bwselect = "imserd")
dist("D_certwo", bwselect = "certwo")
dist("D_cluster", cluster = d$g)

out[["_meta"]] <- list(n = n, b = unname(b), rd2d = as.character(packageVersion("rd2d")),
                       R = R.version.string)
write_json(out, file.path(here, "rd2d_floor_R.json"), auto_unbox = TRUE, digits = NA)
cat("wrote rd2d_floor_R.json\n")
