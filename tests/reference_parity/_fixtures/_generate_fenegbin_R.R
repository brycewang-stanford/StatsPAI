# Reference values for test_fenegbin_parity.py. R fixest and MASS (GPL)
# are used as black boxes: only their output is read. fixef.tol is set
# to 1e-11 so fixest's own stopping rule does not dominate the comparison.
suppressPackageStartupMessages({
  library(fixest)
  library(MASS)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
d <- read.csv(file.path(here, "fenegbin_data.csv"))
setFixest_nthreads(1)
num <- function(x) unname(as.numeric(x))
fit <- function(m, ...) {
  s <- summary(m, ...)
  list(coef = num(coef(m)), se = num(se(s)), theta = num(m$theta),
       nobs = nobs(m), loglik = num(logLik(m)))
}
m2 <- fenegbin(y ~ x1 + x2 | g + t, d, fixef.tol = 1e-11)
m1 <- fenegbin(y ~ x1 + x2 | g, d, fixef.tol = 1e-11)
m0 <- fenegbin(y ~ x1 + x2, d)
out <- list(
  versions = list(fixest = as.character(packageVersion("fixest")),
                  MASS = as.character(packageVersion("MASS")),
                  R = R.version.string),
  two_iid = fit(m2),
  two_hetero = fit(m2, vcov = "hetero"),
  two_cl = fit(m2, cluster = ~cl),
  two_cl_g = fit(m2, cluster = ~g),
  two_cl_gt = fit(m2, cluster = ~g + t),
  one_cl_g = fit(m1, cluster = ~g),
  none_iid = fit(m0)
)
# Independent check of the maximum: MASS::glm.nb with the effects as
# dummies on fixest's estimation sample.
keep <- obs(m2)
dd <- d[keep, ]
dd$g <- factor(dd$g); dd$t <- factor(dd$t)
nb <- glm.nb(y ~ x1 + x2 + g + t, dd, control = glm.control(epsilon = 1e-14, maxit = 200))
out$glmnb <- list(coef = num(coef(nb)[c("x1", "x2")]), theta = num(nb$theta),
                  loglik = num(logLik(nb)))
# fenegbin takes no weights: the weighted maximum comes from glm.nb alone
nbw <- glm.nb(y ~ x1 + x2 + g + t, dd, weights = w,
              control = glm.control(epsilon = 1e-14, maxit = 200))
out$glmnb_w <- list(coef = num(coef(nbw)[c("x1", "x2")]), theta = num(nbw$theta))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           file.path(here, "fenegbin_R.json"))
