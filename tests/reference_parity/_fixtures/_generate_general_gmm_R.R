#!/usr/bin/env Rscript
# R ``gmm`` reference values for StatsPAI's general moment-condition GMM.
#
# ``gmm::gmm`` (Chaussé 2010, JSS 34(11)) is the canonical R implementation and the
# one Stata's ``gmm`` command is usually checked against.  The linear IV
# moment condition ``E[z (y - x'b)] = 0`` is used because it has a
# closed-form answer, so any disagreement is an implementation difference
# rather than two optimisers stopping in different places.
#
# Writes:  general_gmm_data.csv   (the sample, full double precision)
#          general_gmm_R.json     (coefficients, SEs, J test, by weight type)
#
# Requires: gmm (>= 1.8), jsonlite
# Run:      Rscript _generate_general_gmm_R.R   (from this directory)

suppressMessages({
  library(gmm)
  library(jsonlite)
})

here <- tryCatch({
  a <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f)) dirname(normalizePath(sub("^--file=", "", f[1]))) else getwd()
}, error = function(e) getwd())

set.seed(20260731)
n <- 400
z1 <- rnorm(n)
z2 <- rnorm(n)
z3 <- rnorm(n)
u <- rnorm(n)
x1 <- 0.8 * z1 + 0.5 * z2 - 0.3 * z3 + u + rnorm(n)
y <- 1.5 + 2.0 * x1 + u + rnorm(n)
df <- data.frame(y = y, x1 = x1, z1 = z1, z2 = z2, z3 = z3)
write.csv(format(df, digits = 17), file.path(here, "general_gmm_data.csv"),
          row.names = FALSE, quote = FALSE)

X <- cbind(1, x1)          # regressors, intercept first
Zm <- cbind(1, z1, z2, z3)  # instruments -> 4 moments, 2 parameters

g <- function(theta, dat) {
  resid <- as.vector(dat[, "y"] - cbind(1, dat[, "x1"]) %*% theta)
  cbind(1, dat[, "z1"], dat[, "z2"], dat[, "z3"]) * resid
}
dat <- as.matrix(df)
start <- c(0, 0)

# gmm::gmm defaults to optim()'s stock tolerances, which on this design stop
# ~1e-4 away from the optimum -- far enough to swamp any implementation
# difference.  Every fit below therefore uses a tightened control list, and
# the closed-form linear-IV solution is emitted alongside as an independent
# check that the tightened values are the right ones.
CTRL <- list(reltol = 1e-14, abstol = 1e-14, maxit = 100000)

pack <- function(fit) {
  s <- summary(fit)
  co <- s$coefficients
  jt <- s$stest$test
  list(
    coef = as.list(setNames(unname(co[, 1]), c("_cons", "x1"))),
    se = as.list(setNames(unname(co[, 2]), c("_cons", "x1"))),
    J = as.numeric(jt[1]),
    J_df = ncol(Zm) - length(start),
    J_p = as.numeric(jt[2])
  )
}

out <- list(
  `_meta` = list(
    R_version = R.version.string,
    gmm_version = as.character(packageVersion("gmm")),
    moment = "E[z (y - x'b)] = 0, z = (1, z1, z2, z3), x = (1, x1)",
    n = n,
    note = paste(
      "gmm::gmm centres the moment covariance by default (centeredVcov=TRUE);",
      "the identity-weight rows fix the first-step weight so the point",
      "estimate is the closed-form 2SLS-with-identity-weight solution."
    )
  )
)

# Two-step efficient GMM with the i.i.d. (HC) covariance estimator.
fit_iid <- gmm(g, dat, t0 = start, vcov = "iid", wmatrix = "optimal",
               control = CTRL)
out$twostep_iid <- pack(fit_iid)

# Two-step efficient GMM with a heteroskedasticity-robust covariance.
fit_hc <- gmm(g, dat, t0 = start, vcov = "MDS", wmatrix = "optimal",
              control = CTRL)
out$twostep_mds <- pack(fit_hc)

# Iterated GMM.
fit_it <- gmm(g, dat, t0 = start, vcov = "MDS", wmatrix = "optimal",
              type = "iterative", itermax = 200, control = CTRL)
out$iterated_mds <- pack(fit_it)

# Continuously-updated estimator.
fit_cue <- gmm(g, dat, t0 = start, vcov = "MDS", wmatrix = "optimal",
               type = "cue", control = CTRL)
out$cue_mds <- pack(fit_cue)

# HAC (Bartlett kernel, fixed bandwidth 2) -- exercises the kernel path.
fit_hac <- gmm(g, dat, t0 = start, vcov = "HAC", kernel = "Bartlett",
               bw = function(...) 2, prewhite = 0, wmatrix = "optimal",
               control = CTRL)
out$twostep_hac_bartlett2 <- pack(fit_hac)

# Independent closed-form two-step linear IV, written with plain matrix
# algebra rather than gmm::gmm, as a second anchor on the same numbers.
b1 <- solve(crossprod(t(Zm) %*% X), t(t(Zm) %*% X) %*% (t(Zm) %*% y))
e1 <- as.vector(y - X %*% b1)
Gc <- scale(Zm * e1, scale = FALSE)
Wo <- solve(crossprod(Gc) / n)
A <- t(Zm) %*% X
b2 <- solve(t(A) %*% Wo %*% A, t(A) %*% Wo %*% (t(Zm) %*% y))
out$closed_form_twostep_centered <- list(
  coef = list(`_cons` = b2[1], x1 = b2[2]),
  note = "plain matrix algebra, no gmm::gmm involved"
)

write_json(out, file.path(here, "general_gmm_R.json"),
           pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "null")
cat(sprintf("wrote general_gmm_R.json: %d specs\n", length(out) - 1L))
