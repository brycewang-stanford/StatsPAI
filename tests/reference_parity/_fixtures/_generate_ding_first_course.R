# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_ding_first_course_parity.py
#
# Requires: R with sensitivitymw, sandwich, Matching and jsonlite.
# Run:      Rscript _generate_ding_first_course.R      (from this directory)
#
# Peng Ding, "A First Course in Causal Inference" (2024), is the source of
# the four computations checked here. The data are simulated in this script
# (the book's own data are not redistributed) and written to
# ding_first_course_*.csv, which the test reads.
#
#   pairs : Rosenbaum's sensitivity analysis of the mean pair difference,
#           sensitivitymw::senmw(method = "t")              (chapter 19)
#   iv    : the Fieller-Anderson-Rubin test with the Eicker-Huber-White
#           variance, sandwich::vcovHC on the reduced form  (chapters 21, 23)
#   obs   : the Horvitz-Thompson and Hajek weighting estimators of the
#           effect on the treated, from their definitions    (chapter 13),
#           and Matching::Match with its two variance estimators (chapter 15)
#   rct   : Lin's estimator and its HC2 t-ratio              (chapters 6, 8)
#
# sensitivitymw and Matching are GPL: they are run here as programs and only
# their printed numbers are stored.
# ---------------------------------------------------------------------------
suppressMessages({library(sensitivitymw); library(sandwich); library(Matching); library(jsonlite)})
set.seed(20261005)
wr <- function(d, f) { write.csv(format(d, digits = 17), f, row.names = FALSE, quote = FALSE); read.csv(f) }
out <- list()

## pairs -----------------------------------------------------------------
I <- 150
ctl <- rnorm(I, 10, 3)
pairs <- wr(data.frame(treated = ctl + 0.6 + rt(I, 4), control = ctl), "ding_first_course_pairs.csv")
gam <- c(1, 1.2, 1.5, 2, 3)
out$pairs_gamma <- gam
out$pairs_p_upper <- sapply(gam, function(g) senmw(as.matrix(pairs), gamma = g, method = "t")$pval)

## iv --------------------------------------------------------------------
n <- 600
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.4)
z1 <- rnorm(n); z2 <- rbinom(n, 1, 0.5)
u <- rnorm(n)
d <- 0.35 * z1 + 0.25 * z2 + 0.3 * x1 + 0.6 * u + rnorm(n)
y <- 1 + 0.5 * d - 0.4 * x1 + 0.2 * x2 + (0.5 + abs(z1)) * (0.7 * u + rnorm(n))
iv <- wr(data.frame(y = y, d = d, z1 = z1, z2 = z2, x1 = x1, x2 = x2), "ding_first_course_iv.csv")
ar <- function(b0, type, inst) {
  f <- as.formula(paste("I(y - b0 * d) ~", inst, "+ x1 + x2"))
  fit <- lm(f, iv)
  idx <- grep("^z", names(coef(fit)))
  V <- if (type == "classic") vcov(fit) else vcovHC(fit, type = type)
  b <- coef(fit)[idx]
  Fst <- as.numeric(t(b) %*% solve(V[idx, idx]) %*% b) / length(idx)
  c(Fst, pf(Fst, length(idx), fit$df.residual, lower.tail = FALSE))
}
types <- c("classic", "HC0", "HC1", "HC2", "HC3")
out$iv_types <- types
out$iv_b0 <- c(0, 0.5)
out$iv_one <- lapply(types, function(ty) sapply(c(0, 0.5), function(b) ar(b, ty, "z1")))
out$iv_two <- lapply(types, function(ty) sapply(c(0, 0.5), function(b) ar(b, ty, "z1 + z2")))
crit <- qf(0.95, 1, n - 4)
g <- function(b) ar(b, "HC3", "z1")[1] - crit
b2sls <- coef(lm(y ~ z1 + x1 + x2, iv))["z1"] / coef(lm(d ~ z1 + x1 + x2, iv))["z1"]
out$iv_ci_hc3_one <- c(uniroot(g, c(b2sls - 3, b2sls), tol = 1e-13)$root,
                       uniroot(g, c(b2sls, b2sls + 3), tol = 1e-13)$root)

## obs -------------------------------------------------------------------
n <- 800
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rbinom(n, 1, 0.5)
z <- rbinom(n, 1, plogis(-0.3 + 0.8 * x1 - 0.5 * x2 + 0.4 * x3))
y <- 20 + 1.5 * z + 2 * x1 + x2 + z * x1 + rnorm(n)
obs <- wr(data.frame(y = y, z = z, x1 = x1, x2 = x2, x3 = x3), "ding_first_course_obs.csv")
with(obs, {
  x <- cbind(x1, x2, x3)
  # iterate the logit to convergence: the default stops at 1e-8 in the
  # deviance, which leaves the Horvitz-Thompson sums good to 1e-7 only
  ps <- glm(z ~ x, family = binomial,
            control = glm.control(epsilon = 1e-14, maxit = 100))$fitted.values
  odds <- ps / (1 - ps); nn <- length(z); n1 <- sum(z); n0 <- nn - n1
  out$obs_att_ht <<- mean(y[z == 1]) - mean(odds * (1 - z) * y) * nn / n1
  out$obs_att_hajek <<- mean(y[z == 1]) - mean(odds * (1 - z) * y) / mean(odds * (1 - z))
  out$obs_atc_ht <<- mean(z * y / odds) * nn / n0 - mean(y[z == 0])
  out$obs_ate_ht <<- mean(z * y / ps - (1 - z) * y / (1 - ps))
  m0 <- Match(Y = y, Tr = z, X = x, BiasAdjust = TRUE)
  m2 <- Match(Y = y, Tr = z, X = x, BiasAdjust = TRUE, Var.calc = 2)
  out$obs_match <<- c(m0$est, m0$se, m2$se)
})

## rct -------------------------------------------------------------------
n <- 200
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.3)
z <- sample(rep(c(1, 0), c(70, 130)))
y <- 1 + z * (0.5 + x1) + x1 - x2 + rnorm(n)
rct <- wr(data.frame(y = y, z = z, x1 = x1, x2 = x2), "ding_first_course_rct.csv")
with(rct, {
  xc <- scale(cbind(x1, x2), scale = FALSE)
  fit <- lm(y ~ z * xc)
  out$rct_lin <<- c(coef(fit)[2], sqrt(vcovHC(fit, type = "HC2")[2, 2]))
  out$rct_welch_t <<- unname(t.test(y[z == 1], y[z == 0])$statistic)
  out$rct_wilcox_W <<- unname(wilcox.test(y[z == 1], y[z == 0])$statistic)
})

out$versions <- list(R = R.version.string,
                     sensitivitymw = as.character(packageVersion("sensitivitymw")),
                     sandwich = as.character(packageVersion("sandwich")),
                     Matching = as.character(packageVersion("Matching")))
write_json(out, "ding_first_course_R.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
