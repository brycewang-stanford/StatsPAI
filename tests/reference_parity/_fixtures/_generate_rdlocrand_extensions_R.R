# Reference values for the options of sp.rdrandinf / sp.rdwinselect /
# sp.rddensity exercised by Cattaneo, Idrobo & Titiunik (2024), "A Practical
# Introduction to Regression Discontinuity Designs: Extensions", against
# R rdlocrand 2.0 and rddensity 2.6.
#
# Only deterministic quantities are pinned as equalities: observed
# statistics, large-sample p-values, window counts, binomial p-values.
# Randomization p-values are RNG draws; what is stored for them is the mean
# over 60 seeds, which the Python side compares as a stochastic screen.
library(rdlocrand); library(rddensity); library(jsonlite)
d <- read.csv("rdsenate.csv")
X <- d$margin; Y <- d$vote
q <- function(e) { sink("/dev/null"); on.exit(sink()); suppressWarnings(e) }
det <- function(...) {
  o <- q(rdrandinf(Y, X, reps = 10, ...))
  list(obs_stat = unname(as.numeric(o$obs.stat)),
       asy_pvalue = unname(as.numeric(o$asy.pvalue)),
       Nl = unname(o$sumstats[2, 1]), Nr = unname(o$sumstats[2, 2]))
}
out <- list()
out$tri        <- det(wl = -5, wr = 5, kernel = "triangular")
out$epan       <- det(wl = -5, wr = 5, kernel = "epan")
out$tri_asym   <- det(wl = -3, wr = 5, kernel = "triangular")
out$p1         <- det(wl = -5, wr = 5, p = 1)
out$p2         <- det(wl = -5, wr = 5, p = 2)
out$tri_p1     <- det(wl = -5, wr = 5, p = 1, kernel = "triangular")
out$p1_eval    <- det(wl = -5, wr = 5, p = 1, evall = -2, evalr = 2)
out$null3      <- det(wl = -5, wr = 5, nulltau = 3)
out$ks_null3   <- det(wl = -5, wr = 5, nulltau = 3, statistic = "ksmirnov")
out$placebo_c2 <- det(cutoff = 2, wl = 0.5, wr = 3.5)

# ---- fuzzy: a deterministic take-up rule, so the fixture needs no RNG ----
D <- as.numeric(X >= 0)
flip <- (seq_along(X) %% 4 == 0)          # every fourth unit does not comply
D[flip] <- 1 - D[flip]
o <- q(rdrandinf(Y, X, wl = -5, wr = 5, fuzzy = D, reps = 10))
out$fuzzy_itt <- list(obs_stat = unname(o$obs.stat), asy_pvalue = unname(o$asy.pvalue))
o <- q(rdrandinf(Y, X, wl = -5, wr = 5, fuzzy = c(D, "tsls"), reps = 10))
out$fuzzy_tsls <- list(obs_stat = unname(o$obs.stat), asy_pvalue = unname(o$asy.pvalue))

# ---- rdwinselect on a fixed sequence, large-sample p-values --------------
covs <- cbind(d$class, d$termshouse, d$termssenate)
r <- q(rdwinselect(X, covs, wmin = 0.5, wstep = 0.25, nwindows = 12, approx = TRUE))
out$winselect_approx <- list(
  p_value = unname(r$results[, "p-value"]), variable = unname(r$results[, "Variable"]),
  binom = unname(r$results[, "Bi.test"]),
  Nl = unname(r$results[, "Obs<c"]), Nr = unname(r$results[, "Obs>=c"]),
  w_right = unname(r$results[, "w_right"]), rec_left = r$w_left, rec_right = r$w_right
)
# rdwinselect(..., approx = TRUE, p = 1) is not pinned: rdlocrand 2.0 stops
# with "subscript out of bounds" in its HC2 step on this data. The adjusted
# statistic and its large-sample p-value are pinned through rdrandinf above.

# ---- randomization p-values: mean over 60 seeds (a screen, not parity) ---
z <- d$termshouse
sm <- function(...) mean(sapply(1:60, function(s)
  as.numeric(q(rdrandinf(z, X, wl = -1, wr = 1, seed = s, ...))$p.value)))
bp <- ifelse(abs(X) <= 1, 0.5, NA)
out$seedmean <- list(
  diffmeans = sm(), ranksum = sm(statistic = "ranksum"),
  triangular = sm(kernel = "triangular"), bernoulli = sm(bernoulli = bp),
  p1 = sm(p = 1),
  p1_asy = as.numeric(q(rdrandinf(z, X, wl = -1, wr = 1, p = 1))$asy.pvalue)
)

# ---- rddensity binomial tests with user-set windows ----------------------
b <- q(rddensity(X, binoW = c(1, 2), binoWStep = c(0.5, 1), binoNW = 3))$bino
out$bino_asym <- list(Nl = b$LeftN, Nr = b$RightN, pval = b$pval)
b <- q(rddensity(X, binoW = 0.75, binoNW = 4, binoP = 0.4))$bino
out$bino_p04 <- list(Nl = b$LeftN, Nr = b$RightN, pval = b$pval)

out[["_meta"]] <- list(
  n = nrow(d),
  rdlocrand_version = as.character(packageVersion("rdlocrand")),
  rddensity_version = as.character(packageVersion("rddensity"))
)
write_json(out, "rdlocrand_extensions_R.json", auto_unbox = TRUE, digits = 15,
           pretty = TRUE, na = "null")
cat("wrote rdlocrand_extensions_R.json\n")
