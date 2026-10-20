# Reference values for sp.lmtp vs R lmtp 1.5.4 (AGPL: used as a black box,
# outputs only); checked in tests/reference_parity/test_lmtp_parity.py.
# Run from this folder:  Rscript _generate_lmtp_R.R
# SL.glm learners, folds = 1, default lmtp_control() (trim 0.999, bound
# 1e-5); policy A + 0.5 at every time point. lmtp_tmle and lmtp_sdr, the
# latter two-period design also with right censoring (C = 1: observed).
suppressMessages({library(lmtp); library(jsonlite)})
set.seed(56); n <- 500
W <- round(rnorm(n), 6); A <- round(1 + 0.5*W + rnorm(n), 6)
Y <- round(1 + A + 0.5*W + rnorm(n), 6); Yb <- as.integer(Y > 2)
write.csv(data.frame(W, A, Y, Yb), "lmtp_point.csv", row.names = FALSE)
set.seed(57)
W <- round(rnorm(n), 6); A1 <- round(1 + 0.5*W + rnorm(n), 6)
L2 <- round(0.4*A1 + 0.3*W + rnorm(n), 6); A2 <- round(0.5*A1 + 0.4*L2 + rnorm(n), 6)
Y <- round(1 + 0.5*A1 + A2 + 0.5*L2 + rnorm(n), 6); Yb <- as.integer(Y > 2.5)
write.csv(data.frame(W, A1, L2, A2, Y, Yb), "lmtp_long.csv", row.names = FALSE)
set.seed(58)
W <- round(rnorm(n), 6); A1 <- round(1 + 0.5*W + rnorm(n), 6)
C1 <- rbinom(n, 1, plogis(2 + 0.5*W - 0.3*A1))
L2 <- round(0.4*A1 + 0.3*W + rnorm(n), 6); A2 <- round(0.5*A1 + 0.4*L2 + rnorm(n), 6)
C2 <- rbinom(n, 1, plogis(2 - 0.4*L2 + 0.3*A2))
Y <- round(1 + 0.5*A1 + A2 + 0.5*L2 + rnorm(n), 6); Yb <- as.integer(Y > 2.5)
L2[C1 == 0] <- NA; A2[C1 == 0] <- NA; C2[C1 == 0] <- 0
Y[C2 == 0] <- NA; Yb[C2 == 0] <- NA
write.csv(data.frame(W, A1, C1, L2, A2, C2, Y, Yb), "lmtp_cens.csv", row.names = FALSE)
sh <- function(data, trt) data[[trt]] + 0.5
p <- read.csv("lmtp_point.csv"); l <- read.csv("lmtp_long.csv"); cc <- read.csv("lmtp_cens.csv")
fit <- function(d, trt, y, tv, type, est = lmtp_tmle, cens = NULL) {
  r <- est(d, trt, y, "W", time_vary = tv, cens = cens, shift = sh, outcome_type = type,
           folds = 1, learners_outcome = "SL.glm", learners_trt = "SL.glm")
  list(est = r$estimate@x, se = r$estimate@std_error, ratios = r$density_ratios)
}
cn <- c("C1", "C2")
out <- list(
  point_continuous = fit(p, "A", "Y", NULL, "continuous"),
  point_binary = fit(p, "A", "Yb", NULL, "binomial"),
  long_continuous = fit(l, c("A1", "A2"), "Y", list(NULL, "L2"), "continuous"),
  long_binary = fit(l, c("A1", "A2"), "Yb", list(NULL, "L2"), "binomial"),
  sdr_point_continuous = fit(p, "A", "Y", NULL, "continuous", lmtp_sdr),
  sdr_point_binary = fit(p, "A", "Yb", NULL, "binomial", lmtp_sdr),
  sdr_long_continuous = fit(l, c("A1", "A2"), "Y", list(NULL, "L2"), "continuous", lmtp_sdr),
  sdr_long_binary = fit(l, c("A1", "A2"), "Yb", list(NULL, "L2"), "binomial", lmtp_sdr),
  cens_continuous = fit(cc, c("A1", "A2"), "Y", list(NULL, "L2"), "continuous", cens = cn),
  cens_binary = fit(cc, c("A1", "A2"), "Yb", list(NULL, "L2"), "binomial", cens = cn),
  sdr_cens_continuous = fit(cc, c("A1", "A2"), "Y", list(NULL, "L2"), "continuous", lmtp_sdr, cn),
  sdr_cens_binary = fit(cc, c("A1", "A2"), "Yb", list(NULL, "L2"), "binomial", lmtp_sdr, cn),
  versions = list(R = R.version.string, lmtp = as.character(packageVersion("lmtp"))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17)), "lmtp_R.json")
